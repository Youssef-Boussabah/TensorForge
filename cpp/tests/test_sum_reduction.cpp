// Dependency-free C++ test for the sum reduction and its H6 traversal
// dispatch (Phase H, milestone H6). No GoogleTest / Catch2 — a plain
// executable that prints failures and returns a nonzero exit code if any
// check fails, so CTest reports pass/fail.
//
// This binary compiles cpp/src/reduction.cpp (plus storage.cpp and
// error.cpp) directly, so it can reach BOTH the hidden
// ``tf::reduce_prefers_contiguous_blocks`` predicate and the two hidden
// traversals it picks between, as well as the exported ``tf_core_sum``
// wrapper they live inside.
//
// What it proves, at the layer where the property is actually decided —
// i.e. without the Python wrapper, the ctypes boundary, or NumPy anywhere
// in the picture:
//
//   1. The predicate is exactly the documented three conditions, is total
//      (it answers for rank-0, unit dimensions, transposes, narrows,
//      non-unit strides, broadcast stride-0 layouts, non-adjacent reduced
//      axes, and "nothing reduced" alike), and computes a factorization
//      whose product is the element count.
//   2. The two traversals write **identical bits** for every reduction the
//      predicate accepts, over an IEEE-754 sweep that includes signed
//      zeros in every position, infinities, denormals, the smallest
//      normal, the largest finite magnitudes, quiet NaNs of both signs
//      with distinct payloads, signaling NaNs, and a NaN manufactured by
//      the arithmetic itself. The one deliberate exception is stated in
//      (3).
//   3. **NaN payload characterization, asserted in neither direction.**
//      When two or more NaNs are accumulated into one destination cell the
//      two paths may select different payload bits, because x86-64's ADDSD
//      returns the destination operand's NaN and which addend the compiler
//      places there is an instruction-selection decision C++ cannot
//      express. What this file *does* assert is the part that is
//      contractual: NaNs appear in exactly the same positions on both
//      paths, and every NaN either path produces is quiet. A build whose
//      payloads agree and a build whose payloads differ are equally
//      conforming.
//   4. Both paths honor the export's **accumulate-into** contract: a
//      non-zero destination is added to, not overwritten, and both paths
//      produce the same bits when they do it.
//   5. The exported wrapper's rank-0 path and its generic fallback still
//      behave as they always have.
//
// Bit comparison, never a tolerance: the entire question is whether the
// optimized traversal reproduces the reference's bits, and ``==`` on
// doubles cannot see -0.0 versus +0.0 and calls every NaN unequal to
// itself.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "tf_internal.h"
#include "tf_reduction_internal.h"

TF_EXPORT void* tf_storage_create(std::int64_t size);
TF_EXPORT void tf_storage_destroy(void* handle);
TF_EXPORT void tf_storage_copy_from(void* handle, const double* src);
TF_EXPORT void tf_storage_copy_to(const void* handle, double* dst);
TF_EXPORT void tf_core_sum(
    const void* src_handle, void* dst_handle,
    const std::int64_t* shape, const std::int64_t* in_strides,
    const std::int64_t* out_strides, std::int64_t offset, std::int64_t ndim);
TF_EXPORT int tf_last_error_code();
TF_EXPORT void tf_clear_error();

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what);
    }
}

std::uint64_t bits(double value) {
    std::uint64_t out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

double from_bits(std::uint64_t pattern) {
    double out;
    std::memcpy(&out, &pattern, sizeof(out));
    return out;
}

bool is_nan(double value) {
    const std::uint64_t raw = bits(value);
    return (raw & 0x7FF0000000000000ull) == 0x7FF0000000000000ull &&
           (raw & 0x000FFFFFFFFFFFFFull) != 0;
}

// A quiet NaN has the most significant mantissa bit set.
bool is_quiet_nan(double value) {
    return is_nan(value) && (bits(value) & 0x0008000000000000ull) != 0;
}

// The IEEE-754 sweep, deliberately including the values a reduction can
// mishandle: both zeros (whose sign depends on the initial accumulator),
// the infinities (which manufacture a NaN when they meet), the extremes of
// the exponent range, and four distinct NaNs.
const std::uint64_t kQuietNanA = 0x7FF800DEADBEEF01ull;
const std::uint64_t kQuietNanB = 0x7FF8000000000042ull;
const std::uint64_t kQuietNanNeg = 0xFFF80000000000AAull;
const std::uint64_t kSignalingNan = 0x7FF0000000000001ull;

const std::uint64_t kPatterns[] = {
    0x0000000000000000ull,  // +0.0
    0x8000000000000000ull,  // -0.0
    0x3FF0000000000000ull,  // 1.0
    0xBFF0000000000000ull,  // -1.0
    0x0000000000000001ull,  // smallest subnormal
    0x8000000000000001ull,  // -smallest subnormal
    0x0010000000000000ull,  // smallest normal
    0x7FEFFFFFFFFFFFFFull,  // largest finite
    0xFFEFFFFFFFFFFFFFull,  // -largest finite
    0x7FF0000000000000ull,  // +inf
    0xFFF0000000000000ull,  // -inf
    kQuietNanA,
    kQuietNanB,
    kQuietNanNeg,
    kSignalingNan,
    0x4059000000000000ull,  // 100.0
    0xC059000000000000ull,  // -100.0
    0x3CB0000000000000ull,  // 2^-53, the epsilon-scale value
};
const int kPatternCount = int(sizeof(kPatterns) / sizeof(kPatterns[0]));

// ---------------------------------------------------------------------------
// A local re-derivation of the write-stride vector the Python layer builds,
// so this test can drive real reductions without importing Python's rules:
// 0 on each reduced axis, the row-major stride of the surviving output
// otherwise. ``reduced_first``/``reduced_last`` is an inclusive axis run;
// pass first > last to reduce nothing, or 0..ndim-1 to reduce everything.
// ---------------------------------------------------------------------------
void write_strides(const std::vector<std::int64_t>& shape,
                   std::int64_t reduced_first, std::int64_t reduced_last,
                   std::vector<std::int64_t>& out) {
    const std::int64_t ndim = std::int64_t(shape.size());
    out.assign(size_t(ndim), 0);
    std::int64_t kept = 1;
    for (std::int64_t d = ndim - 1; d >= 0; --d) {
        if (d >= reduced_first && d <= reduced_last) {
            continue;
        }
        out[size_t(d)] = kept;
        kept *= shape[size_t(d)];
    }
}

void row_major(const std::vector<std::int64_t>& shape,
               std::vector<std::int64_t>& out) {
    const std::int64_t ndim = std::int64_t(shape.size());
    out.assign(size_t(ndim), 1);
    for (std::int64_t d = ndim - 2; d >= 0; --d) {
        out[size_t(d)] = out[size_t(d + 1)] * shape[size_t(d + 1)];
    }
}

std::int64_t product(const std::vector<std::int64_t>& dims) {
    std::int64_t total = 1;
    for (std::int64_t dim : dims) {
        total *= dim;
    }
    return total;
}

// ---------------------------------------------------------------------------
// 1. The predicate
// ---------------------------------------------------------------------------

void test_predicate() {
    std::int64_t outer = -1, mid = -1, inner = -1;

    // Rank 0: nothing is reduced, so this is not the block path's business.
    // (The export handles rank 0 before either traversal.)
    check(!tf::reduce_prefers_contiguous_blocks(nullptr, nullptr, nullptr, 0,
                                                &outer, &mid, &inner),
          "rank 0 must not take the block path");

    struct Row {
        std::vector<std::int64_t> shape;
        std::vector<std::int64_t> in_strides;
        std::vector<std::int64_t> out_strides;
        bool expected;
        std::int64_t outer, mid, inner;
        const char* what;
    };
    const std::vector<Row> rows = {
        // full reductions: every write stride 0
        {{6}, {1}, {0}, true, 1, 6, 1, "1-D full"},
        {{2, 3}, {3, 1}, {0, 0}, true, 1, 6, 1, "2-D full"},
        {{2, 3, 4}, {12, 4, 1}, {0, 0, 0}, true, 1, 24, 1, "3-D full"},
        // last axis reduced -> suffix, inner == 1
        {{2, 3}, {3, 1}, {1, 0}, true, 2, 3, 1, "2-D last axis"},
        {{2, 3, 4}, {12, 4, 1}, {3, 1, 0}, true, 6, 4, 1, "3-D last axis"},
        // first axis reduced -> prefix, outer == 1
        {{2, 3}, {3, 1}, {0, 1}, true, 1, 2, 3, "2-D first axis"},
        {{2, 3, 4}, {12, 4, 1}, {0, 4, 1}, true, 1, 2, 12, "3-D first axis"},
        // middle axis reduced -> all three extents > 1
        {{2, 3, 4}, {12, 4, 1}, {4, 0, 1}, true, 2, 3, 4, "3-D middle axis"},
        {{2, 3, 4, 5}, {60, 20, 5, 1}, {20, 0, 5, 1}, true, 2, 3, 20,
         "4-D axis 1"},
        {{2, 3, 4, 5}, {60, 20, 5, 1}, {15, 5, 0, 1}, true, 6, 4, 5,
         "4-D axis 2"},
        // an adjacent run of two reduced axes collapses
        {{2, 3, 4, 5}, {60, 20, 5, 1}, {5, 0, 0, 1}, true, 2, 12, 5,
         "4-D adjacent axes 1-2"},
        {{2, 3, 4}, {12, 4, 1}, {1, 0, 0}, true, 2, 12, 1,
         "3-D trailing run 1-2"},
        {{2, 3, 4}, {12, 4, 1}, {0, 0, 1}, true, 1, 6, 4,
         "3-D leading run 0-1"},
        // unit dimensions
        {{1, 6}, {6, 1}, {0, 1}, true, 1, 1, 6, "leading unit dim reduced"},
        {{6, 1}, {1, 1}, {1, 0}, true, 6, 1, 1, "trailing unit dim reduced"},
        {{2, 1, 4}, {4, 4, 1}, {4, 0, 1}, true, 2, 1, 4, "middle unit dim"},
        // --- rejections ---
        // condition 1: the source is not row-major
        {{2, 3}, {1, 2}, {1, 0}, false, 0, 0, 0, "transposed source"},
        {{2, 3}, {6, 1}, {1, 0}, false, 0, 0, 0, "narrowed leading axis"},
        {{2, 3}, {3, 2}, {1, 0}, false, 0, 0, 0, "non-unit last stride"},
        {{2, 3}, {0, 1}, {1, 0}, false, 0, 0, 0, "broadcast source axis"},
        {{2, 3}, {-3, 1}, {1, 0}, false, 0, 0, 0, "negative source stride"},
        // condition 2: nothing reduced, or a non-adjacent run
        {{2, 3}, {3, 1}, {3, 1}, false, 0, 0, 0, "nothing reduced"},
        {{2, 3, 4}, {12, 4, 1}, {4, 1, 4}, false, 0, 0, 0,
         "non-adjacent run (interrupted)"},
        {{2, 3, 4, 5}, {60, 20, 5, 1}, {0, 5, 0, 1}, false, 0, 0, 0,
         "axes 0 and 2 reduced"},
        // condition 3: the kept axes are not the output's row-major strides
        {{2, 3}, {3, 1}, {2, 0}, false, 0, 0, 0, "wrong kept stride"},
        {{2, 3, 4}, {12, 4, 1}, {4, 0, 2}, false, 0, 0, 0,
         "wrong trailing kept stride"},
        {{2, 3, 4}, {12, 4, 1}, {5, 0, 1}, false, 0, 0, 0,
         "wrong leading kept stride"},
    };

    for (const Row& row : rows) {
        outer = mid = inner = -1;
        const bool got = tf::reduce_prefers_contiguous_blocks(
            row.shape.data(), row.in_strides.data(), row.out_strides.data(),
            std::int64_t(row.shape.size()), &outer, &mid, &inner);
        if (got != row.expected) {
            ++g_failures;
            std::printf("FAIL: predicate %s: expected %d got %d\n", row.what,
                        int(row.expected), int(got));
            continue;
        }
        if (!row.expected) {
            // A rejection must leave the out-parameters untouched.
            check(outer == -1 && mid == -1 && inner == -1,
                  "a rejected predicate must write no extent");
            continue;
        }
        if (outer != row.outer || mid != row.mid || inner != row.inner) {
            ++g_failures;
            std::printf("FAIL: predicate %s extents: expected %lld/%lld/%lld "
                        "got %lld/%lld/%lld\n", row.what,
                        (long long)row.outer, (long long)row.mid,
                        (long long)row.inner, (long long)outer,
                        (long long)mid, (long long)inner);
            continue;
        }
        check(outer * mid * inner == product(row.shape),
              "the factorization must cover every element");
        // Purity and repeatability: the same arguments answer the same way,
        // every time, with no state anywhere.
        for (int repeat = 0; repeat < 3; ++repeat) {
            std::int64_t o2 = -1, m2 = -1, i2 = -1;
            const bool again = tf::reduce_prefers_contiguous_blocks(
                row.shape.data(), row.in_strides.data(),
                row.out_strides.data(), std::int64_t(row.shape.size()), &o2,
                &m2, &i2);
            check(again && o2 == outer && m2 == mid && i2 == inner,
                  "the predicate must be pure and repeatable");
        }
    }
}

// ---------------------------------------------------------------------------
// 2/3/4. The two traversals agree
// ---------------------------------------------------------------------------

// Run one reduction through both traversals and compare. ``seed_value`` is
// written into every destination cell first, so the accumulate-into
// contract is exercised too. Returns the number of positions whose bits
// differ *only* in a NaN payload (both NaN, both quiet, different bits).
int compare_traversals(const std::vector<std::int64_t>& shape,
                       std::int64_t reduced_first, std::int64_t reduced_last,
                       const std::vector<double>& values, double seed_value,
                       const char* what) {
    std::vector<std::int64_t> in_strides;
    std::vector<std::int64_t> out_strides;
    row_major(shape, in_strides);
    write_strides(shape, reduced_first, reduced_last, out_strides);
    const std::int64_t ndim = std::int64_t(shape.size());
    const std::int64_t total = product(shape);

    std::int64_t out_n = 1;
    for (std::int64_t d = 0; d < ndim; ++d) {
        if (d < reduced_first || d > reduced_last) {
            out_n *= shape[size_t(d)];
        }
    }

    std::int64_t outer = 0, mid = 0, inner = 0;
    const bool fast = tf::reduce_prefers_contiguous_blocks(
        shape.data(), in_strides.data(), out_strides.data(), ndim, &outer,
        &mid, &inner);
    if (!fast) {
        ++g_failures;
        std::printf("FAIL: %s should have taken the block path\n", what);
        return 0;
    }

    std::vector<double> generic(size_t(out_n), seed_value);
    std::vector<double> blocks(size_t(out_n), seed_value);
    std::vector<std::int64_t> counter(size_t(ndim), 0);

    tf::sum_generic_strided(values.data(), generic.data(), shape.data(),
                            in_strides.data(), out_strides.data(), 0, ndim,
                            counter.data());
    tf::sum_contiguous_blocks(values.data(), blocks.data(), outer, mid, inner,
                              0);

    int payload_only = 0;
    for (std::int64_t i = 0; i < out_n; ++i) {
        const double a = generic[size_t(i)];
        const double b = blocks[size_t(i)];
        if (bits(a) == bits(b)) {
            continue;
        }
        // The one tolerated difference: two NaNs met in this cell and the
        // paths chose different payloads. Both must still be quiet NaNs.
        if (is_nan(a) && is_nan(b)) {
            check(is_quiet_nan(a), "the generic path produced a signaling NaN");
            check(is_quiet_nan(b), "the block path produced a signaling NaN");
            ++payload_only;
            continue;
        }
        ++g_failures;
        std::printf("FAIL: %s cell %lld: generic %#018llx block %#018llx\n",
                    what, (long long)i, (unsigned long long)bits(a),
                    (unsigned long long)bits(b));
    }
    // NaN *positions* are contractual even when payloads are not.
    for (std::int64_t i = 0; i < out_n; ++i) {
        check(is_nan(generic[size_t(i)]) == is_nan(blocks[size_t(i)]),
              "NaN positions must match on both paths");
    }
    (void)total;
    return payload_only;
}

void test_traversals_agree() {
    // Shapes chosen to hit every extent combination the predicate produces:
    // outer==1, inner==1, both > 1, unit dimensions, prime dimensions, and
    // sizes on both sides of a cache line.
    const std::vector<std::vector<std::int64_t>> shapes = {
        {1}, {2}, {7}, {16}, {17}, {64},
        {1, 1}, {1, 5}, {5, 1}, {2, 3}, {3, 2}, {8, 8}, {7, 11}, {13, 3},
        {1, 2, 3}, {2, 1, 3}, {2, 3, 1}, {3, 4, 5}, {2, 2, 2},
        {2, 3, 2, 3}, {1, 2, 1, 2}, {2, 2, 3, 5},
        {33, 3}, {3, 33}, {5, 13, 2},
    };
    int payload_diffs = 0;
    int comparisons = 0;
    for (const std::vector<std::int64_t>& shape : shapes) {
        const std::int64_t ndim = std::int64_t(shape.size());
        const std::int64_t total = product(shape);
        // Two value fills: a plain ramp (finite, exercises ordinary
        // arithmetic) and the special-pattern cycle.
        std::vector<double> ramp(static_cast<size_t>(total));
        std::vector<double> special(static_cast<size_t>(total));
        for (std::int64_t i = 0; i < total; ++i) {
            ramp[size_t(i)] = double(i % 97) * 0.125 - 3.0;
            special[size_t(i)] =
                from_bits(kPatterns[i % kPatternCount]);
        }
        // Every contiguous reduced run, and both seeds (0.0 for the
        // production case, a non-zero value for the accumulate-into case).
        for (std::int64_t first = 0; first < ndim; ++first) {
            for (std::int64_t last = first; last < ndim; ++last) {
                for (double seed : {0.0, -0.0, 2.5}) {
                    payload_diffs += compare_traversals(
                        shape, first, last, ramp, seed, "ramp");
                    payload_diffs += compare_traversals(
                        shape, first, last, special, seed, "special");
                    comparisons += 2;
                }
            }
        }
    }
    std::printf("  traversal comparisons: %d (NaN-payload-only differences: "
                "%d)\n", comparisons, payload_diffs);
}

// Signed zeros get their own targeted pass: the sign of a zero sum depends
// on the initial accumulator, which is exactly what the block path's local
// accumulator could have changed.
void test_signed_zeros() {
    const double pz = 0.0;
    const double nz = -0.0;
    struct Row { const char* what; std::vector<double> values; };
    const std::vector<Row> rows = {
        {"all +0", {pz, pz, pz, pz, pz, pz}},
        {"all -0", {nz, nz, nz, nz, nz, nz}},
        {"alternating", {pz, nz, pz, nz, pz, nz}},
        {"-0 first", {nz, pz, pz, pz, pz, pz}},
        {"-0 last", {pz, pz, pz, pz, pz, nz}},
        {"-0 and finite", {nz, 1.0, -1.0, nz, 2.0, -2.0}},
        {"cancelling finite", {1.0, -1.0, 2.0, -2.0, 3.0, -3.0}},
    };
    for (const Row& row : rows) {
        for (const std::vector<std::int64_t>& shape :
             std::vector<std::vector<std::int64_t>>{{6}, {2, 3}, {3, 2},
                                                    {1, 6}, {6, 1}}) {
            const std::int64_t ndim = std::int64_t(shape.size());
            for (std::int64_t first = 0; first < ndim; ++first) {
                for (std::int64_t last = first; last < ndim; ++last) {
                    compare_traversals(shape, first, last, row.values, 0.0,
                                       row.what);
                }
            }
        }
    }
    // And the exact value: a run of -0.0 accumulated from a +0.0 seed is
    // +0.0 on both paths, because +0.0 + -0.0 is +0.0.
    std::vector<std::int64_t> shape = {4};
    std::vector<std::int64_t> in_strides, out_strides;
    row_major(shape, in_strides);
    write_strides(shape, 0, 0, out_strides);
    const std::vector<double> minus_zeros = {nz, nz, nz, nz};
    std::vector<std::int64_t> counter(1, 0);
    double generic = 0.0;
    double blocks = 0.0;
    tf::sum_generic_strided(minus_zeros.data(), &generic, shape.data(),
                            in_strides.data(), out_strides.data(), 0, 1,
                            counter.data());
    tf::sum_contiguous_blocks(minus_zeros.data(), &blocks, 1, 4, 1, 0);
    check(bits(generic) == 0x0000000000000000ull,
          "generic: sum of -0.0 from a +0.0 seed must be +0.0");
    check(bits(blocks) == 0x0000000000000000ull,
          "blocks: sum of -0.0 from a +0.0 seed must be +0.0");
    // Seeded with -0.0 instead, both must keep -0.0.
    generic = nz;
    blocks = nz;
    tf::sum_generic_strided(minus_zeros.data(), &generic, shape.data(),
                            in_strides.data(), out_strides.data(), 0, 1,
                            counter.data());
    tf::sum_contiguous_blocks(minus_zeros.data(), &blocks, 1, 4, 1, 0);
    check(bits(generic) == 0x8000000000000000ull,
          "generic: -0.0 seed plus -0.0 values stays -0.0");
    check(bits(blocks) == 0x8000000000000000ull,
          "blocks: -0.0 seed plus -0.0 values stays -0.0");
}

// A NaN manufactured by the arithmetic (+inf + -inf), and signaling NaN
// quieting, on both paths.
void test_manufactured_nan_and_quieting() {
    std::vector<std::int64_t> shape = {2, 3};
    std::vector<std::int64_t> in_strides, out_strides;
    row_major(shape, in_strides);
    const std::vector<double> values = {
        from_bits(0x7FF0000000000000ull), from_bits(0xFFF0000000000000ull),
        1.0, 1.0, from_bits(0x7FF0000000000000ull),
        from_bits(0xFFF0000000000000ull)};
    for (std::int64_t first = 0; first < 2; ++first) {
        for (std::int64_t last = first; last < 2; ++last) {
            write_strides(shape, first, last, out_strides);
            std::int64_t outer = 0, mid = 0, inner = 0;
            check(tf::reduce_prefers_contiguous_blocks(
                      shape.data(), in_strides.data(), out_strides.data(), 2,
                      &outer, &mid, &inner),
                  "the manufactured-NaN case should take the block path");
            std::int64_t out_n = 1;
            for (std::int64_t d = 0; d < 2; ++d) {
                if (d < first || d > last) out_n *= shape[size_t(d)];
            }
            std::vector<double> generic(size_t(out_n), 0.0);
            std::vector<double> blocks(size_t(out_n), 0.0);
            std::vector<std::int64_t> counter(2, 0);
            tf::sum_generic_strided(values.data(), generic.data(),
                                    shape.data(), in_strides.data(),
                                    out_strides.data(), 0, 2, counter.data());
            tf::sum_contiguous_blocks(values.data(), blocks.data(), outer, mid,
                                      inner, 0);
            for (std::int64_t i = 0; i < out_n; ++i) {
                check(is_nan(generic[size_t(i)]) == is_nan(blocks[size_t(i)]),
                      "manufactured NaN must appear in the same positions");
                if (is_nan(generic[size_t(i)])) {
                    check(is_quiet_nan(generic[size_t(i)]),
                          "a manufactured NaN must be quiet (generic)");
                    check(is_quiet_nan(blocks[size_t(i)]),
                          "a manufactured NaN must be quiet (blocks)");
                }
            }
        }
    }
    // Signaling NaN input: both paths must return a *quiet* NaN.
    std::vector<std::int64_t> flat = {3};
    row_major(flat, in_strides);
    write_strides(flat, 0, 0, out_strides);
    const std::vector<double> snan = {from_bits(0x7FF0000000000001ull), 1.0,
                                      2.0};
    std::vector<std::int64_t> counter(1, 0);
    double generic = 0.0;
    double blocks = 0.0;
    tf::sum_generic_strided(snan.data(), &generic, flat.data(),
                            in_strides.data(), out_strides.data(), 0, 1,
                            counter.data());
    tf::sum_contiguous_blocks(snan.data(), &blocks, 1, 3, 1, 0);
    check(is_quiet_nan(generic), "a signaling NaN must be quieted (generic)");
    check(is_quiet_nan(blocks), "a signaling NaN must be quieted (blocks)");
    check(bits(generic) == bits(blocks),
          "one signaling NaN with finite addends must agree bit for bit");
}

// ---------------------------------------------------------------------------
// 5. The exported wrapper
// ---------------------------------------------------------------------------

void test_export() {
    // Rank 0: the single element at ``offset`` is the whole sum, and it is
    // *added* to the destination.
    const double value = 3.5;
    void* src = tf_storage_create(1);
    void* dst = tf_storage_create(1);
    check(src != nullptr && dst != nullptr, "storage allocation failed");
    tf_storage_copy_from(src, &value);
    tf_clear_error();
    tf_core_sum(src, dst, nullptr, nullptr, nullptr, 0, 0);
    check(tf_last_error_code() == TF_OK, "rank-0 sum must not error");
    double got = -1.0;
    tf_storage_copy_to(dst, &got);
    check(bits(got) == bits(value), "rank-0 sum must place the element");
    // Called again, it accumulates rather than overwriting.
    tf_core_sum(src, dst, nullptr, nullptr, nullptr, 0, 0);
    tf_storage_copy_to(dst, &got);
    check(bits(got) == bits(7.0), "rank-0 sum must accumulate");
    tf_storage_destroy(src);
    tf_storage_destroy(dst);

    // A real 2-D reduction through the export, on the block path and on the
    // generic path (a transposed source carrying the same logical values),
    // must produce identical bits.
    const std::int64_t rows = 5, cols = 4;
    std::vector<double> values(static_cast<size_t>(rows * cols));
    for (std::int64_t i = 0; i < rows * cols; ++i) {
        values[size_t(i)] = double(i % 13) * 0.25 - 1.0;
    }
    std::vector<double> transposed(static_cast<size_t>(rows * cols));
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t c = 0; c < cols; ++c) {
            transposed[size_t(c * rows + r)] = values[size_t(r * cols + c)];
        }
    }
    void* contiguous_storage = tf_storage_create(rows * cols);
    void* transposed_storage = tf_storage_create(rows * cols);
    tf_storage_copy_from(contiguous_storage, values.data());
    tf_storage_copy_from(transposed_storage, transposed.data());

    const std::vector<std::int64_t> shape = {rows, cols};
    const std::vector<std::int64_t> contiguous_strides = {cols, 1};
    // The same logical (rows, cols) view over column-major storage.
    const std::vector<std::int64_t> view_strides = {1, rows};
    for (std::int64_t axis = 0; axis < 2; ++axis) {
        std::vector<std::int64_t> out_strides;
        write_strides(shape, axis, axis, out_strides);
        const std::int64_t out_n = axis == 0 ? cols : rows;
        void* out_a = tf_storage_create(out_n);
        void* out_b = tf_storage_create(out_n);
        tf_clear_error();
        tf_core_sum(contiguous_storage, out_a, shape.data(),
                    contiguous_strides.data(), out_strides.data(), 0, 2);
        check(tf_last_error_code() == TF_OK, "block-path sum must not error");
        tf_core_sum(transposed_storage, out_b, shape.data(),
                    view_strides.data(), out_strides.data(), 0, 2);
        check(tf_last_error_code() == TF_OK, "generic-path sum must not error");
        std::vector<double> got_a(static_cast<size_t>(out_n));
        std::vector<double> got_b(static_cast<size_t>(out_n));
        tf_storage_copy_to(out_a, got_a.data());
        tf_storage_copy_to(out_b, got_b.data());
        for (std::int64_t i = 0; i < out_n; ++i) {
            check(bits(got_a[size_t(i)]) == bits(got_b[size_t(i)]),
                  "the export's two paths must agree bit for bit");
        }
        tf_storage_destroy(out_a);
        tf_storage_destroy(out_b);
    }
    // A nonzero offset on the block path: reduce the last four rows.
    {
        const std::vector<std::int64_t> sub_shape = {4, cols};
        std::vector<std::int64_t> out_strides;
        write_strides(sub_shape, 0, 0, out_strides);
        void* out = tf_storage_create(cols);
        tf_clear_error();
        tf_core_sum(contiguous_storage, out, sub_shape.data(),
                    contiguous_strides.data(), out_strides.data(), cols, 2);
        check(tf_last_error_code() == TF_OK, "offset sum must not error");
        std::vector<double> got_out(static_cast<size_t>(cols));
        tf_storage_copy_to(out, got_out.data());
        for (std::int64_t c = 0; c < cols; ++c) {
            double expected = 0.0;
            for (std::int64_t r = 1; r < rows; ++r) {
                expected += values[size_t(r * cols + c)];
            }
            check(bits(got_out[size_t(c)]) == bits(expected),
                  "an offset block-path sum must skip the leading rows");
        }
        tf_storage_destroy(out);
    }
    tf_storage_destroy(contiguous_storage);
    tf_storage_destroy(transposed_storage);
}

}  // namespace

int main() {
    std::printf("sum reduction / H6 traversal dispatch tests\n");
    test_predicate();
    test_traversals_agree();
    test_signed_zeros();
    test_manufactured_nan_and_quieting();
    test_export();
    if (g_failures == 0) {
        std::printf("all sum-reduction checks passed\n");
        return 0;
    }
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
}
