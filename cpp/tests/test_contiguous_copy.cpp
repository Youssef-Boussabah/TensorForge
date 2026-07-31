// Dependency-free C++ test for the strided-to-contiguous gather and its
// H5 traversal dispatch (Phase H, milestone H5). No GoogleTest / Catch2 —
// a plain executable that prints failures and returns a nonzero exit code
// if any check fails, so CTest reports pass/fail.
//
// This binary compiles cpp/src/elementwise.cpp (plus storage.cpp and
// error.cpp) directly, so it can reach BOTH the hidden
// ``tf::copy_prefers_contiguous`` predicate and the exported
// ``tf_core_contiguous_copy`` wrapper the predicate lives inside.
//
// What it proves, at the layer where the property is actually decided —
// i.e. without the Python wrapper, the ctypes boundary, or NumPy
// anywhere in the picture:
//
//   1. The predicate is exactly "row-major contiguous", agrees with the
//      definition backends/cpp.py uses, and is total: it answers for
//      rank-0, unit dimensions, transposes, narrows, negative strides,
//      and broadcast (stride-0) layouts alike.
//   2. The two traversals it picks between write **identical bits** for
//      every representable double — the whole point of H5's claim that a
//      value copy preserves values. The sweep deliberately includes the
//      three patterns an arithmetic copy would have changed (-0.0, and
//      both signs of signaling NaN) plus the NaN payloads an arithmetic
//      copy happens to preserve, so a regression toward arithmetic would
//      be caught here rather than inferred.
//   3. The gather still validates its arguments and writes nothing when
//      it rejects — unchanged from E3.1 — on both dispatch paths.
//
// Bit comparison, never a tolerance: the entire question is whether bits
// survive, and ``==`` on doubles cannot see -0.0 versus +0.0 and calls
// every NaN unequal to itself.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "tf_copy_internal.h"
#include "tf_internal.h"

TF_EXPORT void* tf_storage_create(std::int64_t size);
TF_EXPORT void tf_storage_destroy(void* handle);
TF_EXPORT void tf_storage_copy_from(void* handle, const double* src);
TF_EXPORT void tf_storage_copy_to(const void* handle, double* dst);
TF_EXPORT void tf_core_contiguous_copy(
    const void* src, void* dst,
    const std::int64_t* shape, const std::int64_t* strides,
    std::int64_t offset, std::int64_t ndim);
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

// The 17-pattern IEEE-754 sweep the Python suite uses, in the same order.
// Three of these are the ones an arithmetic copy would have changed.
const std::uint64_t kPatterns[] = {
    0x0000000000000000ull,  // +0.0
    0x8000000000000000ull,  // -0.0                 <- addition normalizes
    0x3FF0000000000000ull,  // 1.0
    0xBFF0000000000000ull,  // -1.0
    0x7FF0000000000000ull,  // +inf
    0xFFF0000000000000ull,  // -inf
    0x7FF8000000000001ull,  // quiet NaN, payload 1
    0x7FF800000000000Aull,  // quiet NaN, payload A
    0xFFF8000000000001ull,  // negative quiet NaN
    0x7FF0000000000001ull,  // signaling NaN        <- addition quiets
    0xFFF0000000000001ull,  // negative signaling NaN <- addition quiets
    0x0000000000000001ull,  // smallest subnormal
    0x8000000000000001ull,  // -smallest subnormal
    0x000FFFFFFFFFFFFFull,  // largest subnormal
    0x0010000000000000ull,  // smallest normal
    0x7FEFFFFFFFFFFFFFull,  // largest finite
    0xFFEFFFFFFFFFFFFFull,  // -largest finite
};
const std::int64_t kPatternCount =
    static_cast<std::int64_t>(sizeof(kPatterns) / sizeof(kPatterns[0]));

// ---------------------------------------------------------------------------
// 1. The dispatch predicate
// ---------------------------------------------------------------------------

void test_predicate_matches_row_major_contiguity() {
    // Rank 0: a scalar view is contiguous (one element at the offset).
    check(tf::copy_prefers_contiguous(nullptr, nullptr, 0),
          "rank-0 view is contiguous");

    {   // 1-D unit stride, and anything else.
        const std::int64_t shape[] = {5};
        const std::int64_t good[] = {1};
        const std::int64_t bad[] = {2};
        const std::int64_t rev[] = {-1};
        check(tf::copy_prefers_contiguous(shape, good, 1), "1-D unit stride");
        check(!tf::copy_prefers_contiguous(shape, bad, 1), "1-D stride 2");
        check(!tf::copy_prefers_contiguous(shape, rev, 1), "1-D reversed");
    }
    {   // 2-D row-major versus its transpose.
        const std::int64_t shape[] = {3, 4};
        const std::int64_t row_major[] = {4, 1};
        const std::int64_t transposed_shape[] = {4, 3};
        const std::int64_t transposed[] = {1, 4};
        check(tf::copy_prefers_contiguous(shape, row_major, 2),
              "2-D row-major");
        check(!tf::copy_prefers_contiguous(transposed_shape, transposed, 2),
              "2-D transposed");
    }
    {   // A narrow along axis 0 keeps row-major strides (only the offset
        // moves), which is exactly why the fast path is reachable for it.
        const std::int64_t shape[] = {2, 4};
        const std::int64_t strides[] = {4, 1};
        check(tf::copy_prefers_contiguous(shape, strides, 2),
              "axis-0 narrow stays contiguous");
    }
    {   // A narrow along the last axis does not.
        const std::int64_t shape[] = {3, 2};
        const std::int64_t strides[] = {4, 1};
        check(!tf::copy_prefers_contiguous(shape, strides, 2),
              "last-axis narrow is not contiguous");
    }
    {   // Broadcast layouts carry a zero stride and must never be swept
        // flat: the flat loop would read numel consecutive elements
        // instead of re-reading one.
        const std::int64_t shape[] = {4, 3};
        const std::int64_t strides[] = {0, 1};
        check(!tf::copy_prefers_contiguous(shape, strides, 2),
              "stride-0 broadcast is not contiguous");
    }
    {   // 4-D NCHW, the layout the CNN stack actually copies.
        const std::int64_t shape[] = {2, 3, 4, 5};
        const std::int64_t strides[] = {60, 20, 5, 1};
        check(tf::copy_prefers_contiguous(shape, strides, 4), "4-D NCHW");
    }
    {   // A trailing unit dimension still has to carry stride 1 in the
        // row-major convention backends/cpp.py computes, so the exact
        // comparison is what keeps the two layers in agreement.
        const std::int64_t shape[] = {3, 1};
        const std::int64_t strides[] = {1, 1};
        check(tf::copy_prefers_contiguous(shape, strides, 2),
              "trailing unit dimension");
    }
    // Purity: the same metadata answers the same way every time.
    const std::int64_t shape[] = {3, 4};
    const std::int64_t strides[] = {4, 1};
    for (int i = 0; i < 100; ++i) {
        check(tf::copy_prefers_contiguous(shape, strides, 2),
              "predicate is deterministic");
    }
}

// ---------------------------------------------------------------------------
// 2. The two traversals agree bit for bit
// ---------------------------------------------------------------------------

// Gather ``src`` through the given layout and return the destination.
std::vector<double> gather(const std::vector<double>& source,
                           const std::vector<std::int64_t>& shape,
                           const std::vector<std::int64_t>& strides,
                           std::int64_t offset, std::int64_t numel) {
    void* src = tf_storage_create(static_cast<std::int64_t>(source.size()));
    void* dst = tf_storage_create(numel);
    tf_storage_copy_from(src, source.data());
    tf_clear_error();
    tf_core_contiguous_copy(
        src, dst, shape.empty() ? nullptr : shape.data(),
        strides.empty() ? nullptr : strides.data(), offset,
        static_cast<std::int64_t>(shape.size()));
    std::vector<double> out(static_cast<std::size_t>(numel));
    tf_storage_copy_to(dst, out.data());
    tf_storage_destroy(src);
    tf_storage_destroy(dst);
    return out;
}

void test_every_pattern_survives_the_contiguous_path() {
    std::vector<double> source;
    for (std::int64_t i = 0; i < kPatternCount; ++i) {
        source.push_back(from_bits(kPatterns[i]));
    }
    const std::vector<double> out =
        gather(source, {kPatternCount}, {1}, 0, kPatternCount);
    check(tf_last_error_code() == 0, "contiguous gather reported no error");
    for (std::int64_t i = 0; i < kPatternCount; ++i) {
        check(bits(out[static_cast<std::size_t>(i)]) == kPatterns[i],
              "contiguous path preserved the exact bit pattern");
    }
}

void test_the_two_traversals_write_identical_bits() {
    // One (17, 2) buffer whose column 0 holds the sweep. Reading it as
    // (17, 1) with row-major strides {2, 1} is NOT contiguous, so that
    // layout takes the odometer; reading the same values as a (17,)
    // vector out of a packed buffer takes the flat loop. Both must
    // produce exactly the same 17 destination doubles.
    std::vector<double> padded;
    std::vector<double> packed;
    for (std::int64_t i = 0; i < kPatternCount; ++i) {
        padded.push_back(from_bits(kPatterns[i]));
        padded.push_back(0.0);
        packed.push_back(from_bits(kPatterns[i]));
    }
    const std::vector<double> strided =
        gather(padded, {kPatternCount}, {2}, 0, kPatternCount);
    const std::vector<double> flat =
        gather(packed, {kPatternCount}, {1}, 0, kPatternCount);
    for (std::int64_t i = 0; i < kPatternCount; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        check(bits(strided[k]) == kPatterns[i],
              "odometer path preserved the exact bit pattern");
        check(bits(strided[k]) == bits(flat[k]),
              "odometer and flat traversals agree bit for bit");
    }
}

void test_a_nonzero_offset_and_rank_zero_agree() {
    std::vector<double> source;
    source.push_back(from_bits(0x7FF0000000000001ull));  // signaling NaN
    for (std::int64_t i = 0; i < kPatternCount; ++i) {
        source.push_back(from_bits(kPatterns[i]));
    }
    // Contiguous but offset by one: the flat loop must start at ``offset``.
    const std::vector<double> out =
        gather(source, {kPatternCount}, {1}, 1, kPatternCount);
    for (std::int64_t i = 0; i < kPatternCount; ++i) {
        check(bits(out[static_cast<std::size_t>(i)]) == kPatterns[i],
              "offset contiguous gather read from the right place");
    }
    // Rank 0 at every offset, including the signaling NaN at index 0.
    for (std::int64_t i = 0; i <= kPatternCount; ++i) {
        const std::vector<double> one = gather(source, {}, {}, i, 1);
        check(bits(one[0]) == bits(source[static_cast<std::size_t>(i)]),
              "rank-0 gather preserved the exact bit pattern");
    }
}

void test_a_reversed_view_is_gathered_in_logical_order() {
    // Negative strides can only be walked by the odometer; the predicate
    // must send them there, and the result must be the reversed sequence.
    std::vector<double> source;
    for (std::int64_t i = 0; i < 8; ++i) {
        source.push_back(static_cast<double>(i));
    }
    const std::vector<double> out = gather(source, {8}, {-1}, 7, 8);
    check(tf_last_error_code() == 0, "reversed gather reported no error");
    for (std::int64_t i = 0; i < 8; ++i) {
        check(out[static_cast<std::size_t>(i)] == static_cast<double>(7 - i),
              "reversed view gathered in logical order");
    }
}

// ---------------------------------------------------------------------------
// 3. Validation is unchanged on both paths
// ---------------------------------------------------------------------------

void test_rejections_write_nothing_on_either_path() {
    std::vector<double> source(16, 1.0);
    void* src = tf_storage_create(16);
    tf_storage_copy_from(src, source.data());

    struct Case {
        const char* name;
        std::int64_t shape[2];
        std::int64_t strides[2];
        std::int64_t offset;
        std::int64_t ndim;
    };
    // The first is a would-be *contiguous* layout that overruns the
    // source, the second a strided one: both must be rejected before any
    // traversal is chosen, so neither dispatch path can write.
    const Case cases[] = {
        {"contiguous overrun", {8, 4}, {4, 1}, 0, 2},
        {"strided overrun", {4, 4}, {8, 1}, 0, 2},
        {"negative offset", {2, 2}, {2, 1}, -1, 2},
    };
    for (const Case& c : cases) {
        void* dst = tf_storage_create(32);
        std::vector<double> before(32);
        tf_storage_copy_to(dst, before.data());
        tf_clear_error();
        tf_core_contiguous_copy(src, dst, c.shape, c.strides, c.offset,
                                c.ndim);
        check(tf_last_error_code() != 0, c.name);
        std::vector<double> after(32);
        tf_storage_copy_to(dst, after.data());
        check(std::memcmp(before.data(), after.data(),
                          before.size() * sizeof(double)) == 0,
              "a rejected gather wrote nothing");
        tf_storage_destroy(dst);
    }
    tf_clear_error();
    tf_storage_destroy(src);
}

void test_the_source_is_never_mutated() {
    std::vector<double> source;
    for (std::int64_t i = 0; i < kPatternCount; ++i) {
        source.push_back(from_bits(kPatterns[i]));
    }
    void* src = tf_storage_create(kPatternCount);
    void* dst = tf_storage_create(kPatternCount);
    tf_storage_copy_from(src, source.data());
    const std::int64_t shape[] = {kPatternCount};
    const std::int64_t strides[] = {1};
    tf_clear_error();
    tf_core_contiguous_copy(src, dst, shape, strides, 0, 1);
    std::vector<double> after(static_cast<std::size_t>(kPatternCount));
    tf_storage_copy_to(src, after.data());
    for (std::int64_t i = 0; i < kPatternCount; ++i) {
        check(bits(after[static_cast<std::size_t>(i)]) == kPatterns[i],
              "the gather left its source bit-identical");
    }
    tf_storage_destroy(src);
    tf_storage_destroy(dst);
}

}  // namespace

int main() {
    test_predicate_matches_row_major_contiguity();
    test_every_pattern_survives_the_contiguous_path();
    test_the_two_traversals_write_identical_bits();
    test_a_nonzero_offset_and_rank_zero_agree();
    test_a_reversed_view_is_gathered_in_logical_order();
    test_rejections_write_nothing_on_either_path();
    test_the_source_is_never_mutated();

    if (g_failures != 0) {
        std::printf("%d contiguous-copy check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all contiguous-copy checks passed\n");
    return 0;
}
