// Dependency-free C++ test for the native `index_select` (Phase K,
// milestone K4). No GoogleTest / Catch2 — a plain executable that prints
// failures and returns a nonzero exit code if any check fails, so CTest
// reports pass/fail.
//
// This binary compiles cpp/src/indexing.cpp (plus storage.cpp and error.cpp)
// directly, so it reaches BOTH layers of the milestone:
//
//   1. **The internal traversal** — ``tf::index_select_contiguous`` at both
//      instantiations, which the shared library does not export and which is
//      reachable only by compiling the source in. Driven on plain host
//      arrays at ``double`` and at ``float`` **separately**: outer > 1,
//      inner > 1, both at once, a single index, several indices, duplicates,
//      reversed order, and a guard cell proving it never writes past
//      ``outer * index_count * inner``.
//
//   2. **The exported guarded ABI** — ``tf_core_index_select``'s complete
//      validation matrix, driven through real ``tf::Storage`` handles and the
//      thread-local error slot, with the destination proved **byte-for-byte
//      unchanged after every rejection**.
//
// Two claims here need *driving* rather than asserting:
//
//   * **The mixed-role success.** A floating source, an ``int64`` index, and
//     a floating destination of the source's dtype must be accepted
//     together. A ``require_index`` on the destination, or a
//     ``require_matching_dtype`` across the source/index pair, would reject
//     every valid call (design §22.9, §22.10).
//   * **The complete scan precedes every write.** A bad index that follows
//     several valid ones must reject with the destination byte-for-byte
//     unchanged — which an incremental check-as-you-copy implementation
//     would fail while still "rejecting" (design §14.4).
//
// **Floating values are compared as raw bit patterns**, never with a
// tolerance and never with ``==``: this operation copies object
// representations, so signed zeros, infinities, subnormals, and distinct NaN
// payloads must all arrive identical, and ``==`` cannot see any of that
// (design §29.6). Integer comparisons are exact ``==`` on ``std::int64_t``.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "tf_indexing_internal.h"
#include "tf_internal.h"

// -- the exports this test drives -------------------------------------------
TF_EXPORT void* tf_storage_create_typed(std::int64_t size,
                                        std::int32_t dtype_code);
TF_EXPORT void tf_storage_destroy(void* handle);
TF_EXPORT void tf_storage_copy_from(void* handle, const void* src);
TF_EXPORT void tf_storage_copy_to(const void* handle, void* dst);
TF_EXPORT void tf_core_index_select(
    const void* src_handle, std::int64_t src_offset,
    const void* idx_handle, std::int64_t idx_offset,
    void* dst_handle,
    std::int64_t outer, std::int64_t axis_length,
    std::int64_t index_count, std::int64_t inner);
TF_EXPORT int tf_last_error_code();
TF_EXPORT const char* tf_last_error_message();
TF_EXPORT void tf_clear_error();

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

void check_named(bool condition, const char* what, const char* who) {
    if (!condition) {
        std::printf("FAIL: %s (%s)\n", what, who);
        ++g_failures;
    }
}

// Raw byte view of a storage's buffer. The test needs to inspect bytes,
// which the production typed accessors deliberately cannot do generically;
// this exists only to prove what the bytes are.
std::vector<unsigned char> snapshot(const void* handle) {
    const tf::Storage* storage = tf::as_storage(handle);
    const std::size_t bytes = static_cast<std::size_t>(storage->size)
        * tf::dtype_item_size(storage->dtype);
    const unsigned char* data = static_cast<const unsigned char*>(storage->data);
    return std::vector<unsigned char>(data, data + bytes);
}

bool unchanged(const void* handle, const std::vector<unsigned char>& before) {
    return snapshot(handle) == before;
}

double f64_from_bits(std::uint64_t pattern) {
    double out;
    std::memcpy(&out, &pattern, sizeof(out));
    return out;
}

float f32_from_bits(std::uint32_t pattern) {
    float out;
    std::memcpy(&out, &pattern, sizeof(out));
    return out;
}

// The comparison this file uses for every floating value: the object
// representation, byte for byte. ``==`` would call two NaNs unequal, would
// call +0.0 and -0.0 equal, and would see no difference between two NaN
// payloads — so it can prove none of what this operation promises.
template <class T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// The exceptional values, per width, built from bit patterns so a NaN's
// payload and signalling bit are chosen deliberately rather than inherited.
template <class T>
struct Special;

template <>
struct Special<double> {
    static double nan_quiet() { return f64_from_bits(0x7FF8000000000001ULL); }
    static double nan_other() { return f64_from_bits(0xFFF800000BADF00DULL); }
    static double nan_signalling() { return f64_from_bits(0x7FF0000000000007ULL); }
    static double subnormal() { return f64_from_bits(0x0000000000000003ULL); }
    static const char* name() { return "float64"; }
    static std::int32_t code() { return TF_DTYPE_FLOAT64; }
};

template <>
struct Special<float> {
    static float nan_quiet() { return f32_from_bits(0x7FC00001U); }
    static float nan_other() { return f32_from_bits(0xFFC0BEEFU); }
    static float nan_signalling() { return f32_from_bits(0x7F800007U); }
    static float subnormal() { return f32_from_bits(0x00000003U); }
    static const char* name() { return "float32"; }
    static std::int32_t code() { return TF_DTYPE_FLOAT32; }
};

// ===========================================================================
// 1. The internal traversal
// ===========================================================================

// A rank-1 selection: outer = 1, inner = 1. The destination's j-th element
// is the source's ``indices[j]``-th, exactly.
template <class T>
void test_the_flat_selection_preserves_order_and_duplicates() {
    const char* who = Special<T>::name();
    const std::vector<T> source = {T(10), T(11), T(12), T(13)};

    // Order preserved, duplicates preserved, nothing sorted or deduplicated.
    const std::vector<std::int64_t> indices = {2, 0, 2, 1, 3, 3};
    std::vector<T> out(indices.size(), T(-1));
    tf::index_select_contiguous<T>(
        source.data(), indices.data(), out.data(),
        1, 4, static_cast<std::int64_t>(indices.size()), 1);
    const std::vector<T> expected = {T(12), T(10), T(12), T(11), T(13), T(13)};
    for (std::size_t j = 0; j < expected.size(); ++j) {
        check_named(same_bits(out[j], expected[j]),
                    "a flat selection produced the wrong value", who);
    }

    // A single index.
    std::vector<T> one(1, T(-1));
    const std::int64_t single[1] = {3};
    tf::index_select_contiguous<T>(source.data(), single, one.data(),
                                   1, 4, 1, 1);
    check_named(same_bits(one[0], T(13)), "a single-index selection is wrong",
                who);

    // Reversed order.
    const std::int64_t reversed[4] = {3, 2, 1, 0};
    std::vector<T> back(4, T(-1));
    tf::index_select_contiguous<T>(source.data(), reversed, back.data(),
                                   1, 4, 4, 1);
    check_named(same_bits(back[0], T(13)) && same_bits(back[1], T(12)) &&
                same_bits(back[2], T(11)) && same_bits(back[3], T(10)),
                "a reversed selection is wrong", who);
}

// The full decomposition: outer > 1 and inner > 1 together, so the plane
// arithmetic at both ends is exercised rather than degenerating.
template <class T>
void test_the_axis_decomposition() {
    const char* who = Special<T>::name();
    // Shape (2, 3, 2) selecting along axis 1: outer = 2, axis_length = 3,
    // inner = 2. Element (o, k, i) lives at o*6 + k*2 + i.
    std::vector<T> source(12);
    for (std::size_t n = 0; n < source.size(); ++n) {
        source[n] = static_cast<T>(n);
    }
    const std::int64_t indices[3] = {2, 0, 2};
    std::vector<T> out(2 * 3 * 2, T(-1));
    tf::index_select_contiguous<T>(source.data(), indices, out.data(),
                                   2, 3, 3, 2);
    // o = 0: k = 2 -> (4, 5); k = 0 -> (0, 1); k = 2 -> (4, 5).
    // o = 1: k = 2 -> (10, 11); k = 0 -> (6, 7); k = 2 -> (10, 11).
    const T expected[12] = {T(4), T(5), T(0), T(1), T(4), T(5),
                            T(10), T(11), T(6), T(7), T(10), T(11)};
    for (std::size_t n = 0; n < 12; ++n) {
        check_named(same_bits(out[n], expected[n]),
                    "an (outer > 1, inner > 1) selection is wrong", who);
    }

    // The same block selecting along axis 0: outer = 1, axis_length = 2,
    // inner = 6 — whole planes, in the caller's order.
    const std::int64_t planes[3] = {1, 1, 0};
    std::vector<T> first(3 * 6, T(-1));
    tf::index_select_contiguous<T>(source.data(), planes, first.data(),
                                   1, 2, 3, 6);
    for (std::size_t i = 0; i < 6; ++i) {
        check_named(same_bits(first[i], source[6 + i]) &&
                    same_bits(first[6 + i], source[6 + i]) &&
                    same_bits(first[12 + i], source[i]),
                    "a first-axis selection is wrong", who);
    }

    // ...and along the last axis: outer = 6, axis_length = 2, inner = 1.
    const std::int64_t last[2] = {1, 1};
    std::vector<T> tail(6 * 2, T(-1));
    tf::index_select_contiguous<T>(source.data(), last, tail.data(),
                                   6, 2, 2, 1);
    for (std::size_t o = 0; o < 6; ++o) {
        check_named(same_bits(tail[o * 2], source[o * 2 + 1]) &&
                    same_bits(tail[o * 2 + 1], source[o * 2 + 1]),
                    "a last-axis selection is wrong", who);
    }
}

// Every exceptional representation crosses unchanged, bit for bit. This is
// the claim ``memcpy`` exists for: a floating assignment would be permitted
// to canonicalize a signalling NaN, and ``==`` could not see it if it did.
template <class T>
void test_exceptional_values_cross_by_object_representation() {
    const char* who = Special<T>::name();
    const T inf = std::numeric_limits<T>::infinity();
    const std::vector<T> source = {
        T(0), -T(0), inf, -inf,
        Special<T>::subnormal(), -Special<T>::subnormal(),
        Special<T>::nan_quiet(), Special<T>::nan_other(),
        Special<T>::nan_signalling(), T(1.5),
    };
    // Every position, twice, so a duplicate is proved to copy independently
    // rather than to share.
    std::vector<std::int64_t> indices;
    for (std::int64_t k = 0; k < static_cast<std::int64_t>(source.size()); ++k) {
        indices.push_back(k);
        indices.push_back(k);
    }
    std::vector<T> out(indices.size(), T(-1));
    tf::index_select_contiguous<T>(
        source.data(), indices.data(), out.data(),
        1, static_cast<std::int64_t>(source.size()),
        static_cast<std::int64_t>(indices.size()), 1);
    for (std::size_t j = 0; j < indices.size(); ++j) {
        const std::size_t named = static_cast<std::size_t>(indices[j]);
        check_named(same_bits(out[j], source[named]),
                    "an exceptional value did not cross bit-for-bit", who);
    }
    // The negative control for ``same_bits`` itself: it must be able to tell
    // the two signed zeros and two NaN payloads apart, which ``==`` cannot.
    check_named(!same_bits(T(0), -T(0)),
                "same_bits cannot distinguish the signed zeros", who);
    check_named(!same_bits(Special<T>::nan_quiet(), Special<T>::nan_other()),
                "same_bits cannot distinguish two NaN payloads", who);
}

// The kernel writes exactly outer * index_count * inner elements, never one
// more, and never reads outside the source block it was given.
template <class T>
void test_the_kernel_writes_only_inside_its_destination() {
    const char* who = Special<T>::name();
    const std::vector<T> source = {T(1), T(2), T(3), T(4), T(5), T(6)};
    const std::int64_t indices[2] = {1, 0};
    std::vector<T> out(6, T(-777));   // two guard cells past the end
    tf::index_select_contiguous<T>(source.data(), indices, out.data(),
                                   2, 3, 2, 1);
    // o = 0: (2, 1). o = 1: (5, 4).
    check_named(same_bits(out[0], T(2)) && same_bits(out[1], T(1)) &&
                same_bits(out[2], T(5)) && same_bits(out[3], T(4)),
                "wrong values", who);
    check_named(same_bits(out[4], T(-777)) && same_bits(out[5], T(-777)),
                "the kernel wrote past outer * index_count * inner", who);
}

// ===========================================================================
// 2. The exported guarded ABI
// ===========================================================================

void* make_typed(std::int64_t size, std::int32_t code) {
    void* handle = tf_storage_create_typed(size, code);
    check(handle != nullptr, "a typed storage allocation failed");
    return handle;
}

// The mixed-role success, at each value width: floating source, int64 index,
// floating destination of the source's dtype.
template <class T>
void test_the_mixed_role_call_succeeds() {
    const char* who = Special<T>::name();
    const std::int32_t code = Special<T>::code();
    void* src = make_typed(6, code);
    void* idx = make_typed(4, TF_DTYPE_INT64);
    void* dst = make_typed(4, code);

    const std::vector<T> values = {T(10), T(11), T(12), T(13), T(14), T(15)};
    tf_storage_copy_from(src, values.data());
    const std::int64_t indices[4] = {2, 0, 2, 5};
    tf_storage_copy_from(idx, indices);

    tf_clear_error();
    tf_core_index_select(src, 0, idx, 0, dst, 1, 6, 4, 1);
    check_named(tf_last_error_code() == TF_OK,
                "a valid floating/int64/floating index_select was rejected",
                who);
    std::vector<T> out(4, T(-1));
    tf_storage_copy_to(dst, out.data());
    check_named(same_bits(out[0], T(12)) && same_bits(out[1], T(10)) &&
                same_bits(out[2], T(12)) && same_bits(out[3], T(15)),
                "a valid index_select produced wrong values", who);

    // A nonzero SOURCE offset: the same storage read one element in, so
    // index 0 now names the second value.
    void* shifted = make_typed(2, code);
    const std::int64_t two[2] = {0, 1};
    void* two_idx = make_typed(2, TF_DTYPE_INT64);
    tf_storage_copy_from(two_idx, two);
    tf_clear_error();
    tf_core_index_select(src, 1, two_idx, 0, shifted, 1, 5, 2, 1);
    check_named(tf_last_error_code() == TF_OK,
                "an offset source index_select was rejected", who);
    std::vector<T> shifted_out(2, T(-1));
    tf_storage_copy_to(shifted, shifted_out.data());
    check_named(same_bits(shifted_out[0], T(11)) &&
                same_bits(shifted_out[1], T(12)),
                "an offset source index_select ignored its offset", who);

    // A nonzero INDEX offset: the last two of the four indices above.
    void* from_index_offset = make_typed(2, code);
    tf_clear_error();
    tf_core_index_select(src, 0, idx, 2, from_index_offset, 1, 6, 2, 1);
    check_named(tf_last_error_code() == TF_OK,
                "an offset index index_select was rejected", who);
    std::vector<T> offset_out(2, T(-1));
    tf_storage_copy_to(from_index_offset, offset_out.data());
    check_named(same_bits(offset_out[0], T(12)) &&
                same_bits(offset_out[1], T(15)),
                "an offset index index_select ignored its offset", who);

    // outer > 1 and inner > 1 through the ABI, so the decomposition the
    // export takes is proved rather than only the internal traversal's.
    void* block = make_typed(12, code);
    std::vector<T> block_values(12);
    for (std::size_t n = 0; n < block_values.size(); ++n) {
        block_values[n] = static_cast<T>(n);
    }
    tf_storage_copy_from(block, block_values.data());
    void* block_idx = make_typed(3, TF_DTYPE_INT64);
    const std::int64_t block_indices[3] = {2, 0, 2};
    tf_storage_copy_from(block_idx, block_indices);
    void* block_dst = make_typed(12, code);
    tf_clear_error();
    tf_core_index_select(block, 0, block_idx, 0, block_dst, 2, 3, 3, 2);
    check_named(tf_last_error_code() == TF_OK,
                "a decomposed index_select was rejected", who);
    std::vector<T> block_out(12, T(-1));
    tf_storage_copy_to(block_dst, block_out.data());
    const T expected[12] = {T(4), T(5), T(0), T(1), T(4), T(5),
                            T(10), T(11), T(6), T(7), T(10), T(11)};
    for (std::size_t n = 0; n < 12; ++n) {
        check_named(same_bits(block_out[n], expected[n]),
                    "a decomposed index_select produced wrong values", who);
    }

    tf_storage_destroy(block_dst);
    tf_storage_destroy(block_idx);
    tf_storage_destroy(block);
    tf_storage_destroy(from_index_offset);
    tf_storage_destroy(two_idx);
    tf_storage_destroy(shifted);
    tf_storage_destroy(dst);
    tf_storage_destroy(idx);
    tf_storage_destroy(src);
    tf_clear_error();
}

// Exceptional bit patterns through the real export, not only the internal
// traversal — so the whole path from a storage handle to a storage handle is
// proved to change no bit.
void test_exceptional_values_cross_the_abi_unchanged() {
    void* src = make_typed(6, TF_DTYPE_FLOAT64);
    void* idx = make_typed(6, TF_DTYPE_INT64);
    void* dst = make_typed(6, TF_DTYPE_FLOAT64);
    const double values[6] = {
        0.0, -0.0,
        std::numeric_limits<double>::infinity(),
        Special<double>::subnormal(),
        Special<double>::nan_quiet(),
        Special<double>::nan_signalling(),
    };
    tf_storage_copy_from(src, values);
    const std::int64_t indices[6] = {5, 4, 3, 2, 1, 0};
    tf_storage_copy_from(idx, indices);

    tf_clear_error();
    tf_core_index_select(src, 0, idx, 0, dst, 1, 6, 6, 1);
    check(tf_last_error_code() == TF_OK, "an exceptional-value call failed");
    double out[6];
    tf_storage_copy_to(dst, out);
    for (int j = 0; j < 6; ++j) {
        check(same_bits(out[j], values[5 - j]),
              "an exceptional value changed crossing the ABI");
    }

    tf_storage_destroy(dst);
    tf_storage_destroy(idx);
    tf_storage_destroy(src);
    tf_clear_error();
}

// Every rejection: the call returns, records TF_ERROR_INVALID, and leaves the
// destination byte-for-byte unchanged.
void test_the_validation_matrix() {
    void* f64 = make_typed(6, TF_DTYPE_FLOAT64);
    void* f32 = make_typed(6, TF_DTYPE_FLOAT32);
    void* i64_src = make_typed(6, TF_DTYPE_INT64);
    void* idx = make_typed(3, TF_DTYPE_INT64);
    void* dst = make_typed(3, TF_DTYPE_FLOAT64);
    void* f32_dst = make_typed(3, TF_DTYPE_FLOAT32);
    void* i64_dst = make_typed(3, TF_DTYPE_INT64);
    void* float_idx = make_typed(3, TF_DTYPE_FLOAT64);

    const double values[6] = {10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    tf_storage_copy_from(f64, values);
    const std::int64_t indices[3] = {2, 0, 2};
    tf_storage_copy_from(idx, indices);
    // A recognizable destination pattern, so "unchanged" is a real check
    // rather than "still zero".
    const double poison[3] = {-424242.5, 987654321.25, -7.5};
    tf_storage_copy_from(dst, poison);
    const std::vector<unsigned char> before = snapshot(dst);

    struct Case {
        const char* name;
        const void* src;
        std::int64_t src_offset;
        const void* idx;
        std::int64_t idx_offset;
        void* destination;
        std::int64_t outer;
        std::int64_t axis_length;
        std::int64_t index_count;
        std::int64_t inner;
    };
    const std::int64_t huge = INT64_MAX / 2;
    const Case cases[] = {
        {"null source",           nullptr, 0, idx,       0, dst,     1, 6, 3, 1},
        {"null index",            f64,     0, nullptr,   0, dst,     1, 6, 3, 1},
        {"null destination",      f64,     0, idx,       0, nullptr, 1, 6, 3, 1},
        {"int64 source",          i64_src, 0, idx,       0, dst,     1, 6, 3, 1},
        {"int64 destination",     f64,     0, idx,       0, i64_dst, 1, 6, 3, 1},
        {"dtype mismatch",        f64,     0, idx,       0, f32_dst, 1, 6, 3, 1},
        {"floating index",        f64,     0, float_idx, 0, dst,     1, 6, 3, 1},
        {"zero outer",            f64,     0, idx,       0, dst,     0, 6, 3, 1},
        {"zero axis_length",      f64,     0, idx,       0, dst,     1, 0, 3, 1},
        {"zero index_count",      f64,     0, idx,       0, dst,     1, 6, 0, 1},
        {"zero inner",            f64,     0, idx,       0, dst,     1, 6, 3, 0},
        {"negative outer",        f64,     0, idx,       0, dst,    -1, 6, 3, 1},
        {"negative axis_length",  f64,     0, idx,       0, dst,     1,-6, 3, 1},
        {"negative index_count",  f64,     0, idx,       0, dst,     1, 6,-3, 1},
        {"negative inner",        f64,     0, idx,       0, dst,     1, 6, 3,-1},
        {"source product overflow",
                                  f64,     0, idx,       0, dst,  huge, 4, 3, 1},
        {"destination product overflow",
                                  f64,     0, idx,       0, dst,     1, 6, huge, 4},
        {"negative source offset",
                                  f64,    -1, idx,       0, dst,     1, 6, 3, 1},
        {"negative index offset", f64,     0, idx,      -1, dst,     1, 6, 3, 1},
        {"source span too long",  f64,     0, idx,       0, dst,     1, 7, 3, 1},
        {"source offset pushes the span out",
                                  f64,     1, idx,       0, dst,     1, 6, 3, 1},
        {"index span too long",   f64,     0, idx,       2, dst,     1, 6, 3, 1},
        {"destination too small", f64,     0, idx,       0, dst,     2, 3, 2, 1},
        {"destination too large", f64,     0, idx,       0, dst,     1, 6, 2, 1},
        {"destination aliases the source",
                                  dst,     0, idx,       0, dst,     1, 3, 3, 1},
        // The index handle passed as the destination too. It is refused as a
        // **non-floating destination**, one step before the aliasing check
        // could see it, and that is not a gap: one storage carries one dtype,
        // so a handle that is a valid index operand can never be a valid
        // destination. The aliasing pair below it is defense in depth the
        // role checks already make unreachable -- retained because the C ABI
        // validates independently of what another check happens to imply.
        {"index handle passed as the destination",
                                  f64,     0, i64_dst,   0, i64_dst, 1, 6, 3, 1},
    };
    for (const Case& one : cases) {
        tf_clear_error();
        void* destination = one.destination;
        const std::vector<unsigned char> guard =
            destination == nullptr ? std::vector<unsigned char>()
            : snapshot(destination);
        tf_core_index_select(one.src, one.src_offset, one.idx, one.idx_offset,
                             destination, one.outer, one.axis_length,
                             one.index_count, one.inner);
        check_named(tf_last_error_code() == TF_ERROR_INVALID,
                    "a malformed index_select was not rejected", one.name);
        const char* message = tf_last_error_message();
        check_named(message != nullptr && message[0] != '\0',
                    "a rejection recorded no message", one.name);
        if (destination != nullptr) {
            check_named(unchanged(destination, guard),
                        "a rejected index_select wrote to its destination",
                        one.name);
        }
    }
    check(unchanged(dst, before),
          "the destination moved across the rejection matrix");

    // The non-vacuity control: the same destination and the same arguments
    // with sound operands succeed and write.
    tf_clear_error();
    tf_core_index_select(f64, 0, idx, 0, dst, 1, 6, 3, 1);
    check(tf_last_error_code() == TF_OK, "the control call was rejected too");
    check(!unchanged(dst, before), "the control call did not write");
    double out[3] = {-1.0, -1.0, -1.0};
    tf_storage_copy_to(dst, out);
    check(same_bits(out[0], 12.0) && same_bits(out[1], 10.0) &&
          same_bits(out[2], 12.0), "the control call wrote wrong values");

    // A float32 source into the float32 destination, so the export's one
    // dispatch is proved to reach both arms.
    const float narrow[6] = {2.0f, 2.5f, 1.0f, 0.5f, 4.0f, 4.5f};
    tf_storage_copy_from(f32, narrow);
    tf_clear_error();
    tf_core_index_select(f32, 0, idx, 0, f32_dst, 1, 6, 3, 1);
    check(tf_last_error_code() == TF_OK, "a float32 index_select was rejected");
    float narrow_out[3] = {-1.0f, -1.0f, -1.0f};
    tf_storage_copy_to(f32_dst, narrow_out);
    check(same_bits(narrow_out[0], 1.0f) && same_bits(narrow_out[1], 2.0f) &&
          same_bits(narrow_out[2], 1.0f),
          "a float32 index_select produced wrong values");

    tf_storage_destroy(float_idx);
    tf_storage_destroy(i64_dst);
    tf_storage_destroy(f32_dst);
    tf_storage_destroy(dst);
    tf_storage_destroy(idx);
    tf_storage_destroy(i64_src);
    tf_storage_destroy(f32);
    tf_storage_destroy(f64);
    tf_clear_error();
}

// The scan is COMPLETE and precedes every write: a bad index anywhere in the
// span rejects, and the destination is byte-for-byte unchanged even when
// several valid indices precede the bad one.
void test_the_index_scan_is_complete_and_precedes_every_write() {
    void* src = make_typed(4, TF_DTYPE_FLOAT64);
    void* dst = make_typed(4, TF_DTYPE_FLOAT64);
    void* idx = make_typed(4, TF_DTYPE_INT64);
    const double values[4] = {10.0, 11.0, 12.0, 13.0};
    tf_storage_copy_from(src, values);
    const double poison[4] = {-1.5, -2.5, -3.5, -4.5};
    tf_storage_copy_from(dst, poison);
    const std::vector<unsigned char> before = snapshot(dst);

    struct Case {
        const char* name;
        std::int64_t indices[4];
    };
    const Case cases[] = {
        {"negative at position 0",       {-1, 0, 1, 2}},
        {"negative at the end",          {0, 1, 2, -1}},
        {"negative one after valid",     {0, 1, -5, 2}},
        {"equal to axis_length",         {0, 1, 4, 2}},
        {"beyond axis_length",           {0, 1, 2, 99}},
        {"INT64_MIN",                    {0, INT64_MIN, 2, 3}},
        {"INT64_MAX",                    {0, 1, 2, INT64_MAX}},
    };
    for (const Case& one : cases) {
        tf_storage_copy_from(idx, one.indices);
        tf_clear_error();
        tf_core_index_select(src, 0, idx, 0, dst, 1, 4, 4, 1);
        check_named(tf_last_error_code() == TF_ERROR_INVALID,
                    "an out-of-range index was not rejected", one.name);
        const char* message = tf_last_error_message();
        check_named(message != nullptr && std::strstr(message, "index_select")
                        != nullptr,
                    "the rejection does not name the operation", one.name);
        check_named(message != nullptr && std::strstr(message, "position")
                        != nullptr,
                    "the rejection does not name the offending position",
                    one.name);
        check_named(unchanged(dst, before),
                    "a rejected index_select wrote to its destination",
                    one.name);
    }

    // The boundary that must be ACCEPTED, beside the ones rejected: the last
    // valid position, axis_length - 1.
    const std::int64_t edge[4] = {3, 3, 0, 3};
    tf_storage_copy_from(idx, edge);
    tf_clear_error();
    tf_core_index_select(src, 0, idx, 0, dst, 1, 4, 4, 1);
    check(tf_last_error_code() == TF_OK,
          "the last valid index position was rejected");
    double out[4];
    tf_storage_copy_to(dst, out);
    check(same_bits(out[0], 13.0) && same_bits(out[1], 13.0) &&
          same_bits(out[2], 10.0) && same_bits(out[3], 13.0),
          "the boundary control wrote wrong values");

    tf_storage_destroy(idx);
    tf_storage_destroy(dst);
    tf_storage_destroy(src);
    tf_clear_error();
}

// A guarded export clears the thread-local slot on entry, so a rejection
// recorded by an earlier call is never misread as this one's failure.
void test_the_guard_clears_the_slot_on_entry() {
    void* src = make_typed(4, TF_DTYPE_FLOAT64);
    void* idx = make_typed(2, TF_DTYPE_INT64);
    void* dst = make_typed(2, TF_DTYPE_FLOAT64);
    const double values[4] = {1.0, 4.0, 2.0, 3.0};
    tf_storage_copy_from(src, values);
    const std::int64_t indices[2] = {1, 3};
    tf_storage_copy_from(idx, indices);

    tf_core_index_select(nullptr, 0, idx, 0, dst, 1, 4, 2, 1);
    check(tf_last_error_code() == TF_ERROR_INVALID,
          "a null-handle index_select did not record a rejection");
    tf_core_index_select(src, 0, idx, 0, dst, 1, 4, 2, 1);
    check(tf_last_error_code() == TF_OK,
          "a valid index_select did not clear the error slot on entry");
    double out[2];
    tf_storage_copy_to(dst, out);
    check(same_bits(out[0], 4.0) && same_bits(out[1], 3.0),
          "the recovering call produced wrong values");

    tf_storage_destroy(dst);
    tf_storage_destroy(idx);
    tf_storage_destroy(src);
    tf_clear_error();
}

}  // namespace

int main() {
    test_the_flat_selection_preserves_order_and_duplicates<double>();
    test_the_flat_selection_preserves_order_and_duplicates<float>();
    test_the_axis_decomposition<double>();
    test_the_axis_decomposition<float>();
    test_exceptional_values_cross_by_object_representation<double>();
    test_exceptional_values_cross_by_object_representation<float>();
    test_the_kernel_writes_only_inside_its_destination<double>();
    test_the_kernel_writes_only_inside_its_destination<float>();

    test_the_mixed_role_call_succeeds<double>();
    test_the_mixed_role_call_succeeds<float>();
    test_exceptional_values_cross_the_abi_unchanged();
    test_the_validation_matrix();
    test_the_index_scan_is_complete_and_precedes_every_write();
    test_the_guard_clears_the_slot_on_entry();

    if (g_failures == 0) {
        std::printf("test_index_select: all checks passed\n");
        return 0;
    }
    std::printf("test_index_select: %d check(s) failed\n", g_failures);
    return 1;
}
