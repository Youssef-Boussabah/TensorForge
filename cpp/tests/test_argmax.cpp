// Dependency-free C++ test for the native `argmax` (Phase K, milestone K3).
// No GoogleTest / Catch2 — a plain executable that prints failures and
// returns a nonzero exit code if any check fails, so CTest reports
// pass/fail.
//
// This binary compiles cpp/src/indexing.cpp (plus storage.cpp and error.cpp)
// directly, so it reaches BOTH layers of the milestone:
//
//   1. **The internal traversal** — ``tf::argmax_contiguous`` at both
//      instantiations, which the shared library does not export and which is
//      reachable only by compiling the source in. Every row of the normative
//      value table (design §17.5) is driven here on plain host arrays, at
//      ``double`` and at ``float`` **separately**: a unique maximum, equal
//      maxima, both signed zeros, an all-``-inf`` run, repeated ``+inf``, one
//      NaN, several NaNs, NaN against either infinity, NaN at index 0, and a
//      length-1 run — full and per-axis decompositions, several independent
//      runs, and runs that differ in whether they contain a NaN at all.
//
//   2. **The exported guarded ABI** — ``tf_core_argmax``'s complete
//      validation matrix, driven through real ``tf::Storage`` handles and the
//      thread-local error slot, with the destination proved **byte-for-byte
//      unchanged after every rejection**.
//
// The claim that matters most here, and the one a structural check could not
// make, is the **mixed-role success**: a floating source and an ``int64``
// destination must be *accepted*. A ``tf::require_floating`` on the
// destination or a ``tf::require_matching_dtype`` across the pair would
// reject every valid call, so the valid call is driven at both source widths
// rather than merely asserted to be permitted (design §22.8, exit gate 3a).
//
// Integer comparison only — exact ``==`` on ``std::int64_t``. No tolerance is
// used anywhere in this file, because every value it checks is an index.

#include <cmath>
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
TF_EXPORT std::int64_t tf_storage_size(const void* handle);
TF_EXPORT void tf_storage_copy_from(void* handle, const void* src);
TF_EXPORT void tf_storage_copy_to(const void* handle, void* dst);
TF_EXPORT void tf_core_argmax(
    const void* src_handle, std::int64_t src_offset,
    void* dst_handle,
    std::int64_t outer, std::int64_t axis_length, std::int64_t inner);
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

// ===========================================================================
// 1. The internal traversal — the normative value rule, row by row
// ===========================================================================
//
// One templated driver, instantiated at both widths, so a rule that held at
// one width and not the other cannot pass. Every expectation below is
// written from the contract's algorithm rather than delegated to any
// library's convention.

template <class T>
std::int64_t argmax_of(const std::vector<T>& run) {
    std::vector<std::int64_t> out(1, -1);
    tf::argmax_contiguous<T>(run.data(), out.data(), 1,
                             static_cast<std::int64_t>(run.size()), 1);
    return out[0];
}

template <class T>
void expect_run(const std::vector<T>& run, std::int64_t expected,
                const char* who) {
    check_named(argmax_of<T>(run) == expected, "wrong index for a run", who);
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
    static double negative_zero() { return -0.0; }
    static const char* name() { return "float64"; }
};

template <>
struct Special<float> {
    static float nan_quiet() { return f32_from_bits(0x7FC00001U); }
    static float nan_other() { return f32_from_bits(0xFFC0BEEFU); }
    static float nan_signalling() { return f32_from_bits(0x7F800007U); }
    static float negative_zero() { return -0.0f; }
    static const char* name() { return "float32"; }
};

template <class T>
void test_every_value_rule_row() {
    const char* who = Special<T>::name();
    const T inf = std::numeric_limits<T>::infinity();
    const T quiet = Special<T>::nan_quiet();
    const T other = Special<T>::nan_other();
    const T signalling = Special<T>::nan_signalling();

    // A unique maximum -> its own index.
    expect_run<T>({T(1), T(5), T(3), T(2)}, 1, who);
    expect_run<T>({T(-9), T(-4), T(-100)}, 1, who);

    // Equal maxima -> the LOWEST index, because `>` is strict.
    expect_run<T>({T(7), T(7), T(2)}, 0, who);
    expect_run<T>({T(1), T(7), T(7), T(7)}, 1, who);
    expect_run<T>({T(4), T(4), T(4), T(4)}, 0, who);

    // Both signed zeros among the maxima -> the lowest index of either,
    // because IEEE comparison does not order +0.0 and -0.0.
    expect_run<T>({Special<T>::negative_zero(), T(0), T(-1)}, 0, who);
    expect_run<T>({T(-1), T(0), Special<T>::negative_zero()}, 1, who);
    expect_run<T>({T(-1), Special<T>::negative_zero(), T(0)}, 1, who);

    // Every element -inf -> 0; they all tie.
    expect_run<T>({-inf, -inf, -inf}, 0, who);
    // +inf present -> the lowest index holding it.
    expect_run<T>({T(3), inf, T(9), inf}, 1, who);
    expect_run<T>({-inf, T(0), inf}, 2, who);

    // Exactly one NaN -> its index, whatever else the run holds. A NaN beats
    // every finite value and either infinity.
    expect_run<T>({T(1), quiet, T(3)}, 1, who);
    expect_run<T>({T(1), T(2), quiet}, 2, who);
    expect_run<T>({inf, quiet}, 1, who);
    expect_run<T>({-inf, quiet}, 1, who);
    expect_run<T>({quiet, inf, T(1000)}, 0, who);

    // NaN at index 0 -> 0, and nothing thereafter displaces it.
    expect_run<T>({quiet, T(1), T(2), T(3)}, 0, who);
    expect_run<T>({quiet, inf}, 0, who);

    // Several NaNs -> the FIRST. A later NaN cannot displace an incumbent
    // one, and the payload/sign/signalling bit is never inspected: the
    // three NaNs below differ in all three respects.
    expect_run<T>({T(1), quiet, other, T(2)}, 1, who);
    expect_run<T>({other, quiet, signalling}, 0, who);
    expect_run<T>({T(0), signalling, quiet, other}, 1, who);
    expect_run<T>({quiet, quiet, quiet}, 0, who);

    // A run of length 1 -> 0, NaN or not.
    expect_run<T>({T(42)}, 0, who);
    expect_run<T>({quiet}, 0, who);
    expect_run<T>({-inf}, 0, who);
}

// The per-axis decomposition: one run per (outer, inner) pair, each answered
// independently. A NaN in one run must not reach another.
template <class T>
void test_the_axis_decomposition_is_run_local() {
    const char* who = Special<T>::name();
    const T quiet = Special<T>::nan_quiet();
    const T inf = std::numeric_limits<T>::infinity();

    // Shape (2, 3, 2) searched along axis 1: outer = 2, axis_length = 3,
    // inner = 2. Element (o, k, i) lives at o*6 + k*2 + i.
    const std::vector<T> source = {
        // o = 0
        T(1), T(9),      // k = 0
        T(5), T(2),      // k = 1
        T(3), T(4),      // k = 2
        // o = 1
        quiet, T(0),     // k = 0
        T(8), T(7),      // k = 1
        T(8), quiet,     // k = 2
    };
    std::vector<std::int64_t> out(4, -1);
    tf::argmax_contiguous<T>(source.data(), out.data(), 2, 3, 2);
    // (0, i=0): 1, 5, 3   -> 1.   (0, i=1): 9, 2, 4 -> 0.
    // (1, i=0): NaN, 8, 8 -> 0.   (1, i=1): 0, 7, NaN -> 2.
    check_named(out[0] == 1 && out[1] == 0 && out[2] == 0 && out[3] == 2,
                "a per-axis decomposition answered a run wrongly", who);

    // The same block searched along the LAST axis: outer = 6, axis_length =
    // 2, inner = 1 — six independent runs of two, read straight off the
    // flat order above: (1,9) (5,2) (3,4) (NaN,0) (8,7) (8,NaN).
    std::vector<std::int64_t> last(6, -1);
    tf::argmax_contiguous<T>(source.data(), last.data(), 6, 2, 1);
    check_named(last[0] == 1 && last[1] == 0 && last[2] == 1 &&
                last[3] == 0 && last[4] == 0 && last[5] == 1,
                "a last-axis decomposition answered a run wrongly", who);

    // ...and along the FIRST axis: outer = 1, axis_length = 2, inner = 6.
    std::vector<std::int64_t> first(6, -1);
    tf::argmax_contiguous<T>(source.data(), first.data(), 1, 2, 6);
    // Column-wise over the two planes: 1 vs NaN -> 1; 9 vs 0 -> 0;
    // 5 vs 8 -> 1; 2 vs 7 -> 1; 3 vs 8 -> 1; 4 vs NaN -> 1.
    check_named(first[0] == 1 && first[1] == 0 && first[2] == 1 &&
                first[3] == 1 && first[4] == 1 && first[5] == 1,
                "a first-axis decomposition answered a run wrongly", who);

    // A full reduction is the same kernel with (1, numel, 1), and the answer
    // is the flat row-major index.
    std::vector<std::int64_t> whole(1, -1);
    const std::vector<T> finite = {T(1), T(9), T(5), T(2), T(3), inf};
    tf::argmax_contiguous<T>(finite.data(), whole.data(), 1, 6, 1);
    check_named(whole[0] == 5, "a full reduction answered wrongly", who);
}

// The kernel writes exactly outer * inner indices and never one more.
template <class T>
void test_the_kernel_writes_only_inside_its_destination() {
    const char* who = Special<T>::name();
    const std::vector<T> source = {T(1), T(2), T(3), T(4), T(5), T(6)};
    std::vector<std::int64_t> out(5, -777);   // one guard cell past the end
    tf::argmax_contiguous<T>(source.data(), out.data(), 2, 3, 1);
    check_named(out[0] == 2 && out[1] == 2, "wrong indices", who);
    check_named(out[2] == -777 && out[3] == -777 && out[4] == -777,
                "the kernel wrote past outer * inner", who);
}

// ===========================================================================
// 2. The exported guarded ABI
// ===========================================================================

void* make_typed(std::int64_t size, std::int32_t code) {
    void* handle = tf_storage_create_typed(size, code);
    check(handle != nullptr, "a typed storage allocation failed");
    return handle;
}

// The one claim a structural check cannot make: a **floating source and an
// int64 destination succeed**. A require_floating(destination) or a
// require_matching_dtype(source, destination) would reject this call, so
// driving it is what proves neither is applied.
template <class T>
void test_the_mixed_role_call_succeeds(std::int32_t code, const char* who) {
    void* src = make_typed(6, code);
    void* dst = make_typed(2, TF_DTYPE_INT64);
    const std::vector<T> values = {T(1), T(7), T(3), T(9), T(9), T(2)};
    tf_storage_copy_from(src, values.data());

    tf_core_argmax(src, 0, dst, 2, 3, 1);
    check_named(tf_last_error_code() == TF_OK,
                "a valid floating -> int64 argmax was rejected", who);
    std::vector<std::int64_t> out(2, -1);
    tf_storage_copy_to(dst, out.data());
    check_named(out[0] == 1 && out[1] == 0,
                "a valid argmax produced wrong indices", who);

    // A full reduction through the same symbol: (1, numel, 1).
    void* whole = make_typed(1, TF_DTYPE_INT64);
    tf_core_argmax(src, 0, whole, 1, 6, 1);
    check_named(tf_last_error_code() == TF_OK,
                "a valid full argmax was rejected", who);
    std::int64_t flat = -1;
    tf_storage_copy_to(whole, &flat);
    check_named(flat == 3, "a full argmax produced the wrong flat index", who);

    // A non-zero source offset addresses the same block one element in.
    void* shifted = make_typed(1, TF_DTYPE_INT64);
    tf_core_argmax(src, 1, shifted, 1, 5, 1);
    check_named(tf_last_error_code() == TF_OK,
                "an offset argmax was rejected", who);
    std::int64_t shifted_index = -1;
    tf_storage_copy_to(shifted, &shifted_index);
    // Values from index 1: 7, 3, 9, 9, 2 -> the first 9, at 2.
    check_named(shifted_index == 2,
                "an offset argmax ignored its offset", who);

    tf_storage_destroy(shifted);
    tf_storage_destroy(whole);
    tf_storage_destroy(dst);
    tf_storage_destroy(src);
    tf_clear_error();
}

// Every rejection: the call returns, records TF_ERROR_INVALID, and leaves the
// destination byte-for-byte unchanged.
void test_the_validation_matrix() {
    void* f64 = make_typed(6, TF_DTYPE_FLOAT64);
    void* f32 = make_typed(6, TF_DTYPE_FLOAT32);
    void* i64_src = make_typed(6, TF_DTYPE_INT64);
    void* dst = make_typed(2, TF_DTYPE_INT64);
    void* float_dst = make_typed(2, TF_DTYPE_FLOAT64);

    const double values[6] = {1.0, 7.0, 3.0, 9.0, 9.0, 2.0};
    tf_storage_copy_from(f64, values);
    // A recognizable destination pattern, so "unchanged" is a real check
    // rather than "still zero".
    const std::int64_t poison[2] = {-424242, 987654321};
    tf_storage_copy_from(dst, poison);
    const std::vector<unsigned char> before = snapshot(dst);
    const std::vector<unsigned char> float_before = snapshot(float_dst);

    struct Case {
        const char* name;
        const void* src;
        std::int64_t offset;
        void* destination;
        std::int64_t outer;
        std::int64_t axis_length;
        std::int64_t inner;
    };
    const std::int64_t huge = INT64_MAX / 2;
    const Case cases[] = {
        {"null source",            nullptr, 0, dst,       2, 3, 1},
        {"null destination",       f64,     0, nullptr,   2, 3, 1},
        {"int64 source",           i64_src, 0, dst,       2, 3, 1},
        {"floating destination",   f64,     0, float_dst, 2, 3, 1},
        {"zero outer",             f64,     0, dst,       0, 3, 1},
        {"zero axis_length",       f64,     0, dst,       2, 0, 1},
        {"zero inner",             f64,     0, dst,       2, 3, 0},
        {"negative outer",         f64,     0, dst,      -2, 3, 1},
        {"negative axis_length",   f64,     0, dst,       2, -3, 1},
        {"negative inner",         f64,     0, dst,       2, 3, -1},
        {"product overflow",       f64,     0, dst,    huge, 4, 1},
        {"negative offset",        f64,    -1, dst,       2, 3, 1},
        {"source span too long",   f64,     0, dst,       2, 4, 1},
        {"offset pushes span out", f64,     2, dst,       2, 3, 1},
        {"destination too small",  f64,     0, dst,       3, 2, 1},
        {"destination too large",  f64,     0, dst,       1, 6, 1},
        {"self-aliasing handle",   dst,     0, dst,       2, 1, 1},
    };
    for (const Case& one : cases) {
        tf_clear_error();
        void* destination = one.destination;
        const std::vector<unsigned char> guard =
            destination == nullptr ? std::vector<unsigned char>()
            : snapshot(destination);
        tf_core_argmax(one.src, one.offset, destination,
                       one.outer, one.axis_length, one.inner);
        check_named(tf_last_error_code() == TF_ERROR_INVALID,
                    "a malformed argmax was not rejected", one.name);
        const char* message = tf_last_error_message();
        check_named(message != nullptr && message[0] != '\0',
                    "a rejection recorded no message", one.name);
        if (destination != nullptr) {
            check_named(unchanged(destination, guard),
                        "a rejected argmax wrote to its destination",
                        one.name);
        }
    }
    // ...and both destinations are still exactly what they were before the
    // whole matrix ran.
    check(unchanged(dst, before),
          "the int64 destination moved across the rejection matrix");
    check(unchanged(float_dst, float_before),
          "the float destination moved across the rejection matrix");

    // The non-vacuity control: the same destination and the same arguments
    // with a valid source succeed and write. Without this, "every row
    // rejected" would also be satisfied by a matrix that failed for some
    // unrelated reason.
    tf_clear_error();
    tf_core_argmax(f64, 0, dst, 2, 3, 1);
    check(tf_last_error_code() == TF_OK,
          "the control call was rejected too");
    check(!unchanged(dst, before),
          "the control call did not write its destination");
    std::int64_t out[2] = {-1, -1};
    tf_storage_copy_to(dst, out);
    check(out[0] == 1 && out[1] == 0, "the control call wrote wrong indices");

    // A float32 source through the identical destination, so the export's
    // one dispatch is proved to reach both arms.
    const float narrow[6] = {2.0f, 2.0f, 1.0f, 0.5f, 4.0f, 4.0f};
    tf_storage_copy_from(f32, narrow);
    tf_clear_error();
    tf_core_argmax(f32, 0, dst, 2, 3, 1);
    check(tf_last_error_code() == TF_OK, "a float32 argmax was rejected");
    tf_storage_copy_to(dst, out);
    check(out[0] == 0 && out[1] == 1,
          "a float32 argmax produced wrong indices");

    tf_storage_destroy(float_dst);
    tf_storage_destroy(dst);
    tf_storage_destroy(i64_src);
    tf_storage_destroy(f32);
    tf_storage_destroy(f64);
    tf_clear_error();
}

// A guarded export clears the thread-local slot on entry, so a rejection
// recorded by an earlier call is never misread as this one's failure.
void test_the_guard_clears_the_slot_on_entry() {
    void* f64 = make_typed(4, TF_DTYPE_FLOAT64);
    void* dst = make_typed(1, TF_DTYPE_INT64);
    const double values[4] = {1.0, 4.0, 2.0, 3.0};
    tf_storage_copy_from(f64, values);

    tf_core_argmax(nullptr, 0, dst, 1, 4, 1);
    check(tf_last_error_code() == TF_ERROR_INVALID,
          "a null-handle argmax did not record a rejection");
    tf_core_argmax(f64, 0, dst, 1, 4, 1);
    check(tf_last_error_code() == TF_OK,
          "a valid argmax did not clear the error slot on entry");
    std::int64_t index = -1;
    tf_storage_copy_to(dst, &index);
    check(index == 1, "the recovering call produced the wrong index");

    tf_storage_destroy(dst);
    tf_storage_destroy(f64);
    tf_clear_error();
}

// The index-role guard itself, driven directly: it is a *different* question
// from require_floating, and the two must not be interchangeable.
void test_the_index_role_guard() {
    void* i64 = make_typed(2, TF_DTYPE_INT64);
    void* f64 = make_typed(2, TF_DTYPE_FLOAT64);
    void* f32 = make_typed(2, TF_DTYPE_FLOAT32);

    tf_clear_error();
    check(tf::require_index("probe", "destination", i64),
          "require_index rejected int64 storage");
    check(tf_last_error_code() == TF_OK,
          "require_index recorded an error for a valid handle");
    // A null handle passes, so each export keeps its own null validation.
    check(tf::require_index("probe", "destination", nullptr),
          "require_index rejected a null handle");

    for (void* handle : {f64, f32}) {
        tf_clear_error();
        check(!tf::require_index("probe", "destination", handle),
              "require_index accepted floating storage");
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "require_index recorded no rejection");
        const char* message = tf_last_error_message();
        check(message != nullptr && std::strstr(message, "destination") != nullptr,
              "the rejection message does not name the operand's role");
    }
    // ...and the two guards really do answer opposite questions on the same
    // handles, which is why applying either in the other's place would
    // reject every valid argmax call.
    tf_clear_error();
    check(tf::require_floating("probe", {f64}) &&
          !tf::require_index("probe", "destination", f64),
          "the floating and index guards agree about a float64 handle");
    tf_clear_error();
    check(!tf::require_floating("probe", {i64}) &&
          tf::require_index("probe", "destination", i64),
          "the floating and index guards agree about an int64 handle");

    tf_storage_destroy(f32);
    tf_storage_destroy(f64);
    tf_storage_destroy(i64);
    tf_clear_error();
}

}  // namespace

int main() {
    test_every_value_rule_row<double>();
    test_every_value_rule_row<float>();
    test_the_axis_decomposition_is_run_local<double>();
    test_the_axis_decomposition_is_run_local<float>();
    test_the_kernel_writes_only_inside_its_destination<double>();
    test_the_kernel_writes_only_inside_its_destination<float>();

    test_the_mixed_role_call_succeeds<double>(TF_DTYPE_FLOAT64, "float64");
    test_the_mixed_role_call_succeeds<float>(TF_DTYPE_FLOAT32, "float32");
    test_the_validation_matrix();
    test_the_guard_clears_the_slot_on_entry();
    test_the_index_role_guard();

    if (g_failures == 0) {
        std::printf("test_argmax: all checks passed\n");
        return 0;
    }
    std::printf("test_argmax: %d check(s) failed\n", g_failures);
    return 1;
}
