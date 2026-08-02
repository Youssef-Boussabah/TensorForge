// Dependency-free C++ test for dtype-general reductions, matrix
// multiplication, and the narrow-backward scatter (Phase I, milestone I4).
// No GoogleTest / Catch2 — a plain executable that prints failures and
// returns a nonzero exit code if any check fails, so CTest reports
// pass/fail.
//
// This binary compiles cpp/src/reduction.cpp, cpp/src/matmul.cpp,
// cpp/src/storage.cpp, and cpp/src/error.cpp directly, so it reaches the
// hidden ``tf::reduce_prefers_contiguous_blocks`` and
// ``tf::matmul_prefers_row_sweep`` predicates and **both** traversals of
// each family alongside the exported wrappers they live inside — at the
// layer where the properties are actually decided, with no Python wrapper,
// no ctypes boundary, and no NumPy anywhere.
//
// It is deliberately complementary to the neighbouring targets rather than a
// superset of any of them:
//
//   * ``test_sum_reduction`` proves H6's block traversal agrees with the
//     retained odometer at **float64**, including its signed-zero and NaN
//     sweeps. That stays exactly as it is and is not restated here.
//   * ``test_matmul`` proves H2's four-part contract at **float64**,
//     including the predicate boundaries and the uninitialized-destination
//     argument. Also untouched.
//   * ``test_dtype_storage`` proves float32 is allocatable; ``test_typed_-
//     transfer`` proves the boundaries I2 opened; ``test_dtype_elementwise``
//     proves the family I3 opened.
//
// This one proves the set that opened at I4.
//
// What it proves:
//
//   1. **Both traversals agree, per dtype.** For every reduction layout the
//      predicate accepts, the optimized block walk and the retained
//      odometer produce **bit-identical** destinations — at float32 as well
//      as float64 — and the odometer alone answers for the layouts the
//      predicate declines (transposed, narrowed, negatively strided,
//      broadcast). The same, per path, for matmul.
//
//   2. **float32 really accumulates in float32.** This is the claim that
//      *could not* be tested at I3 and can be here. Every I3 operation
//      produced each destination element with a single correctly-rounded
//      IEEE operation, where computing in binary64 and rounding once is
//      provably indistinguishable from computing in binary32; a *sum* of
//      three or more values is not. So each family gets a deterministic
//      **witness vector** on which:
//
//          sequential binary32 accumulation
//              != binary64 accumulation narrowed once
//
//      and TensorForge is asserted, by raw bit pattern, to equal the first
//      and to differ from the second. A hidden widening accumulator would
//      fail that assertion at a named index rather than passing quietly.
//      The witnesses are chosen so the gap is a *large* relative one, not a
//      last-bit coincidence, and they are exercised on **both** traversals
//      of each family, because a widening accumulator introduced on only
//      the optimized path is exactly the plausible mistake.
//
//   3. **The mean scalar is narrowed once, before the loop.** ``1/count``
//      crosses the unchanged ``double`` ABI parameter of
//      ``tf_storage_scale``; the result is asserted equal to
//      ``value * T(1.0/count)`` by bit pattern, and — on a witness where
//      the two differ — *unequal* to ``T(value * (1.0/count))`` computed in
//      binary64. That distinguishes "narrow the scalar once, then multiply
//      in T" from "multiply in double, then narrow", which is the whole of
//      design §7.4 at this level.
//
//   4. **Output dtype follows the operands**, the storage tag is immutable
//      across a call, and the logical size still counts elements.
//
//   5. **Mixed dtype is rejected before anything is written**, in every
//      participating handle position independently, with TF_ERROR_INVALID
//      and a byte-for-byte unchanged destination.
//
//   6. **The scatter is still a scatter.** ``narrow_backward`` writes only
//      the narrowed region at both dtypes and leaves every other cell
//      holding the zero the allocation gave it, and it reproduces its
//      source's bits exactly (it assigns; it does not compute).
//
//   7. **float64 did not move.** Every check runs at float64 too, against
//      the same independently written references, so the generalization is
//      proved not to have disturbed the width Phase H measured.
//
// Comparison is by raw bit pattern throughout: ``==`` on floating values
// cannot see -0.0 versus +0.0 and calls every NaN unequal to itself.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

#include "tf_internal.h"
#include "tf_matmul_internal.h"
#include "tf_reduction_internal.h"

TF_EXPORT void* tf_storage_create_typed(std::int64_t size,
                                        std::int32_t dtype_code);
TF_EXPORT void tf_storage_destroy(void* handle);
TF_EXPORT std::int64_t tf_storage_size(const void* handle);
TF_EXPORT void tf_storage_copy_from(void* handle, const void* src);
TF_EXPORT void tf_storage_copy_to(const void* handle, void* dst);
TF_EXPORT void tf_storage_fill(void* handle, double value);
TF_EXPORT void tf_storage_scale(void* handle, double factor);

TF_EXPORT void tf_core_sum(const void* src, void* dst,
                           const std::int64_t* shape,
                           const std::int64_t* in_strides,
                           const std::int64_t* out_strides,
                           std::int64_t offset, std::int64_t ndim);
TF_EXPORT void tf_core_narrow_backward(const void* upstream, void* dst,
                                       const std::int64_t* shape,
                                       const std::int64_t* u_strides,
                                       const std::int64_t* out_strides,
                                       std::int64_t u_offset,
                                       std::int64_t out_offset,
                                       std::int64_t ndim);
TF_EXPORT void tf_core_matmul(const void* a, const void* b, void* dst,
                              std::int64_t m, std::int64_t n, std::int64_t p,
                              std::int64_t a_stride0, std::int64_t a_stride1,
                              std::int64_t b_stride0, std::int64_t b_stride1,
                              std::int64_t a_offset, std::int64_t b_offset);
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

// ---------------------------------------------------------------------------
// Bit-pattern plumbing and per-dtype traits, in the shape
// test_dtype_elementwise established so the files read alike.
// ---------------------------------------------------------------------------

template <class T> struct BitsOf;
template <> struct BitsOf<double> { using type = std::uint64_t; };
template <> struct BitsOf<float> { using type = std::uint32_t; };

template <class T>
typename BitsOf<T>::type bits(T value) {
    typename BitsOf<T>::type out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

template <class T>
T from_bits(typename BitsOf<T>::type pattern) {
    T out;
    std::memcpy(&out, &pattern, sizeof(out));
    return out;
}

template <class T> struct DtypeTraits;
template <> struct DtypeTraits<double> {
    static constexpr std::int32_t code = TF_DTYPE_FLOAT64;
    static const char* name() { return "float64"; }
    static double sentinel() { return from_bits<double>(0x4B2D000000000000ull); }
};
template <> struct DtypeTraits<float> {
    static constexpr std::int32_t code = TF_DTYPE_FLOAT32;
    static const char* name() { return "float32"; }
    static float sentinel() { return from_bits<float>(0x4B2D0000u); }
};

template <class T>
void* storage_of(const std::vector<T>& values) {
    void* handle = tf_storage_create_typed(
        static_cast<std::int64_t>(values.size()), DtypeTraits<T>::code);
    if (handle != nullptr) {
        tf_storage_copy_from(handle, values.data());
    }
    return handle;
}

template <class T>
std::vector<T> read_back(const void* handle) {
    std::vector<T> out(static_cast<std::size_t>(tf_storage_size(handle)));
    tf_storage_copy_to(handle, out.data());
    return out;
}

template <class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b,
               std::int64_t n, const char* what) {
    for (std::int64_t i = 0; i < n; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        if (bits<T>(a[k]) != bits<T>(b[k])) {
            char message[256];
            std::snprintf(message, sizeof message,
                          "%.150s [%.16s]: element %lld differs in bits", what,
                          DtypeTraits<T>::name(), static_cast<long long>(i));
            check(false, message);
            return false;
        }
    }
    return true;
}

void note(char* buffer, std::size_t size, const char* what, const char* dtype) {
    std::snprintf(buffer, size, "%.180s [%.16s]", what, dtype);
}

// ---------------------------------------------------------------------------
// Independently written references.
//
// Written here, in ``T``, rather than reached for from the production
// traversals: a test that called ``tf::sum_generic_strided`` for its
// expectation would be comparing the kernel against itself.
// ---------------------------------------------------------------------------

// The reduction, as an ordinary odometer accumulating in ``T``.
template <class T>
std::vector<T> reference_sum(const std::vector<T>& src, std::int64_t out_size,
                             const std::int64_t* shape,
                             const std::int64_t* in_strides,
                             const std::int64_t* out_strides,
                             std::int64_t offset, std::int64_t ndim) {
    std::vector<T> dst(static_cast<std::size_t>(out_size), T(0));
    if (ndim == 0) {
        dst[0] += src[static_cast<std::size_t>(offset)];
        return dst;
    }
    std::int64_t total = 1;
    for (std::int64_t d = 0; d < ndim; ++d) total *= shape[d];
    std::vector<std::int64_t> counter(static_cast<std::size_t>(ndim), 0);
    std::int64_t in_pos = offset;
    std::int64_t out_pos = 0;
    for (std::int64_t i = 0; i < total; ++i) {
        dst[static_cast<std::size_t>(out_pos)] +=
            src[static_cast<std::size_t>(in_pos)];
        for (std::int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[static_cast<std::size_t>(d)];
            in_pos += in_strides[d];
            out_pos += out_strides[d];
            if (counter[static_cast<std::size_t>(d)] < shape[d]) break;
            counter[static_cast<std::size_t>(d)] = 0;
            in_pos -= shape[d] * in_strides[d];
            out_pos -= shape[d] * out_strides[d];
        }
    }
    return dst;
}

// The matmul, as the textbook i-j-k loop accumulating in ``T``.
template <class T>
std::vector<T> reference_matmul(const std::vector<T>& a, const std::vector<T>& b,
                                std::int64_t m, std::int64_t n, std::int64_t p,
                                std::int64_t a_stride0, std::int64_t a_stride1,
                                std::int64_t b_stride0, std::int64_t b_stride1,
                                std::int64_t a_offset, std::int64_t b_offset) {
    std::vector<T> dst(static_cast<std::size_t>(m * p), T(0));
    for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t j = 0; j < p; ++j) {
            T sum = T(0);
            for (std::int64_t k = 0; k < n; ++k) {
                sum += a[static_cast<std::size_t>(a_offset + i * a_stride0
                                                  + k * a_stride1)]
                     * b[static_cast<std::size_t>(b_offset + k * b_stride0
                                                  + j * b_stride1)];
            }
            dst[static_cast<std::size_t>(i * p + j)] = sum;
        }
    }
    return dst;
}

// ---------------------------------------------------------------------------
// 1. Reductions: both traversals, every layout, per dtype
// ---------------------------------------------------------------------------

// One reduction case, driven through the export and compared against the
// independent reference. ``expect_blocks`` records which traversal the
// metadata predicate selects, so the test knows — and asserts — which of the
// two shipped paths actually ran.
template <class T>
void run_reduction(const std::vector<T>& values, std::int64_t out_size,
                   const std::vector<std::int64_t>& shape,
                   const std::vector<std::int64_t>& in_strides,
                   const std::vector<std::int64_t>& out_strides,
                   std::int64_t offset, bool expect_blocks, const char* label) {
    const std::int64_t ndim = static_cast<std::int64_t>(shape.size());
    char message[256];

    std::int64_t outer = 0, mid = 0, inner = 0;
    const bool blocks = ndim > 0 && tf::reduce_prefers_contiguous_blocks(
        shape.data(), in_strides.data(), out_strides.data(), ndim,
        &outer, &mid, &inner);
    note(message, sizeof message, label, DtypeTraits<T>::name());
    if (blocks != expect_blocks) {
        char detail[320];
        std::snprintf(detail, sizeof detail,
                      "%.200s: predicate chose %s, expected %s", message,
                      blocks ? "blocks" : "odometer",
                      expect_blocks ? "blocks" : "odometer");
        check(false, detail);
    }

    void* src = storage_of(values);
    void* dst = tf_storage_create_typed(out_size, DtypeTraits<T>::code);
    if (src == nullptr || dst == nullptr) {
        check(false, "reduction storage");
        if (src) tf_storage_destroy(src);
        if (dst) tf_storage_destroy(dst);
        return;
    }
    tf_clear_error();
    tf_core_sum(src, dst, shape.data(), in_strides.data(), out_strides.data(),
                offset, ndim);
    char detail[320];
    std::snprintf(detail, sizeof detail, "%.200s: error set", message);
    check(tf_last_error_code() == TF_OK, detail);

    const std::vector<T> produced = read_back<T>(dst);
    const std::vector<T> expected = reference_sum<T>(
        values, out_size, shape.data(), in_strides.data(), out_strides.data(),
        offset, ndim);
    std::snprintf(detail, sizeof detail, "%.200s: reference", message);
    same_bits<T>(produced, expected, out_size, detail);

    // The destination tag and logical size are immutable across the call,
    // and the source is never written.
    std::snprintf(detail, sizeof detail, "%.200s: dtype tag moved", message);
    check(tf::storage_dtype(dst) == tf::storage_dtype(src), detail);
    std::snprintf(detail, sizeof detail, "%.200s: element count moved", message);
    check(tf_storage_size(dst) == out_size, detail);
    const std::vector<T> source_after = read_back<T>(src);
    std::snprintf(detail, sizeof detail, "%.200s: the source was mutated",
                  message);
    same_bits<T>(source_after, values,
                 static_cast<std::int64_t>(values.size()), detail);

    tf_storage_destroy(src);
    tf_storage_destroy(dst);
}

// Every supported axis form and every layout family, at one dtype.
template <class T>
void test_reduction_layouts() {
    // A 3 x 4 source with distinct, exactly representable values.
    std::vector<T> values(12);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<T>(static_cast<double>(i) + 1.0);
    }

    // -- full reduction: one contiguous run, the block path's hottest shape.
    run_reduction<T>(values, 1, {3, 4}, {4, 1}, {0, 0}, 0, true,
                     "sum: full reduction");
    // -- one axis, leading (reduced run is a prefix, inner > 1).
    run_reduction<T>(values, 4, {3, 4}, {4, 1}, {0, 1}, 0, true,
                     "sum: axis 0");
    // -- one axis, trailing (reduced run is a suffix, inner == 1).
    run_reduction<T>(values, 3, {3, 4}, {4, 1}, {1, 0}, 0, true,
                     "sum: axis 1");
    // -- keepdims changes no write stride: a retained reduced axis has
    //    write stride 0 either way, so the same call answers for both.
    run_reduction<T>(values, 4, {3, 4}, {4, 1}, {0, 1}, 0, true,
                     "sum: axis 0 keepdims");

    // -- rank 3, one axis, and adjacent axes as one run.
    std::vector<T> cube(24);
    for (std::size_t i = 0; i < cube.size(); ++i) {
        cube[i] = static_cast<T>(static_cast<double>(i) - 11.0);
    }
    run_reduction<T>(cube, 12, {2, 3, 4}, {12, 4, 1}, {0, 4, 1}, 0, true,
                     "sum: rank 3, axis 0");
    run_reduction<T>(cube, 6, {2, 3, 4}, {12, 4, 1}, {3, 1, 0}, 0, true,
                     "sum: rank 3, axis 2");
    run_reduction<T>(cube, 4, {2, 3, 4}, {12, 4, 1}, {0, 0, 1}, 0, true,
                     "sum: rank 3, adjacent axes 0 and 1");
    // -- non-adjacent axes: a kept axis interrupts the reduced run, so the
    //    predicate declines and the retained odometer answers. (The Python
    //    layer cannot express this today; the ABI can, and the kernel must.)
    run_reduction<T>(cube, 3, {2, 3, 4}, {12, 4, 1}, {0, 1, 0}, 0, false,
                     "sum: rank 3, non-adjacent axes 0 and 2");
    // -- rank 4, the widest a NCHW tensor reaches.
    run_reduction<T>(cube, 12, {2, 3, 2, 2}, {12, 4, 2, 1}, {0, 4, 2, 1}, 0,
                     true, "sum: rank 4, axis 0");

    // -- transposed source: not row-major, so the odometer answers.
    run_reduction<T>(values, 3, {4, 3}, {1, 4}, {0, 1}, 0, false,
                     "sum: transposed source");
    // -- narrowed source with a nonzero offset.
    run_reduction<T>(values, 2, {3, 2}, {4, 1}, {0, 1}, 1, false,
                     "sum: narrowed source at an offset");
    // -- negative stride: the odometer addresses it, walking backwards.
    run_reduction<T>(values, 4, {3, 4}, {-4, 1}, {0, 1}, 8, false,
                     "sum: negative leading stride");
    // -- broadcast (zero-stride) source: one element read many times.
    run_reduction<T>(values, 4, {3, 4}, {0, 1}, {0, 1}, 0, false,
                     "sum: zero-stride (broadcast) source");
    // -- rank 0: a scalar source is its own sum.
    run_reduction<T>(values, 1, {}, {}, {}, 5, false, "sum: rank 0");
}

// The layouts both traversals can address must produce identical bits.
// Driven at the hidden traversals directly, so the comparison is between the
// two shipped paths rather than between one path and a reference.
template <class T>
void test_both_reduction_traversals_agree() {
    std::vector<T> values(24);
    for (std::size_t i = 0; i < values.size(); ++i) {
        // A mix of magnitudes, so cancellation and rounding actually occur.
        values[i] = static_cast<T>((static_cast<double>(i) - 12.0) * 0.375
                                   + (i % 3 == 0 ? 1e7 : -1e-3));
    }
    // Reducing all axes; axis 0 (output (3, 4)); axis 2 (output (2, 3));
    // axes 0 and 1 together (output (4,)). Each kept axis carries the
    // row-major stride of the output formed by dropping the reduced run,
    // which is what the predicate requires.
    const std::vector<std::vector<std::int64_t>> out_strides = {
        {0, 0, 0}, {0, 4, 1}, {3, 1, 0}, {0, 0, 1},
    };
    const std::vector<std::int64_t> shape = {2, 3, 4};
    const std::vector<std::int64_t> in_strides = {12, 4, 1};
    char message[256];
    for (const std::vector<std::int64_t>& writes : out_strides) {
        std::int64_t outer = 0, mid = 0, inner = 0;
        if (!tf::reduce_prefers_contiguous_blocks(
                shape.data(), in_strides.data(), writes.data(), 3,
                &outer, &mid, &inner)) {
            check(false, "traversal parity: predicate declined a block layout");
            continue;
        }
        std::vector<T> blocks(static_cast<std::size_t>(outer * inner), T(0));
        std::vector<T> generic(static_cast<std::size_t>(outer * inner), T(0));
        std::vector<std::int64_t> counter(3, 0);
        tf::sum_contiguous_blocks(values.data(), blocks.data(), outer, mid,
                                  inner, 0);
        tf::sum_generic_strided(values.data(), generic.data(), shape.data(),
                                in_strides.data(), writes.data(), 0, 3,
                                counter.data());
        note(message, sizeof message,
             "the two reduction traversals disagree", DtypeTraits<T>::name());
        same_bits<T>(blocks, generic,
                     static_cast<std::int64_t>(blocks.size()), message);
    }
}

// Signed zeros, as raw bit patterns, on both traversals — H6's float64
// property restated at binary32 rather than inherited.
template <class T>
void test_reduction_signed_zeros() {
    const std::vector<T> minus_zeros(8, from_bits<T>(
        std::is_same<T, float>::value
            ? static_cast<typename BitsOf<T>::type>(0x80000000u)
            : static_cast<typename BitsOf<T>::type>(0x8000000000000000ull)));
    const std::vector<std::int64_t> shape = {8};
    const std::vector<std::int64_t> in_strides = {1};
    const std::vector<std::int64_t> writes = {0};
    T blocks = T(0);
    T generic = T(0);
    std::vector<std::int64_t> counter(1, 0);
    tf::sum_contiguous_blocks(minus_zeros.data(), &blocks, 1, 8, 1, 0);
    tf::sum_generic_strided(minus_zeros.data(), &generic, shape.data(),
                            in_strides.data(), writes.data(), 0, 1,
                            counter.data());
    char message[256];
    // Both start from the destination's +0.0 (the additive identity a zeroed
    // buffer holds) and +0.0 + -0.0 is +0.0, so the sum of any number of
    // negative zeros is *positive* zero — on both paths, at both widths.
    note(message, sizeof message,
         "a run of -0.0 did not sum to +0.0 on the block path",
         DtypeTraits<T>::name());
    check(bits<T>(blocks) == bits<T>(T(0)) , message);
    note(message, sizeof message,
         "a run of -0.0 did not sum to +0.0 on the generic path",
         DtypeTraits<T>::name());
    check(bits<T>(generic) == bits<T>(T(0)), message);
}

// NaN and infinity: positions and classification agree on both paths, and
// every NaN a reduction produces is quiet. Stated for the reduction in its
// own right rather than inherited from matmul's rule.
template <class T>
void test_reduction_exceptional_values() {
    using U = typename BitsOf<T>::type;
    const U quiet_bit = std::is_same<T, float>::value
        ? static_cast<U>(0x00400000u) : static_cast<U>(0x0008000000000000ull);
    const T inf = std::numeric_limits<T>::infinity();
    const T signalling = from_bits<T>(
        std::is_same<T, float>::value ? static_cast<U>(0x7F800001u)
                                      : static_cast<U>(0x7FF0000000000001ull));
    // One NaN entering an accumulation, +inf + -inf manufacturing one, and
    // an ordinary infinity that must survive.
    const std::vector<std::vector<T>> cases = {
        {T(1), signalling, T(2), T(3)},
        {inf, -inf, T(1), T(2)},
        {inf, T(1), T(2), T(3)},
        {T(1), T(2), T(3), T(4)},
    };
    const std::vector<std::int64_t> shape = {4};
    const std::vector<std::int64_t> in_strides = {1};
    const std::vector<std::int64_t> writes = {0};
    char message[256];
    for (const std::vector<T>& values : cases) {
        T blocks = T(0);
        T generic = T(0);
        std::vector<std::int64_t> counter(1, 0);
        tf::sum_contiguous_blocks(values.data(), &blocks, 1, 4, 1, 0);
        tf::sum_generic_strided(values.data(), &generic, shape.data(),
                                in_strides.data(), writes.data(), 0, 1,
                                counter.data());
        // At most one NaN enters each accumulation here, so the contract is
        // full bit identity — payload included.
        note(message, sizeof message,
             "an exceptional reduction differs between the two traversals",
             DtypeTraits<T>::name());
        check(bits<T>(blocks) == bits<T>(generic), message);
        if (blocks != blocks) {  // NaN
            note(message, sizeof message,
                 "a reduction produced a signalling NaN",
                 DtypeTraits<T>::name());
            check((bits<T>(blocks) & quiet_bit) != 0, message);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. The float32 accumulation witness — the claim I3 could not make
// ---------------------------------------------------------------------------
//
// ``1 + 2^-24 + 2^-24 + ...`` is the classic absorption case. In binary32
// each addend is exactly half an ULP of 1.0, so round-to-nearest-even leaves
// the running total at exactly 1.0 forever; in binary64 the addends
// accumulate and the single final narrowing lands one ULP above. The two
// answers therefore differ in the *last mantissa bit of the result*, which
// is the smallest possible difference and so the sharpest possible test: a
// widening accumulator cannot hide inside it.
//
// It is deterministic, exactly representable at both widths, and independent
// of the toolchain, the optimization level, and the machine.

template <class T>
struct AccumulationWitness;

template <>
struct AccumulationWitness<float> {
    // 1.0 followed by eight copies of 2^-24.
    static std::vector<float> values() {
        std::vector<float> out(9, from_bits<float>(0x33800000u));  // 2^-24
        out[0] = 1.0f;
        return out;
    }
    // Sequential binary32: every addend is absorbed, so the answer is 1.0.
    static float sequential() { return 1.0f; }
    // binary64 accumulation narrowed once: 1 + 8 * 2^-24 = 1 + 2^-21, which
    // rounds to the float immediately above 1.0f.
    static float widened() { return from_bits<float>(0x3F800004u); }
    static bool meaningful() { return true; }
};

template <>
struct AccumulationWitness<double> {
    // The float64 half of the witness exists so the *test* is symmetric and
    // the float64 accumulator is pinned too, but there is no wider type to
    // accumulate in here, so ``sequential`` and ``widened`` coincide and the
    // inequality half is skipped.
    static std::vector<double> values() {
        std::vector<double> out(9, from_bits<double>(0x3CA0000000000000ull));
        out[0] = 1.0;
        return out;
    }
    static double sequential() { return 1.0; }
    static double widened() { return 1.0; }
    static bool meaningful() { return false; }
};

// The witness, run through the reduction — on **both** traversals.
template <class T>
void test_reduction_accumulates_in_the_element_type() {
    const std::vector<T> values = AccumulationWitness<T>::values();
    const std::int64_t n = static_cast<std::int64_t>(values.size());
    char message[256];

    // First, prove the witness is a witness on this toolchain: sequential
    // binary32 and narrowed-binary64 really do differ. Without this, the
    // assertions below could pass vacuously.
    T sequential = T(0);
    for (std::size_t i = 0; i < values.size(); ++i) sequential += values[i];
    double widened = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        widened += static_cast<double>(values[i]);
    }
    const T narrowed = static_cast<T>(widened);
    note(message, sizeof message, "the accumulation witness is not a witness",
         DtypeTraits<T>::name());
    check(bits<T>(sequential) == bits<T>(AccumulationWitness<T>::sequential())
          && bits<T>(narrowed) == bits<T>(AccumulationWitness<T>::widened()),
          message);
    if (AccumulationWitness<T>::meaningful()) {
        note(message, sizeof message,
             "sequential and widened accumulation agree, so nothing is proved",
             DtypeTraits<T>::name());
        check(bits<T>(sequential) != bits<T>(narrowed), message);
    }

    // The optimized block traversal, whose local accumulator is the one that
    // could most plausibly have been declared ``double``.
    T blocks = T(0);
    tf::sum_contiguous_blocks(values.data(), &blocks, 1, n, 1, 0);
    note(message, sizeof message,
         "the block traversal does not accumulate in the element type",
         DtypeTraits<T>::name());
    check(bits<T>(blocks) == bits<T>(sequential), message);

    // The retained odometer, which accumulates through the destination.
    const std::vector<std::int64_t> shape = {n};
    const std::vector<std::int64_t> in_strides = {1};
    const std::vector<std::int64_t> writes = {0};
    T generic = T(0);
    std::vector<std::int64_t> counter(1, 0);
    tf::sum_generic_strided(values.data(), &generic, shape.data(),
                            in_strides.data(), writes.data(), 0, 1,
                            counter.data());
    note(message, sizeof message,
         "the generic traversal does not accumulate in the element type",
         DtypeTraits<T>::name());
    check(bits<T>(generic) == bits<T>(sequential), message);

    // ...and through the export, which is what production actually calls.
    void* src = storage_of(values);
    void* dst = tf_storage_create_typed(1, DtypeTraits<T>::code);
    if (src != nullptr && dst != nullptr) {
        tf_clear_error();
        tf_core_sum(src, dst, shape.data(), in_strides.data(), writes.data(),
                    0, 1);
        const std::vector<T> produced = read_back<T>(dst);
        note(message, sizeof message,
             "tf_core_sum does not accumulate in the element type",
             DtypeTraits<T>::name());
        check(bits<T>(produced[0]) == bits<T>(sequential), message);
        if (AccumulationWitness<T>::meaningful()) {
            note(message, sizeof message,
                 "tf_core_sum produced the WIDENED result — a hidden float64 "
                 "accumulator", DtypeTraits<T>::name());
            check(bits<T>(produced[0]) != bits<T>(narrowed), message);
        }
    } else {
        check(false, "accumulation witness storage");
    }
    if (src) tf_storage_destroy(src);
    if (dst) tf_storage_destroy(dst);
}

// The same witness through matmul: a 1 x n row of ones times an n x 1
// column of the witness values is exactly the witness sum, computed by the
// ``k`` accumulator rather than by a reduction. Both paths are driven —
// with p == 1 the predicate declines the row sweep (below
// MATMUL_MIN_COLUMNS), so the row sweep is exercised separately with the
// witness broadcast across a wide destination.
template <class T>
void test_matmul_accumulates_in_the_element_type() {
    const std::vector<T> witness = AccumulationWitness<T>::values();
    const std::int64_t n = static_cast<std::int64_t>(witness.size());
    const T expected = AccumulationWitness<T>::sequential();
    const T widened = AccumulationWitness<T>::widened();
    const std::vector<T> ones(static_cast<std::size_t>(n), T(1));
    char message[256];

    // -- the retained generic kernel: (1 x n) @ (n x 1).
    {
        std::vector<T> dst(1, DtypeTraits<T>::sentinel());
        check(!tf::matmul_prefers_row_sweep(1, n, 1, 1),
              "matmul witness: p == 1 should decline the row sweep");
        tf::matmul_generic_strided(ones.data(), witness.data(), dst.data(),
                                   1, n, 1, n, 1, 1, 1, 0, 0);
        note(message, sizeof message,
             "the generic matmul does not accumulate in the element type",
             DtypeTraits<T>::name());
        check(bits<T>(dst[0]) == bits<T>(expected), message);
        if (AccumulationWitness<T>::meaningful()) {
            note(message, sizeof message,
                 "the generic matmul produced the WIDENED result",
                 DtypeTraits<T>::name());
            check(bits<T>(dst[0]) != bits<T>(widened), message);
        }
    }

    // -- the H2 row sweep: (1 x n) @ (n x 8), every column the witness.
    {
        const std::int64_t p = 8;
        std::vector<T> wide(static_cast<std::size_t>(n * p));
        for (std::int64_t k = 0; k < n; ++k) {
            for (std::int64_t j = 0; j < p; ++j) {
                wide[static_cast<std::size_t>(k * p + j)] =
                    witness[static_cast<std::size_t>(k)];
            }
        }
        check(tf::matmul_prefers_row_sweep(1, n, p, 1),
              "matmul witness: p == 8 should take the row sweep");
        std::vector<T> sweep(static_cast<std::size_t>(p),
                             DtypeTraits<T>::sentinel());
        tf::matmul_row_sweep(ones.data(), wide.data(), sweep.data(), 1, n, p,
                             n, 1, p, 0, 0);
        for (std::int64_t j = 0; j < p; ++j) {
            note(message, sizeof message,
                 "the row sweep does not accumulate in the element type",
                 DtypeTraits<T>::name());
            check(bits<T>(sweep[static_cast<std::size_t>(j)])
                      == bits<T>(expected), message);
            if (AccumulationWitness<T>::meaningful()) {
                note(message, sizeof message,
                     "the row sweep produced the WIDENED result",
                     DtypeTraits<T>::name());
                check(bits<T>(sweep[static_cast<std::size_t>(j)])
                          != bits<T>(widened), message);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Matmul: H2's four-part contract, restated per dtype
// ---------------------------------------------------------------------------

template <class T>
void test_matmul_paths_agree() {
    // Shapes around the predicate's boundary (MATMUL_MIN_COLUMNS == 8) and
    // across tall / wide / square / small, plus a row count that is not a
    // multiple of MATMUL_ROW_BLOCK so the ragged tail group runs.
    struct Case { std::int64_t m, n, p; };
    const Case cases[] = {
        {1, 1, 8}, {4, 4, 8}, {5, 3, 9}, {7, 6, 8}, {2, 9, 16},
        {9, 2, 8}, {3, 3, 7}, {3, 3, 8}, {16, 16, 16}, {1, 5, 8},
    };
    char message[256];
    for (const Case& c : cases) {
        std::vector<T> a(static_cast<std::size_t>(c.m * c.n));
        std::vector<T> b(static_cast<std::size_t>(c.n * c.p));
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = static_cast<T>(std::sin(static_cast<double>(i) * 0.7) * 3.0);
        }
        for (std::size_t i = 0; i < b.size(); ++i) {
            b[i] = static_cast<T>(std::cos(static_cast<double>(i) * 0.9) * 2.0);
        }
        const std::vector<T> reference = reference_matmul<T>(
            a, b, c.m, c.n, c.p, c.n, 1, c.p, 1, 0, 0);

        std::vector<T> generic(static_cast<std::size_t>(c.m * c.p),
                               DtypeTraits<T>::sentinel());
        tf::matmul_generic_strided(a.data(), b.data(), generic.data(),
                                   c.m, c.n, c.p, c.n, 1, c.p, 1, 0, 0);
        std::snprintf(message, sizeof message,
                      "matmul generic vs reference [%.16s] %lldx%lldx%lld",
                      DtypeTraits<T>::name(), static_cast<long long>(c.m),
                      static_cast<long long>(c.n), static_cast<long long>(c.p));
        same_bits<T>(generic, reference,
                     static_cast<std::int64_t>(reference.size()), message);

        if (!tf::matmul_prefers_row_sweep(c.m, c.n, c.p, 1)) {
            continue;  // below the threshold: only the generic path ships
        }
        std::vector<T> sweep(static_cast<std::size_t>(c.m * c.p),
                             DtypeTraits<T>::sentinel());
        tf::matmul_row_sweep(a.data(), b.data(), sweep.data(), c.m, c.n, c.p,
                             c.n, 1, c.p, 0, 0);
        // Part 1 and part 2 of the contract: the per-output ``k`` order is
        // preserved exactly, so every non-NaN result is bit-identical.
        std::snprintf(message, sizeof message,
                      "matmul sweep vs generic [%.16s] %lldx%lldx%lld",
                      DtypeTraits<T>::name(), static_cast<long long>(c.m),
                      static_cast<long long>(c.n), static_cast<long long>(c.p));
        same_bits<T>(sweep, generic,
                     static_cast<std::int64_t>(generic.size()), message);
    }
}

// Signed zeros, infinities, subnormals, and NaNs through both matmul paths.
template <class T>
void test_matmul_exceptional_values() {
    using U = typename BitsOf<T>::type;
    const U quiet_bit = std::is_same<T, float>::value
        ? static_cast<U>(0x00400000u) : static_cast<U>(0x0008000000000000ull);
    const T inf = std::numeric_limits<T>::infinity();
    const T tiny = std::numeric_limits<T>::denorm_min();
    const T minus_zero = from_bits<T>(
        std::is_same<T, float>::value ? static_cast<U>(0x80000000u)
                                      : static_cast<U>(0x8000000000000000ull));
    const std::int64_t m = 2, n = 3, p = 8;

    std::vector<T> a{T(-1), minus_zero, tiny, inf, T(2), T(0)};
    std::vector<T> b(static_cast<std::size_t>(n * p));
    for (std::int64_t k = 0; k < n; ++k) {
        for (std::int64_t j = 0; j < p; ++j) {
            const std::size_t i = static_cast<std::size_t>(k * p + j);
            if (j == 0) b[i] = T(0);
            else if (j == 1) b[i] = minus_zero;
            else if (j == 2) b[i] = inf;
            else if (j == 3) b[i] = tiny;
            else b[i] = static_cast<T>((k + 1) * (j - 3));
        }
    }
    std::vector<T> generic(static_cast<std::size_t>(m * p),
                           DtypeTraits<T>::sentinel());
    std::vector<T> sweep(static_cast<std::size_t>(m * p),
                         DtypeTraits<T>::sentinel());
    tf::matmul_generic_strided(a.data(), b.data(), generic.data(), m, n, p,
                               n, 1, p, 1, 0, 0);
    tf::matmul_row_sweep(a.data(), b.data(), sweep.data(), m, n, p, n, 1, p,
                         0, 0);
    char message[256];
    for (std::size_t i = 0; i < generic.size(); ++i) {
        const bool g_nan = generic[i] != generic[i];
        const bool s_nan = sweep[i] != sweep[i];
        // Part 3: NaN positions agree and every NaN is quiet.
        note(message, sizeof message, "matmul NaN positions disagree",
             DtypeTraits<T>::name());
        check(g_nan == s_nan, message);
        if (g_nan) {
            note(message, sizeof message, "matmul produced a signalling NaN",
                 DtypeTraits<T>::name());
            check((bits<T>(generic[i]) & quiet_bit) != 0
                  && (bits<T>(sweep[i]) & quiet_bit) != 0, message);
            continue;  // part 4: payload bits are outside the contract
        }
        // Part 2: every non-NaN result bit-identical, signed zeros included.
        note(message, sizeof message,
             "a non-NaN matmul result differs between the two paths",
             DtypeTraits<T>::name());
        check(bits<T>(generic[i]) == bits<T>(sweep[i]), message);
    }
}

// The export: strided operands, output dtype, and the source left unmoved.
template <class T>
void test_matmul_export() {
    // (3 x 2) @ (2 x 8) with a **transposed** left operand read through its
    // own strides — the case that proves nothing is materialized first.
    const std::int64_t m = 3, n = 2, p = 8;
    std::vector<T> a_source(static_cast<std::size_t>(n * m));  // stored (2 x 3)
    for (std::size_t i = 0; i < a_source.size(); ++i) {
        a_source[i] = static_cast<T>(static_cast<double>(i) * 0.5 - 1.0);
    }
    std::vector<T> b(static_cast<std::size_t>(n * p));
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<T>(static_cast<double>(i) * 0.25 + 0.125);
    }
    void* ah = storage_of(a_source);
    void* bh = storage_of(b);
    void* dh = tf_storage_create_typed(m * p, DtypeTraits<T>::code);
    if (ah == nullptr || bh == nullptr || dh == nullptr) {
        check(false, "matmul export storage");
        if (ah) tf_storage_destroy(ah);
        if (bh) tf_storage_destroy(bh);
        if (dh) tf_storage_destroy(dh);
        return;
    }
    tf_clear_error();
    // a is the transpose of the stored (2 x 3): strides (1, 3).
    tf_core_matmul(ah, bh, dh, m, n, p, 1, m, p, 1, 0, 0);
    char message[256];
    note(message, sizeof message, "matmul export: error set",
         DtypeTraits<T>::name());
    check(tf_last_error_code() == TF_OK, message);
    const std::vector<T> produced = read_back<T>(dh);
    const std::vector<T> expected = reference_matmul<T>(
        a_source, b, m, n, p, 1, m, p, 1, 0, 0);
    note(message, sizeof message, "matmul export vs reference",
         DtypeTraits<T>::name());
    same_bits<T>(produced, expected, m * p, message);
    note(message, sizeof message, "matmul export: dtype tag moved",
         DtypeTraits<T>::name());
    check(tf::storage_dtype(dh) == tf::storage_dtype(ah), message);
    note(message, sizeof message, "matmul export: element count moved",
         DtypeTraits<T>::name());
    check(tf_storage_size(dh) == m * p, message);
    const std::vector<T> a_after = read_back<T>(ah);
    note(message, sizeof message, "matmul export: an operand was mutated",
         DtypeTraits<T>::name());
    same_bits<T>(a_after, a_source,
                 static_cast<std::int64_t>(a_source.size()), message);
    tf_storage_destroy(ah);
    tf_storage_destroy(bh);
    tf_storage_destroy(dh);
}

// ---------------------------------------------------------------------------
// 4. narrow_backward: a scatter, at both dtypes
// ---------------------------------------------------------------------------

template <class T>
void test_narrow_backward() {
    using U = typename BitsOf<T>::type;
    // The upstream carries values a *computation* would normalize away, so
    // the "this assigns, it does not compute" claim is testable: a negative
    // zero and a signalling NaN both have to survive verbatim.
    const T minus_zero = from_bits<T>(
        std::is_same<T, float>::value ? static_cast<U>(0x80000000u)
                                      : static_cast<U>(0x8000000000000000ull));
    const T signalling = from_bits<T>(
        std::is_same<T, float>::value ? static_cast<U>(0x7F800001u)
                                      : static_cast<U>(0x7FF0000000000001ull));
    // Upstream (2 x 2) scattered into a (4 x 2) parent at start == 1 on
    // axis 0: destination rows 1 and 2 are written, 0 and 3 stay zero.
    const std::vector<T> upstream{T(3), minus_zero, signalling, T(-7)};
    const std::vector<std::int64_t> shape = {2, 2};
    const std::vector<std::int64_t> u_strides = {2, 1};
    const std::vector<std::int64_t> out_strides = {2, 1};  // parent (4 x 2)

    void* uh = storage_of(upstream);
    void* dh = tf_storage_create_typed(8, DtypeTraits<T>::code);
    if (uh == nullptr || dh == nullptr) {
        check(false, "narrow_backward storage");
        if (uh) tf_storage_destroy(uh);
        if (dh) tf_storage_destroy(dh);
        return;
    }
    tf_clear_error();
    tf_core_narrow_backward(uh, dh, shape.data(), u_strides.data(),
                            out_strides.data(), 0, /*out_offset=*/2, 2);
    char message[256];
    note(message, sizeof message, "narrow_backward: error set",
         DtypeTraits<T>::name());
    check(tf_last_error_code() == TF_OK, message);
    const std::vector<T> produced = read_back<T>(dh);

    // The narrowed region reproduces its source's bits exactly — an
    // assignment performs no arithmetic, so -0.0 stays negative and a
    // signalling NaN stays signalling, at both widths.
    for (std::size_t i = 0; i < upstream.size(); ++i) {
        note(message, sizeof message,
             "narrow_backward did not reproduce the upstream's bits",
             DtypeTraits<T>::name());
        check(bits<T>(produced[i + 2]) == bits<T>(upstream[i]), message);
    }
    // Everything outside it keeps the zero the allocation gave it — and
    // *that zero is the gradient*, which is why H1 rejected this
    // destination from the uninitialized path.
    for (std::size_t i : {std::size_t(0), std::size_t(1), std::size_t(6),
                          std::size_t(7)}) {
        note(message, sizeof message,
             "narrow_backward wrote outside the narrowed region",
             DtypeTraits<T>::name());
        check(bits<T>(produced[i]) == bits<T>(T(0)), message);
    }
    note(message, sizeof message, "narrow_backward: dtype tag moved",
         DtypeTraits<T>::name());
    check(tf::storage_dtype(dh) == tf::storage_dtype(uh), message);

    // A strided upstream (transposed) scatters correctly too.
    void* d2 = tf_storage_create_typed(8, DtypeTraits<T>::code);
    if (d2 != nullptr) {
        const std::vector<std::int64_t> transposed = {1, 2};
        tf_clear_error();
        tf_core_narrow_backward(uh, d2, shape.data(), transposed.data(),
                                out_strides.data(), 0, 2, 2);
        const std::vector<T> got = read_back<T>(d2);
        const std::size_t order[4] = {0, 2, 1, 3};
        for (std::size_t i = 0; i < 4; ++i) {
            note(message, sizeof message,
                 "narrow_backward mis-scattered a transposed upstream",
                 DtypeTraits<T>::name());
            check(bits<T>(got[i + 2]) == bits<T>(upstream[order[i]]), message);
        }
        tf_storage_destroy(d2);
    }
    tf_storage_destroy(uh);
    tf_storage_destroy(dh);
}

// ---------------------------------------------------------------------------
// 5. The mean scalar: narrowed once, before the loop
// ---------------------------------------------------------------------------

void test_scale_narrows_the_scalar_once() {
    // count == 3, so 1/3 is inexact at both widths. The distinction the
    // test draws:
    //
    //   narrow-then-multiply : float(value) * float(1.0/3)   <- specified
    //   multiply-then-narrow : float(double(value) * (1.0/3))
    //
    // A value is chosen for which those two differ in the last bit, so the
    // assertion is not vacuous.
    const double factor = 1.0 / 3.0;
    const float value = 7.0f;
    const float specified = value * static_cast<float>(factor);
    const float alternative = static_cast<float>(static_cast<double>(value)
                                                 * factor);
    check(bits<float>(specified) != bits<float>(alternative),
          "the scale witness is not a witness: both orders agree");

    std::vector<float> values(4, value);
    void* handle = storage_of(values);
    if (handle == nullptr) {
        check(false, "scale storage");
        return;
    }
    tf_storage_scale(handle, factor);
    const std::vector<float> produced = read_back<float>(handle);
    for (std::size_t i = 0; i < produced.size(); ++i) {
        check(bits<float>(produced[i]) == bits<float>(specified),
              "tf_storage_scale did not narrow the scalar before the loop");
        check(bits<float>(produced[i]) != bits<float>(alternative),
              "tf_storage_scale multiplied in binary64 and narrowed after");
    }
    tf_storage_destroy(handle);

    // float64 is unmoved: the same call is the pre-I4 statement.
    std::vector<double> wide(4, 7.0);
    void* dh = storage_of(wide);
    if (dh != nullptr) {
        tf_storage_scale(dh, factor);
        const std::vector<double> got = read_back<double>(dh);
        for (std::size_t i = 0; i < got.size(); ++i) {
            check(bits<double>(got[i]) == bits<double>(7.0 * factor),
                  "float64 scale moved");
        }
        tf_storage_destroy(dh);
    }
}

void test_fill_narrows_the_scalar_once() {
    // 0.1 is inexact at both widths; float(0.1) is the only sensible
    // reading of "fill a float32 tensor with 0.1", and it is documented
    // rather than silent (design §7.4).
    void* handle = tf_storage_create_typed(5, TF_DTYPE_FLOAT32);
    if (handle == nullptr) {
        check(false, "fill storage");
        return;
    }
    tf_storage_fill(handle, 0.1);
    const std::vector<float> produced = read_back<float>(handle);
    for (std::size_t i = 0; i < produced.size(); ++i) {
        check(bits<float>(produced[i]) == bits<float>(0.1f),
              "tf_storage_fill did not narrow the scalar to the element type");
    }
    check(tf_storage_size(handle) == 5, "fill moved the logical size");
    tf_storage_destroy(handle);

    void* wide = tf_storage_create_typed(5, TF_DTYPE_FLOAT64);
    if (wide != nullptr) {
        tf_storage_fill(wide, 0.1);
        const std::vector<double> got = read_back<double>(wide);
        for (std::size_t i = 0; i < got.size(); ++i) {
            check(bits<double>(got[i]) == bits<double>(0.1),
                  "float64 fill moved");
        }
        tf_storage_destroy(wide);
    }
}

// ---------------------------------------------------------------------------
// 6. Mixed dtype is rejected before anything is written
// ---------------------------------------------------------------------------

void test_mixed_dtype_is_rejected_before_any_write() {
    const std::vector<double> wide(16, 2.0);
    const std::vector<float> narrow(16, 2.0f);
    // Every participating handle position gets its own case, so a guard that
    // only checked two of three would fail here.
    const std::int64_t shape[2] = {4, 4};
    const std::int64_t strides[2] = {4, 1};
    const std::int64_t writes[2] = {0, 1};

    struct Case { const char* what; bool a_wide, b_wide, dst_wide; };
    const Case cases[] = {
        {"sum: float32 source, float64 destination", false, false, true},
        {"sum: float64 source, float32 destination", true, true, false},
        {"matmul: left operand differs", false, true, true},
        {"matmul: right operand differs", true, false, true},
        {"matmul: destination differs", true, true, false},
        {"narrow_backward: upstream differs", false, false, true},
    };
    for (const Case& c : cases) {
        void* a = c.a_wide ? storage_of(wide) : storage_of(narrow);
        void* b = c.b_wide ? storage_of(wide) : storage_of(narrow);
        void* dst = c.dst_wide ? storage_of(wide) : storage_of(narrow);
        if (a == nullptr || b == nullptr || dst == nullptr) {
            check(false, "mixed-dtype storage");
            if (a) tf_storage_destroy(a);
            if (b) tf_storage_destroy(b);
            if (dst) tf_storage_destroy(dst);
            continue;
        }
        // A snapshot of the destination's raw bytes: a rejected call must
        // leave it byte-for-byte unchanged, which is stronger than "the
        // values look right".
        std::vector<double> before_wide;
        std::vector<float> before_narrow;
        if (c.dst_wide) before_wide = read_back<double>(dst);
        else before_narrow = read_back<float>(dst);

        tf_clear_error();
        if (std::strncmp(c.what, "sum", 3) == 0) {
            tf_core_sum(a, dst, shape, strides, writes, 0, 2);
        } else if (std::strncmp(c.what, "matmul", 6) == 0) {
            tf_core_matmul(a, b, dst, 4, 4, 4, 4, 1, 4, 1, 0, 0);
        } else {
            tf_core_narrow_backward(a, dst, shape, strides, strides, 0, 0, 2);
        }
        char message[256];
        std::snprintf(message, sizeof message, "%.180s: not rejected", c.what);
        check(tf_last_error_code() == TF_ERROR_INVALID, message);
        std::snprintf(message, sizeof message,
                      "%.180s: the destination was written", c.what);
        if (c.dst_wide) {
            same_bits<double>(read_back<double>(dst), before_wide, 16, message);
        } else {
            same_bits<float>(read_back<float>(dst), before_narrow, 16, message);
        }
        tf_storage_destroy(a);
        tf_storage_destroy(b);
        tf_storage_destroy(dst);
    }
    tf_clear_error();
}

// Matching dtypes must NOT be rejected — the negative control that keeps the
// test above from passing because everything is refused.
void test_matching_dtype_is_not_rejected() {
    const std::vector<float> narrow(16, 2.0f);
    const std::int64_t shape[2] = {4, 4};
    const std::int64_t strides[2] = {4, 1};
    const std::int64_t writes[2] = {0, 1};
    void* a = storage_of(narrow);
    void* b = storage_of(narrow);
    void* dst = storage_of(narrow);
    if (a != nullptr && b != nullptr && dst != nullptr) {
        tf_clear_error();
        tf_core_sum(a, dst, shape, strides, writes, 0, 2);
        check(tf_last_error_code() == TF_OK, "float32 sum was refused");
        tf_core_matmul(a, b, dst, 4, 4, 4, 4, 1, 4, 1, 0, 0);
        check(tf_last_error_code() == TF_OK, "float32 matmul was refused");
        tf_core_narrow_backward(a, dst, shape, strides, strides, 0, 0, 2);
        check(tf_last_error_code() == TF_OK,
              "float32 narrow_backward was refused");
    } else {
        check(false, "matching-dtype storage");
    }
    if (a) tf_storage_destroy(a);
    if (b) tf_storage_destroy(b);
    if (dst) tf_storage_destroy(dst);
}

}  // namespace

int main() {
    test_reduction_layouts<double>();
    test_reduction_layouts<float>();

    test_both_reduction_traversals_agree<double>();
    test_both_reduction_traversals_agree<float>();

    test_reduction_signed_zeros<double>();
    test_reduction_signed_zeros<float>();

    test_reduction_exceptional_values<double>();
    test_reduction_exceptional_values<float>();

    test_reduction_accumulates_in_the_element_type<double>();
    test_reduction_accumulates_in_the_element_type<float>();

    test_matmul_accumulates_in_the_element_type<double>();
    test_matmul_accumulates_in_the_element_type<float>();

    test_matmul_paths_agree<double>();
    test_matmul_paths_agree<float>();

    test_matmul_exceptional_values<double>();
    test_matmul_exceptional_values<float>();

    test_matmul_export<double>();
    test_matmul_export<float>();

    test_narrow_backward<double>();
    test_narrow_backward<float>();

    test_scale_narrows_the_scalar_once();
    test_fill_narrows_the_scalar_once();

    test_mixed_dtype_is_rejected_before_any_write();
    test_matching_dtype_is_not_rejected();

    if (g_failures == 0) {
        std::printf("dtype reduction/matmul: all checks passed\n");
        return 0;
    }
    std::printf("dtype reduction/matmul: %d check(s) failed\n", g_failures);
    return 1;
}
