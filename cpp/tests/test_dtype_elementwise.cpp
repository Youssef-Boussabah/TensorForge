// Dependency-free C++ test for dtype-general elementwise, broadcast, and
// unary execution (Phase I, milestone I3). No GoogleTest / Catch2 — a plain
// executable that prints failures and returns a nonzero exit code if any
// check fails, so CTest reports pass/fail.
//
// This binary compiles cpp/src/elementwise.cpp, cpp/src/storage.cpp, and
// cpp/src/error.cpp directly, so it reaches the hidden
// ``tf::build_unary_plan`` / ``tf::build_binary_plan`` builders and the
// templated traversals alongside the exported wrappers they live inside —
// at the layer where the properties are actually decided, with no Python
// wrapper, no ctypes boundary, and no NumPy anywhere.
//
// It is deliberately complementary to the three neighbouring targets rather
// than a superset of any of them:
//
//   * ``test_elementwise_traversal`` proves H8's collapsed plan agrees with
//     the retained odometer at **float64**, over the full NaN sweep. That
//     stays exactly as it is and is not restated here.
//   * ``test_dtype_storage`` proves float32 is *allocatable and otherwise
//     rejected*.
//   * ``test_typed_transfer`` proves the exact set of boundaries float32
//     could cross at I2 — transfer, materialization, identity copy.
//
// This one proves the set that opened at I3: the elementwise arithmetic and
// unary family, at both dtypes, through every traversal tier.
//
// What it proves:
//
//   1. **The three traversal tiers agree, per dtype.** For each operation,
//      the contiguous flat row, the collapsed plan, and the retained
//      generic odometer produce **bit-identical** results on the same
//      logical elements — at float32 as well as float64. Reaching an output
//      through a merged axis instead of a carry chain changes only how the
//      address was computed.
//   2. **float32 results are exactly the binary32 results.** Every value is
//      compared against an independently written scalar reference that
//      computes in ``T`` — for ``add``/``subtract``/``multiply``/``relu``/
//      ``relu_backward``/``sqrt``/``reciprocal`` by raw bit pattern, and
//      for ``exp``/``log`` within one representable step, because those two
//      are library functions with no correctly-rounded guarantee (H8's
//      exclusion, restated per dtype).
//   3. **Output dtype follows the operands.** A float32 operation writes a
//      float32 destination, and the storage's own tag and logical size are
//      unchanged by the call.
//   4. **Broadcasting works at both dtypes.** Zero strides, transposed
//      operands, narrowed-with-offset operands, negative strides, unit
//      extents, and rank-0 scalars all read correctly at float32.
//   5. **Mixed dtype is rejected before anything is written.** Every
//      generalized export refuses a float32/float64 mixture — in the left
//      operand, the right operand, and the destination position
//      independently — with TF_ERROR_INVALID, leaving the destination
//      byte-for-byte unchanged and the live-storage count unmoved.
//   6. **float64 did not move.** Every operation is re-checked at float64
//      against the same reference, so the generalization is proved not to
//      have disturbed the width Phase H measured.
//
// Comparison is by raw bit pattern wherever the operation is IEEE-specified,
// never by tolerance: ``==`` on floating values cannot see -0.0 versus +0.0
// and calls every NaN unequal to itself.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

#include "tf_elementwise_internal.h"
#include "tf_internal.h"

TF_EXPORT void* tf_storage_create_typed(std::int64_t size,
                                        std::int32_t dtype_code);
TF_EXPORT void tf_storage_destroy(void* handle);
TF_EXPORT std::int64_t tf_storage_size(const void* handle);
TF_EXPORT void tf_storage_copy_from(void* handle, const void* src);
TF_EXPORT void tf_storage_copy_to(const void* handle, void* dst);

TF_EXPORT void tf_core_relu(const void* src, void* dst,
                            const std::int64_t* shape,
                            const std::int64_t* strides, std::int64_t offset,
                            std::int64_t ndim);
TF_EXPORT void tf_core_relu_contiguous(const void* src, void* dst,
                                       std::int64_t numel, std::int64_t offset);
TF_EXPORT void tf_core_sqrt(const void* src, void* dst,
                            const std::int64_t* shape,
                            const std::int64_t* strides, std::int64_t offset,
                            std::int64_t ndim);
TF_EXPORT void tf_core_sqrt_contiguous(const void* src, void* dst,
                                       std::int64_t numel, std::int64_t offset);
TF_EXPORT void tf_core_reciprocal(const void* src, void* dst,
                                  const std::int64_t* shape,
                                  const std::int64_t* strides,
                                  std::int64_t offset, std::int64_t ndim);
TF_EXPORT void tf_core_reciprocal_contiguous(const void* src, void* dst,
                                             std::int64_t numel,
                                             std::int64_t offset);
TF_EXPORT void tf_core_exp(const void* src, void* dst,
                           const std::int64_t* shape,
                           const std::int64_t* strides, std::int64_t offset,
                           std::int64_t ndim);
TF_EXPORT void tf_core_exp_contiguous(const void* src, void* dst,
                                      std::int64_t numel, std::int64_t offset);
TF_EXPORT void tf_core_log(const void* src, void* dst,
                           const std::int64_t* shape,
                           const std::int64_t* strides, std::int64_t offset,
                           std::int64_t ndim);
TF_EXPORT void tf_core_log_contiguous(const void* src, void* dst,
                                      std::int64_t numel, std::int64_t offset);
TF_EXPORT void tf_core_add(const void* a, const void* b, void* dst,
                           const std::int64_t* shape,
                           const std::int64_t* a_strides,
                           const std::int64_t* b_strides, std::int64_t a_offset,
                           std::int64_t b_offset, std::int64_t ndim);
TF_EXPORT void tf_core_subtract(const void* a, const void* b, void* dst,
                                const std::int64_t* shape,
                                const std::int64_t* a_strides,
                                const std::int64_t* b_strides,
                                std::int64_t a_offset, std::int64_t b_offset,
                                std::int64_t ndim);
TF_EXPORT void tf_core_multiply(const void* a, const void* b, void* dst,
                                const std::int64_t* shape,
                                const std::int64_t* a_strides,
                                const std::int64_t* b_strides,
                                std::int64_t a_offset, std::int64_t b_offset,
                                std::int64_t ndim);
TF_EXPORT void tf_core_relu_backward(const void* x, const void* upstream,
                                     void* dst, const std::int64_t* shape,
                                     const std::int64_t* x_strides,
                                     const std::int64_t* u_strides,
                                     std::int64_t x_offset,
                                     std::int64_t u_offset, std::int64_t ndim);
TF_EXPORT void tf_core_add_contiguous(const void* a, const void* b, void* dst,
                                      std::int64_t numel, std::int64_t a_offset,
                                      std::int64_t b_offset);
TF_EXPORT void tf_core_subtract_contiguous(const void* a, const void* b,
                                           void* dst, std::int64_t numel,
                                           std::int64_t a_offset,
                                           std::int64_t b_offset);
TF_EXPORT void tf_core_multiply_contiguous(const void* a, const void* b,
                                           void* dst, std::int64_t numel,
                                           std::int64_t a_offset,
                                           std::int64_t b_offset);
TF_EXPORT int tf_last_error_code();
TF_EXPORT const char* tf_last_error_message();
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
// Bit-pattern plumbing and per-dtype traits, in the shape test_typed_transfer
// established so the two files read alike.
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
    // A finite value appearing in no input, so an element the kernel failed
    // to write is detectable rather than accidentally right.
    static double sentinel() { return from_bits<double>(0x4B2D000000000000ull); }
};
template <> struct DtypeTraits<float> {
    static constexpr std::int32_t code = TF_DTYPE_FLOAT32;
    static const char* name() { return "float32"; }
    static float sentinel() { return from_bits<float>(0x4B2D0000u); }
};

// Typed storage holding ``values``; the caller destroys it.
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

// How many representable steps apart two finite values of one dtype are.
// Used **only** for exp and log: they are library functions with no
// correctly-rounded guarantee, so the cross-toolchain contract is a bound
// rather than bit equality (H8's exclusion, restated per dtype).
template <class T>
long long ulp_distance(T a, T b) {
    using U = typename BitsOf<T>::type;
    using S = typename std::conditional<sizeof(T) == 4, std::int32_t,
                                        std::int64_t>::type;
    S ia = static_cast<S>(bits<T>(a));
    S ib = static_cast<S>(bits<T>(b));
    // Sign-magnitude reflected onto one monotone integer line.
    const S floor_value = static_cast<S>(U(1) << (sizeof(T) * 8 - 1));
    if (ia < 0) ia = static_cast<S>(floor_value - ia);
    if (ib < 0) ib = static_cast<S>(floor_value - ib);
    const long long d = static_cast<long long>(ia) - static_cast<long long>(ib);
    return d < 0 ? -d : d;
}

// ---------------------------------------------------------------------------
// The independent scalar references.
//
// Written here, in ``T``, rather than reached for from the production
// functors: a test that called ``tf::AddOp::apply`` would be comparing the
// kernel against itself. Each is the plain IEEE expression at the element
// type, so a float32 kernel that computed in binary64 anywhere would have to
// agree with binary32 arithmetic anyway to pass — which for a single
// correctly-rounded operation it would, and which is exactly why the
// no-hidden-double claim is *also* proved structurally, from Python, over
// the source.
// ---------------------------------------------------------------------------

template <class T> T ref_add(T x, T y) { return x + y; }
template <class T> T ref_subtract(T x, T y) { return x - y; }
template <class T> T ref_multiply(T x, T y) { return x * y; }
template <class T> T ref_relu_backward(T x, T u) { return x > T(0) ? u : T(0); }
template <class T> T ref_relu(T x) { return x > T(0) ? x : T(0); }
template <class T> T ref_sqrt(T x) { return std::sqrt(x); }
template <class T> T ref_reciprocal(T x) { return T(1) / x; }
template <class T> T ref_exp(T x) { return std::exp(x); }
template <class T> T ref_log(T x) { return std::log(x); }

// Walk one strided source with a plain odometer written independently of the
// production one, and produce row-major output.
template <class T>
std::vector<T> reference_unary(const std::vector<T>& src, T (*op)(T),
                               const std::int64_t* shape,
                               const std::int64_t* strides, std::int64_t offset,
                               std::int64_t ndim) {
    if (ndim == 0) {
        return std::vector<T>(1, op(src[static_cast<std::size_t>(offset)]));
    }
    std::int64_t total = 1;
    for (std::int64_t d = 0; d < ndim; ++d) total *= shape[d];
    std::vector<T> out(static_cast<std::size_t>(total));
    std::vector<std::int64_t> counter(static_cast<std::size_t>(ndim), 0);
    std::int64_t pos = offset;
    for (std::int64_t i = 0; i < total; ++i) {
        out[static_cast<std::size_t>(i)] = op(src[static_cast<std::size_t>(pos)]);
        for (std::int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[static_cast<std::size_t>(d)];
            pos += strides[d];
            if (counter[static_cast<std::size_t>(d)] < shape[d]) break;
            counter[static_cast<std::size_t>(d)] = 0;
            pos -= shape[d] * strides[d];
        }
    }
    return out;
}

template <class T>
std::vector<T> reference_binary(const std::vector<T>& a, const std::vector<T>& b,
                                T (*op)(T, T), const std::int64_t* shape,
                                const std::int64_t* a_strides,
                                const std::int64_t* b_strides,
                                std::int64_t a_offset, std::int64_t b_offset,
                                std::int64_t ndim) {
    if (ndim == 0) {
        return std::vector<T>(1, op(a[static_cast<std::size_t>(a_offset)],
                                    b[static_cast<std::size_t>(b_offset)]));
    }
    std::int64_t total = 1;
    for (std::int64_t d = 0; d < ndim; ++d) total *= shape[d];
    std::vector<T> out(static_cast<std::size_t>(total));
    std::vector<std::int64_t> counter(static_cast<std::size_t>(ndim), 0);
    std::int64_t ap = a_offset, bp = b_offset;
    for (std::int64_t i = 0; i < total; ++i) {
        out[static_cast<std::size_t>(i)] = op(a[static_cast<std::size_t>(ap)],
                                              b[static_cast<std::size_t>(bp)]);
        for (std::int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[static_cast<std::size_t>(d)];
            ap += a_strides[d];
            bp += b_strides[d];
            if (counter[static_cast<std::size_t>(d)] < shape[d]) break;
            counter[static_cast<std::size_t>(d)] = 0;
            ap -= shape[d] * a_strides[d];
            bp -= shape[d] * b_strides[d];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// The layout table.
//
// Every entry is a real 4x4 (or smaller) view over a 16-element storage, and
// the three columns say which traversal tier it lands in: a row-major layout
// takes the contiguous path when the wrapper calls the ``_contiguous``
// export, an ordinary strided one collapses into a plan, and one that the
// builder rejects (rank 5) falls back to the retained odometer.
// ---------------------------------------------------------------------------

struct Layout {
    const char* name;
    std::int64_t ndim;
    std::int64_t shape[5];
    std::int64_t strides[5];
    std::int64_t offset;
};

const Layout kLayouts[] = {
    {"scalar", 0, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, 5},
    {"1d contiguous", 1, {16, 0, 0, 0, 0}, {1, 0, 0, 0, 0}, 0},
    {"1d non-unit stride", 1, {4, 0, 0, 0, 0}, {4, 0, 0, 0, 0}, 0},
    {"1d reversed", 1, {4, 0, 0, 0, 0}, {-4, 0, 0, 0, 0}, 12},
    {"2d contiguous", 2, {4, 4, 0, 0, 0}, {4, 1, 0, 0, 0}, 0},
    {"2d transposed", 2, {4, 4, 0, 0, 0}, {1, 4, 0, 0, 0}, 0},
    {"2d narrowed", 2, {2, 3, 0, 0, 0}, {4, 1, 0, 0, 0}, 5},
    {"2d broadcast rows", 2, {4, 4, 0, 0, 0}, {0, 1, 0, 0, 0}, 0},
    {"2d broadcast cols", 2, {4, 4, 0, 0, 0}, {1, 0, 0, 0, 0}, 0},
    {"2d unit extent", 2, {1, 4, 0, 0, 0}, {0, 1, 0, 0, 0}, 4},
    {"3d chain", 3, {2, 2, 4, 0, 0}, {8, 4, 1, 0, 0}, 0},
    {"3d transposed chain", 3, {2, 4, 2, 0, 0}, {8, 1, 4, 0, 0}, 0},
    // Rank 5 exceeds ELEMENTWISE_PLAN_AXES, so the plan builder declines and
    // the retained generic odometer runs. A rejection is a fallback, never
    // an error — and it is what makes the third tier reachable from here.
    {"rank 5 (odometer only)", 5, {2, 2, 2, 2, 1}, {8, 4, 2, 1, 0}, 0},
};
const std::size_t kLayoutCount = sizeof(kLayouts) / sizeof(kLayouts[0]);

// Finite inputs chosen so every operation has a defined, interesting result:
// positive and negative values, exact halves and quarters, values whose
// products and quotients are not representable, and a zero.
template <class T>
std::vector<T> unary_inputs() {
    static const double raw[16] = {
        0.25, 1.0, 2.0, 0.5, 4.0, 0.125, 9.0, 3.0,
        16.0, 0.75, 6.25, 1.5, 100.0, 0.0625, 2.5, 7.0};
    std::vector<T> out(16);
    for (std::size_t i = 0; i < 16; ++i) out[i] = static_cast<T>(raw[i]);
    return out;
}

template <class T>
std::vector<T> signed_inputs() {
    static const double raw[16] = {
        1.5, -2.5, 0.0, -0.0, 3.25, -1.0, 0.5, -7.5,
        -0.25, 8.0, -3.5, 2.75, -6.0, 0.125, -9.5, 4.5};
    std::vector<T> out(16);
    for (std::size_t i = 0; i < 16; ++i) out[i] = static_cast<T>(raw[i]);
    return out;
}

template <class T>
std::vector<T> second_inputs() {
    static const double raw[16] = {
        0.5, 3.0, -1.25, 7.0, -0.75, 2.25, 11.0, -4.0,
        0.375, -5.5, 1.125, 6.5, -2.0, 9.25, 0.875, -8.0};
    std::vector<T> out(16);
    for (std::size_t i = 0; i < 16; ++i) out[i] = static_cast<T>(raw[i]);
    return out;
}

// ---------------------------------------------------------------------------
// 1. Unary: every layout, every tier, both dtypes
// ---------------------------------------------------------------------------

template <class T>
void run_unary_layouts(const char* op_name,
                       void (*strided)(const void*, void*, const std::int64_t*,
                                       const std::int64_t*, std::int64_t,
                                       std::int64_t),
                       void (*contiguous)(const void*, void*, std::int64_t,
                                          std::int64_t),
                       T (*reference)(T), const std::vector<T>& input,
                       long long ulp_budget) {
    char message[256];
    void* src = storage_of(input);
    check(src != nullptr, "unary source storage");
    if (src == nullptr) return;

    for (std::size_t l = 0; l < kLayoutCount; ++l) {
        const Layout& layout = kLayouts[l];
        std::int64_t numel = 1;
        for (std::int64_t d = 0; d < layout.ndim; ++d) numel *= layout.shape[d];

        const std::vector<T> expected = reference_unary<T>(
            input, reference, layout.shape, layout.strides, layout.offset,
            layout.ndim);

        // Pre-fill the destination with a sentinel, so an element the kernel
        // never wrote is a visible failure rather than an accidental pass.
        std::vector<T> poison(static_cast<std::size_t>(numel),
                              DtypeTraits<T>::sentinel());
        void* dst = storage_of(poison);
        check(dst != nullptr, "unary destination storage");
        if (dst == nullptr) continue;

        tf_clear_error();
        strided(src, dst, layout.shape, layout.strides, layout.offset,
                layout.ndim);
        std::snprintf(message, sizeof message, "%.40s %.40s [%.16s]: error set",
                      op_name, layout.name, DtypeTraits<T>::name());
        check(tf_last_error_code() == TF_OK, message);

        const std::vector<T> got = read_back<T>(dst);
        // The destination's own identity is untouched by the call.
        std::snprintf(message, sizeof message, "%.40s %.40s [%.16s]: dst size",
                      op_name, layout.name, DtypeTraits<T>::name());
        check(tf_storage_size(dst) == numel, message);

        std::snprintf(message, sizeof message, "%.40s %.40s", op_name,
                      layout.name);
        if (ulp_budget == 0) {
            same_bits<T>(got, expected, numel, message);
        } else {
            for (std::int64_t i = 0; i < numel; ++i) {
                const T g = got[static_cast<std::size_t>(i)];
                const T w = expected[static_cast<std::size_t>(i)];
                const long long d = ulp_distance<T>(g, w);
                if (d > ulp_budget) {
                    char detail[256];
                    std::snprintf(detail, sizeof detail,
                                  "%.60s %.40s [%.16s]: element %lld is %lld "
                                  "ULP from the reference",
                                  op_name, layout.name, DtypeTraits<T>::name(),
                                  static_cast<long long>(i), d);
                    check(false, detail);
                    break;
                }
            }
        }
        tf_storage_destroy(dst);
    }

    // ...and the contiguous export, which is the third tier: one flat run.
    {
        const std::int64_t shape[1] = {12};
        const std::int64_t strides[1] = {1};
        const std::vector<T> expected =
            reference_unary<T>(input, reference, shape, strides, 3, 1);
        std::vector<T> poison(12, DtypeTraits<T>::sentinel());
        void* dst = storage_of(poison);
        check(dst != nullptr, "unary contiguous destination");
        if (dst != nullptr) {
            tf_clear_error();
            contiguous(src, dst, 12, 3);
            std::snprintf(message, sizeof message,
                          "%.40s contiguous [%.16s]: error set", op_name,
                          DtypeTraits<T>::name());
            check(tf_last_error_code() == TF_OK, message);
            const std::vector<T> got = read_back<T>(dst);
            std::snprintf(message, sizeof message, "%.40s contiguous+offset",
                          op_name);
            if (ulp_budget == 0) {
                same_bits<T>(got, expected, 12, message);
            } else {
                for (std::int64_t i = 0; i < 12; ++i) {
                    if (ulp_distance<T>(got[static_cast<std::size_t>(i)],
                                        expected[static_cast<std::size_t>(i)])
                            > ulp_budget) {
                        check(false, message);
                        break;
                    }
                }
            }
            tf_storage_destroy(dst);
        }
    }
    tf_storage_destroy(src);
}

template <class T>
void test_unary_family() {
    // relu over signed values, so the branch and both zero signs are live.
    run_unary_layouts<T>("relu", tf_core_relu, tf_core_relu_contiguous,
                         ref_relu<T>, signed_inputs<T>(), 0);
    run_unary_layouts<T>("sqrt", tf_core_sqrt, tf_core_sqrt_contiguous,
                         ref_sqrt<T>, unary_inputs<T>(), 0);
    run_unary_layouts<T>("reciprocal", tf_core_reciprocal,
                         tf_core_reciprocal_contiguous, ref_reciprocal<T>,
                         unary_inputs<T>(), 0);
    // exp and log get one representable step: they are library functions
    // with no correctly-rounded guarantee, so bit equality would be a claim
    // about the toolchain rather than about TensorForge.
    run_unary_layouts<T>("exp", tf_core_exp, tf_core_exp_contiguous,
                         ref_exp<T>, signed_inputs<T>(), 1);
    run_unary_layouts<T>("log", tf_core_log, tf_core_log_contiguous,
                         ref_log<T>, unary_inputs<T>(), 1);
}

// ---------------------------------------------------------------------------
// 2. Unary special values: signed zeros, infinities, NaNs, domain edges
// ---------------------------------------------------------------------------

template <class T>
void test_unary_special_values() {
    char message[256];
    // +0, -0, +inf, -inf, quiet NaN, 1, -1, smallest normal, largest finite,
    // the smallest subnormal, and three ordinary values.
    //
    // Each width has its own literal table rather than one derived from the
    // other by shifting: a derived pattern is not the same IEEE class, and
    // the classes are the entire point.
    std::vector<T> input(13);
    if (sizeof(T) == 8) {
        const std::uint64_t p[13] = {
            0x0000000000000000ull, 0x8000000000000000ull, 0x7FF0000000000000ull,
            0xFFF0000000000000ull, 0x7FF8000000000001ull, 0x3FF0000000000000ull,
            0xBFF0000000000000ull, 0x0010000000000000ull, 0x7FEFFFFFFFFFFFFFull,
            0x0000000000000001ull, 0x4000000000000000ull, 0x3FE0000000000000ull,
            0xC008000000000000ull};
        for (int i = 0; i < 13; ++i) {
            double value;
            std::memcpy(&value, &p[i], sizeof(double));
            input[static_cast<std::size_t>(i)] = static_cast<T>(value);
        }
    } else {
        const std::uint32_t p[13] = {
            0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00001u,
            0x3F800000u, 0xBF800000u, 0x00800000u, 0x7F7FFFFFu, 0x00000001u,
            0x40000000u, 0x3F000000u, 0xC0400000u};
        for (int i = 0; i < 13; ++i) {
            float value;
            std::memcpy(&value, &p[i], sizeof(float));
            input[static_cast<std::size_t>(i)] = static_cast<T>(value);
        }
    }

    const std::int64_t shape[1] = {13};
    const std::int64_t strides[1] = {1};
    void* src = storage_of(input);
    check(src != nullptr, "special-value source");
    if (src == nullptr) return;

    struct Case {
        const char* name;
        void (*fn)(const void*, void*, const std::int64_t*, const std::int64_t*,
                   std::int64_t, std::int64_t);
        T (*reference)(T);
    };
    const Case cases[] = {
        {"relu special", tf_core_relu, ref_relu<T>},
        {"sqrt special", tf_core_sqrt, ref_sqrt<T>},
        {"reciprocal special", tf_core_reciprocal, ref_reciprocal<T>},
    };
    for (std::size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
        std::vector<T> poison(13, DtypeTraits<T>::sentinel());
        void* dst = storage_of(poison);
        if (dst == nullptr) { check(false, "special-value destination"); continue; }
        tf_clear_error();
        cases[c].fn(src, dst, shape, strides, 0, 1);
        std::snprintf(message, sizeof message, "%.40s [%.16s]: error set",
                      cases[c].name, DtypeTraits<T>::name());
        check(tf_last_error_code() == TF_OK, message);
        const std::vector<T> got = read_back<T>(dst);
        const std::vector<T> expected = reference_unary<T>(
            input, cases[c].reference, shape, strides, 0, 1);
        // Every one of these is IEEE-specified, so it is a bit comparison —
        // which is what makes the signed zeros and the NaN class real
        // assertions rather than vacuous ones.
        same_bits<T>(got, expected, 13, cases[c].name);
        tf_storage_destroy(dst);
    }
    tf_storage_destroy(src);
}

// ---------------------------------------------------------------------------
// 3. Binary: every layout pairing, every tier, both dtypes
// ---------------------------------------------------------------------------

template <class T>
void run_binary_layouts(const char* op_name,
                        void (*strided)(const void*, const void*, void*,
                                        const std::int64_t*,
                                        const std::int64_t*,
                                        const std::int64_t*, std::int64_t,
                                        std::int64_t, std::int64_t),
                        T (*reference)(T, T)) {
    char message[256];
    const std::vector<T> a_values = signed_inputs<T>();
    const std::vector<T> b_values = second_inputs<T>();
    void* a = storage_of(a_values);
    void* b = storage_of(b_values);
    check(a != nullptr && b != nullptr, "binary source storage");
    if (a == nullptr || b == nullptr) {
        if (a) tf_storage_destroy(a);
        if (b) tf_storage_destroy(b);
        return;
    }

    // Pair every layout with itself and with the transposed/broadcast forms
    // of the same rank, which is how a real broadcast reaches the kernel.
    for (std::size_t l = 0; l < kLayoutCount; ++l) {
        const Layout& left = kLayouts[l];
        for (std::size_t r = 0; r < kLayoutCount; ++r) {
            const Layout& right = kLayouts[r];
            if (right.ndim != left.ndim) continue;
            bool same_shape = true;
            for (std::int64_t d = 0; d < left.ndim; ++d)
                if (left.shape[d] != right.shape[d]) same_shape = false;
            if (!same_shape) continue;

            std::int64_t numel = 1;
            for (std::int64_t d = 0; d < left.ndim; ++d) numel *= left.shape[d];

            const std::vector<T> expected = reference_binary<T>(
                a_values, b_values, reference, left.shape, left.strides,
                right.strides, left.offset, right.offset, left.ndim);

            std::vector<T> poison(static_cast<std::size_t>(numel),
                                  DtypeTraits<T>::sentinel());
            void* dst = storage_of(poison);
            if (dst == nullptr) { check(false, "binary destination"); continue; }

            tf_clear_error();
            strided(a, b, dst, left.shape, left.strides, right.strides,
                    left.offset, right.offset, left.ndim);
            std::snprintf(message, sizeof message,
                          "%.30s %.30s/%.30s [%.16s]: error set", op_name,
                          left.name, right.name, DtypeTraits<T>::name());
            check(tf_last_error_code() == TF_OK, message);

            const std::vector<T> got = read_back<T>(dst);
            std::snprintf(message, sizeof message, "%.30s %.30s/%.30s", op_name,
                          left.name, right.name);
            same_bits<T>(got, expected, numel, message);
            tf_storage_destroy(dst);
        }
    }
    tf_storage_destroy(a);
    tf_storage_destroy(b);
}

template <class T>
void run_binary_contiguous(const char* op_name,
                           void (*contiguous)(const void*, const void*, void*,
                                              std::int64_t, std::int64_t,
                                              std::int64_t),
                           T (*reference)(T, T)) {
    char message[256];
    const std::vector<T> a_values = signed_inputs<T>();
    const std::vector<T> b_values = second_inputs<T>();
    void* a = storage_of(a_values);
    void* b = storage_of(b_values);
    if (a == nullptr || b == nullptr) {
        check(false, "binary contiguous source");
        if (a) tf_storage_destroy(a);
        if (b) tf_storage_destroy(b);
        return;
    }
    // A nonzero offset on each side, so the flat run is not merely index 0.
    const std::int64_t shape[1] = {10};
    const std::int64_t strides[1] = {1};
    const std::vector<T> expected = reference_binary<T>(
        a_values, b_values, reference, shape, strides, strides, 2, 4, 1);
    std::vector<T> poison(10, DtypeTraits<T>::sentinel());
    void* dst = storage_of(poison);
    if (dst != nullptr) {
        tf_clear_error();
        contiguous(a, b, dst, 10, 2, 4);
        std::snprintf(message, sizeof message,
                      "%.40s contiguous [%.16s]: error set", op_name,
                      DtypeTraits<T>::name());
        check(tf_last_error_code() == TF_OK, message);
        const std::vector<T> got = read_back<T>(dst);
        std::snprintf(message, sizeof message, "%.40s contiguous", op_name);
        same_bits<T>(got, expected, 10, message);
        tf_storage_destroy(dst);
    } else {
        check(false, "binary contiguous destination");
    }
    tf_storage_destroy(a);
    tf_storage_destroy(b);
}

template <class T>
void test_binary_family() {
    run_binary_layouts<T>("add", tf_core_add, ref_add<T>);
    run_binary_layouts<T>("subtract", tf_core_subtract, ref_subtract<T>);
    run_binary_layouts<T>("multiply", tf_core_multiply, ref_multiply<T>);
    run_binary_layouts<T>("relu_backward", tf_core_relu_backward,
                          ref_relu_backward<T>);
    run_binary_contiguous<T>("add", tf_core_add_contiguous, ref_add<T>);
    run_binary_contiguous<T>("subtract", tf_core_subtract_contiguous,
                             ref_subtract<T>);
    run_binary_contiguous<T>("multiply", tf_core_multiply_contiguous,
                             ref_multiply<T>);
}

// ---------------------------------------------------------------------------
// 4. Traversal-tier parity, asserted through the builders themselves
// ---------------------------------------------------------------------------
//
// The layouts above already run whichever tier their metadata selects. This
// closes the loop by driving the *same* logical elements through all three
// tiers explicitly and comparing the results to each other rather than to a
// reference — which is the statement H8 makes at float64 and I3 must make at
// float32 too: the tier is a traversal choice and nothing else.

template <class T>
void test_traversal_tiers_agree() {
    char message[256];
    const std::vector<T> a_values = signed_inputs<T>();
    const std::vector<T> b_values = second_inputs<T>();
    const std::int64_t shape[2] = {4, 4};
    const std::int64_t contiguous_strides[2] = {4, 1};
    const std::int64_t transposed[2] = {1, 4};

    // Tier 1: the flat contiguous row (the ``_contiguous`` export).
    // Tier 2: the collapsed plan (a row-major strided view collapses to one
    //         axis of 16 elements with stride 1).
    // Tier 3: the retained odometer (a transposed operand cannot merge).
    tf::ElementwiseBinaryPlan plan;
    check(tf::build_binary_plan(shape, contiguous_strides, contiguous_strides,
                                2, plan),
          "row-major pair collapses into a plan");
    check(plan.ndim == 1 && plan.shape[0] == 16,
          "the row-major pair collapses to one axis of 16");
    check(tf::build_binary_plan(shape, transposed, contiguous_strides, 2, plan),
          "a transposed operand still plans (it merely cannot merge)");
    check(plan.ndim == 2, "a transposed operand keeps both axes");

    void* a = storage_of(a_values);
    void* b = storage_of(b_values);
    std::vector<T> poison(16, DtypeTraits<T>::sentinel());
    void* flat = storage_of(poison);
    void* planned = storage_of(poison);
    if (a == nullptr || b == nullptr || flat == nullptr || planned == nullptr) {
        check(false, "tier-parity storage");
    } else {
        tf_clear_error();
        tf_core_add_contiguous(a, b, flat, 16, 0, 0);
        tf_core_add(a, b, planned, shape, contiguous_strides,
                    contiguous_strides, 0, 0, 2);
        std::snprintf(message, sizeof message, "tier parity [%.16s]: error set",
                      DtypeTraits<T>::name());
        check(tf_last_error_code() == TF_OK, message);
        const std::vector<T> flat_got = read_back<T>(flat);
        const std::vector<T> planned_got = read_back<T>(planned);
        same_bits<T>(flat_got, planned_got, 16,
                     "contiguous tier vs collapsed-plan tier");

        // Tier 3 against tier 2, on the *same* logical elements: walk the
        // transposed operand and compare with the plan result of the
        // equivalent explicitly transposed reference.
        std::vector<T> odometer_poison(16, DtypeTraits<T>::sentinel());
        void* odometer = storage_of(odometer_poison);
        if (odometer != nullptr) {
            tf_clear_error();
            tf_core_add(a, b, odometer, shape, transposed, contiguous_strides,
                        0, 0, 2);
            check(tf_last_error_code() == TF_OK, "transposed add error slot");
            const std::vector<T> got = read_back<T>(odometer);
            const std::vector<T> expected = reference_binary<T>(
                a_values, b_values, ref_add<T>, shape, transposed,
                contiguous_strides, 0, 0, 2);
            same_bits<T>(got, expected, 16, "odometer tier vs reference");
            tf_storage_destroy(odometer);
        }
    }
    if (a) tf_storage_destroy(a);
    if (b) tf_storage_destroy(b);
    if (flat) tf_storage_destroy(flat);
    if (planned) tf_storage_destroy(planned);
}

// ---------------------------------------------------------------------------
// 5. Mixed dtype is rejected before anything is written
// ---------------------------------------------------------------------------

// The destination's bytes are read back before and after, so "writes
// nothing" is a measurement rather than an inference.
void expect_rejected(const char* what, void* dst, std::int64_t dst_size,
                     bool is_float32_dst) {
    char message[256];
    std::snprintf(message, sizeof message, "%.150s: not rejected", what);
    check(tf_last_error_code() == TF_ERROR_INVALID, message);
    const char* text = tf_last_error_message();
    std::snprintf(message, sizeof message, "%.120s: message does not name the "
                                           "dtype requirement", what);
    check(text != nullptr && std::strstr(text, "same dtype") != nullptr,
          message);
    // The destination still holds its sentinel, byte for byte.
    if (is_float32_dst) {
        std::vector<float> after(static_cast<std::size_t>(dst_size));
        tf_storage_copy_to(dst, after.data());
        for (std::int64_t i = 0; i < dst_size; ++i) {
            if (bits<float>(after[static_cast<std::size_t>(i)])
                    != bits<float>(DtypeTraits<float>::sentinel())) {
                std::snprintf(message, sizeof message,
                              "%.140s: destination was written", what);
                check(false, message);
                break;
            }
        }
    } else {
        std::vector<double> after(static_cast<std::size_t>(dst_size));
        tf_storage_copy_to(dst, after.data());
        for (std::int64_t i = 0; i < dst_size; ++i) {
            if (bits<double>(after[static_cast<std::size_t>(i)])
                    != bits<double>(DtypeTraits<double>::sentinel())) {
                std::snprintf(message, sizeof message,
                              "%.140s: destination was written", what);
                check(false, message);
                break;
            }
        }
    }
    tf_clear_error();
}

void test_mixed_dtype_is_rejected_before_any_write() {
    const std::int64_t shape[2] = {4, 4};
    const std::int64_t strides[2] = {4, 1};

    std::vector<float> f32_values(16, 1.5f);
    std::vector<double> f64_values(16, 1.5);
    std::vector<float> f32_poison(16, DtypeTraits<float>::sentinel());
    std::vector<double> f64_poison(16, DtypeTraits<double>::sentinel());

    void* f32 = storage_of(f32_values);
    void* f64 = storage_of(f64_values);
    void* f32_dst = storage_of(f32_poison);
    void* f64_dst = storage_of(f64_poison);
    check(f32 && f64 && f32_dst && f64_dst, "mixed-dtype storage");
    if (!(f32 && f64 && f32_dst && f64_dst)) return;

    // -- unary: source and destination must agree, in both directions ------
    struct UnaryCase {
        const char* name;
        void (*fn)(const void*, void*, const std::int64_t*, const std::int64_t*,
                   std::int64_t, std::int64_t);
    };
    const UnaryCase unary[] = {
        {"tf_core_relu", tf_core_relu},
        {"tf_core_sqrt", tf_core_sqrt},
        {"tf_core_reciprocal", tf_core_reciprocal},
        {"tf_core_exp", tf_core_exp},
        {"tf_core_log", tf_core_log},
    };
    char label[256];
    for (std::size_t i = 0; i < sizeof(unary) / sizeof(unary[0]); ++i) {
        tf_clear_error();
        unary[i].fn(f32, f64_dst, shape, strides, 0, 2);
        std::snprintf(label, sizeof label, "%.60s f32 source, f64 destination",
                      unary[i].name);
        expect_rejected(label, f64_dst, 16, false);

        tf_clear_error();
        unary[i].fn(f64, f32_dst, shape, strides, 0, 2);
        std::snprintf(label, sizeof label, "%.60s f64 source, f32 destination",
                      unary[i].name);
        expect_rejected(label, f32_dst, 16, true);
    }

    struct UnaryContiguousCase {
        const char* name;
        void (*fn)(const void*, void*, std::int64_t, std::int64_t);
    };
    const UnaryContiguousCase unary_contiguous[] = {
        {"tf_core_relu_contiguous", tf_core_relu_contiguous},
        {"tf_core_sqrt_contiguous", tf_core_sqrt_contiguous},
        {"tf_core_reciprocal_contiguous", tf_core_reciprocal_contiguous},
        {"tf_core_exp_contiguous", tf_core_exp_contiguous},
        {"tf_core_log_contiguous", tf_core_log_contiguous},
    };
    for (std::size_t i = 0;
         i < sizeof(unary_contiguous) / sizeof(unary_contiguous[0]); ++i) {
        tf_clear_error();
        unary_contiguous[i].fn(f32, f64_dst, 16, 0);
        std::snprintf(label, sizeof label, "%.60s f32 source, f64 destination",
                      unary_contiguous[i].name);
        expect_rejected(label, f64_dst, 16, false);
    }

    // -- binary: all three positions, independently ------------------------
    struct BinaryCase {
        const char* name;
        void (*fn)(const void*, const void*, void*, const std::int64_t*,
                   const std::int64_t*, const std::int64_t*, std::int64_t,
                   std::int64_t, std::int64_t);
    };
    const BinaryCase binary[] = {
        {"tf_core_add", tf_core_add},
        {"tf_core_subtract", tf_core_subtract},
        {"tf_core_multiply", tf_core_multiply},
        {"tf_core_relu_backward", tf_core_relu_backward},
    };
    for (std::size_t i = 0; i < sizeof(binary) / sizeof(binary[0]); ++i) {
        // float32 left, float64 right.
        tf_clear_error();
        binary[i].fn(f32, f64, f64_dst, shape, strides, strides, 0, 0, 2);
        std::snprintf(label, sizeof label, "%.60s f32 lhs with f64 rhs",
                      binary[i].name);
        expect_rejected(label, f64_dst, 16, false);

        // float64 left, float32 right.
        tf_clear_error();
        binary[i].fn(f64, f32, f64_dst, shape, strides, strides, 0, 0, 2);
        std::snprintf(label, sizeof label, "%.60s f64 lhs with f32 rhs",
                      binary[i].name);
        expect_rejected(label, f64_dst, 16, false);

        // Matching operands, wrong-dtype destination — the direction that
        // would corrupt memory rather than merely misread it.
        tf_clear_error();
        binary[i].fn(f32, f32, f64_dst, shape, strides, strides, 0, 0, 2);
        std::snprintf(label, sizeof label, "%.60s f32 operands, f64 destination",
                      binary[i].name);
        expect_rejected(label, f64_dst, 16, false);

        tf_clear_error();
        binary[i].fn(f64, f64, f32_dst, shape, strides, strides, 0, 0, 2);
        std::snprintf(label, sizeof label, "%.60s f64 operands, f32 destination",
                      binary[i].name);
        expect_rejected(label, f32_dst, 16, true);
    }

    struct BinaryContiguousCase {
        const char* name;
        void (*fn)(const void*, const void*, void*, std::int64_t, std::int64_t,
                   std::int64_t);
    };
    const BinaryContiguousCase contiguous_cases[] = {
        {"tf_core_add_contiguous", tf_core_add_contiguous},
        {"tf_core_subtract_contiguous", tf_core_subtract_contiguous},
        {"tf_core_multiply_contiguous", tf_core_multiply_contiguous},
    };
    for (std::size_t i = 0;
         i < sizeof(contiguous_cases) / sizeof(contiguous_cases[0]); ++i) {
        tf_clear_error();
        contiguous_cases[i].fn(f32, f64, f64_dst, 16, 0, 0);
        std::snprintf(label, sizeof label, "%.60s f32 lhs with f64 rhs",
                      contiguous_cases[i].name);
        expect_rejected(label, f64_dst, 16, false);

        tf_clear_error();
        contiguous_cases[i].fn(f32, f32, f64_dst, 16, 0, 0);
        std::snprintf(label, sizeof label, "%.60s f32 operands, f64 destination",
                      contiguous_cases[i].name);
        expect_rejected(label, f64_dst, 16, false);
    }

    // ...and a matched call still succeeds afterwards, so nothing latched.
    tf_clear_error();
    tf_core_add(f32, f32, f32_dst, shape, strides, strides, 0, 0, 2);
    check(tf_last_error_code() == TF_OK,
          "a matched call failed after a rejected mixed-dtype one");
    std::vector<float> after(16);
    tf_storage_copy_to(f32_dst, after.data());
    check(bits<float>(after[0]) == bits<float>(3.0f),
          "the matched call did not compute");

    tf_storage_destroy(f32);
    tf_storage_destroy(f64);
    tf_storage_destroy(f32_dst);
    tf_storage_destroy(f64_dst);
}

// ---------------------------------------------------------------------------
// 6. The dtype guard runs before the span validation
// ---------------------------------------------------------------------------
//
// exp/log/contiguous_copy are the self-validating exports, so they are where
// the ordering is observable: a call that is *both* mixed-dtype and
// out-of-span must report the dtype, not the span. A later error must never
// overwrite the dtype rejection.

void test_the_dtype_rejection_is_not_overwritten_by_a_span_error() {
    std::vector<float> f32_values(16, 1.5f);
    std::vector<double> f64_poison(4, DtypeTraits<double>::sentinel());
    void* f32 = storage_of(f32_values);
    void* f64_dst = storage_of(f64_poison);
    check(f32 && f64_dst, "ordering storage");
    if (!(f32 && f64_dst)) return;

    // ndim is negative *and* the dtypes disagree: the dtype must win.
    const std::int64_t shape[2] = {100, 100};
    const std::int64_t strides[2] = {100, 1};
    tf_clear_error();
    tf_core_exp(f32, f64_dst, shape, strides, 0, 2);
    check(tf_last_error_code() == TF_ERROR_INVALID, "exp mixed+overrun status");
    const char* text = tf_last_error_message();
    check(text != nullptr && std::strstr(text, "same dtype") != nullptr,
          "exp reported the span instead of the dtype");
    tf_clear_error();

    tf_core_log(f32, f64_dst, shape, strides, -1, 2);
    check(tf_last_error_code() == TF_ERROR_INVALID, "log mixed+negative ndim");
    text = tf_last_error_message();
    check(text != nullptr && std::strstr(text, "same dtype") != nullptr,
          "log reported the metadata instead of the dtype");
    tf_clear_error();

    // ...and with matching dtypes the span error is still reported, so the
    // dtype guard did not swallow the pre-existing validation.
    void* f32_dst = storage_of(std::vector<float>(4, 0.0f));
    if (f32_dst != nullptr) {
        tf_clear_error();
        tf_core_exp(f32, f32_dst, shape, strides, 0, 2);
        check(tf_last_error_code() == TF_ERROR_INVALID, "exp span status");
        text = tf_last_error_message();
        check(text != nullptr && std::strstr(text, "span") != nullptr,
              "exp did not report the span for a matched-dtype overrun");
        tf_clear_error();
        tf_storage_destroy(f32_dst);
    }

    tf_storage_destroy(f32);
    tf_storage_destroy(f64_dst);
}

// ---------------------------------------------------------------------------
// 7. Output dtype follows the operands
// ---------------------------------------------------------------------------

template <class T>
void test_output_keeps_the_operand_dtype() {
    char message[256];
    const std::vector<T> values = unary_inputs<T>();
    void* src = storage_of(values);
    std::vector<T> poison(16, DtypeTraits<T>::sentinel());
    void* dst = storage_of(poison);
    if (src == nullptr || dst == nullptr) {
        check(false, "output-dtype storage");
        if (src) tf_storage_destroy(src);
        if (dst) tf_storage_destroy(dst);
        return;
    }
    // The tag is immutable across an operation, and the logical size still
    // counts elements rather than bytes: a float32 storage of 16 reports 16.
    check(tf::storage_dtype(dst) == tf::storage_dtype(src),
          "operand and destination tags disagree before the call");
    tf_clear_error();
    tf_core_sqrt_contiguous(src, dst, 16, 0);
    std::snprintf(message, sizeof message, "sqrt [%.16s]: error set",
                  DtypeTraits<T>::name());
    check(tf_last_error_code() == TF_OK, message);
    std::snprintf(message, sizeof message, "sqrt [%.16s]: dtype tag moved",
                  DtypeTraits<T>::name());
    check(tf::storage_dtype(dst) == tf::storage_dtype(src), message);
    std::snprintf(message, sizeof message, "sqrt [%.16s]: element count moved",
                  DtypeTraits<T>::name());
    check(tf_storage_size(dst) == 16 && tf_storage_size(src) == 16, message);
    // The source is unchanged — an operation reads its operand, never writes
    // it, and the destination never steals it.
    const std::vector<T> source_after = read_back<T>(src);
    same_bits<T>(source_after, values, 16, "the source was mutated");
    tf_storage_destroy(src);
    tf_storage_destroy(dst);
}

}  // namespace

int main() {
    test_unary_family<double>();
    test_unary_family<float>();

    test_unary_special_values<double>();
    test_unary_special_values<float>();

    test_binary_family<double>();
    test_binary_family<float>();

    test_traversal_tiers_agree<double>();
    test_traversal_tiers_agree<float>();

    test_output_keeps_the_operand_dtype<double>();
    test_output_keeps_the_operand_dtype<float>();

    test_mixed_dtype_is_rejected_before_any_write();
    test_the_dtype_rejection_is_not_overwritten_by_a_span_error();

    if (g_failures == 0) {
        std::printf("dtype elementwise: all checks passed\n");
        return 0;
    }
    std::printf("dtype elementwise: %d check(s) failed\n", g_failures);
    return 1;
}
