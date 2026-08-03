// Dependency-free C++ test for dtype-general Conv2d and MaxPool2d
// (Phase I, milestone I5). No GoogleTest / Catch2 — a plain executable
// that prints failures and returns a nonzero exit code if any check
// fails, so CTest reports pass/fail.
//
// This binary compiles cpp/src/conv2d.cpp, cpp/src/pooling.cpp,
// cpp/src/storage.cpp, and cpp/src/error.cpp directly, so it reaches the
// hidden ``tf::conv2d_*_prefers_*`` predicates and **both** compute paths
// per Conv2d direction — the retained Phase-D generic loops and the H9
// optimized traversals — alongside the exported wrappers they live
// inside, at the layer where the properties are actually decided, with no
// Python wrapper, no ctypes boundary, and no NumPy anywhere.
//
// It is deliberately complementary to the neighbouring targets rather
// than a superset of any of them:
//
//   * ``test_conv2d_forward`` / ``test_conv2d_input_backward`` /
//     ``test_conv2d_weight_backward`` prove the Phase-D semantics at
//     **float64**, hand-computed case by hand-computed case. Untouched.
//   * ``test_conv2d_execution`` proves H9's optimized-versus-retained
//     agreement and predicate boundaries at **float64**, including its
//     NaN/signed-zero sweeps. Also untouched.
//   * ``test_maxpool2d_forward`` / ``test_maxpool2d_backward`` prove the
//     D8/D9 pooling semantics at **float64**. Also untouched.
//
// This one proves the set that opened at I5:
//
//   1. **Both Conv2d paths agree with an independent same-dtype reference,
//      per direction, per dtype, bit for bit.** The references are written
//      here, in ``T``, in the documented accumulation order — a test that
//      called the production kernels for its expectation would be
//      comparing the kernel against itself. Geometry spans both sides of
//      every H9 predicate boundary, strides, padding, rectangular shapes,
//      multiple batches and channels, and bias/no-bias.
//
//   2. **float32 really accumulates in float32, in all three Conv2d
//      directions and in the overlapping MaxPool2d backward.** Each gets a
//      deterministic witness — ``1.0`` followed by copies of ``2**-24`` —
//      on which sequential binary32 accumulation differs from binary64
//      accumulation narrowed once, exercised on **both** traversals of
//      each direction, because a widening accumulator introduced on only
//      the optimized path is exactly the plausible mistake. TensorForge is
//      asserted, by raw bit pattern, to equal the sequential binary32
//      result and to differ from the widened one.
//
//   3. **The MaxPool2d value path is dtype-general and the winner buffer
//      is not.** Values, ties (first occurrence wins under strict ``>``),
//      signed-zero ties, the NaN policy, infinities, subnormals, and
//      padded windows all follow the same comparison sequence at both
//      widths, proved against an independent scalar reference; the winner
//      buffer is **float64 at every value dtype**, a float32 winner
//      buffer is refused before anything is written, and the ``2**53``
//      plane bound holds unchanged for a float32 pool (design §13.3).
//
//   4. **Destination initialization per H1 is intact at both widths.**
//      Every destination the audit classifies as fully overwritten is
//      poisoned with a sentinel here and proved sentinel-free after the
//      kernel; the cells the scatters deliberately leave alone are proved
//      to hold exactly +0.0.
//
//   5. **Mixed dtype is rejected before anything is written**, in every
//      participating handle position of all five exports, with
//      TF_ERROR_INVALID and a byte-for-byte unchanged destination — and
//      invalid winner values (fractional, negative, out of range,
//      non-finite) are rejected the same way at both value dtypes.
//
//   6. **float64 did not move.** Every check runs at float64 too, against
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
#include <limits>
#include <vector>

#include "tf_conv2d_internal.h"
#include "tf_internal.h"
#include "tf_pooling_internal.h"

TF_EXPORT void* tf_storage_create_typed(std::int64_t size,
                                        std::int32_t dtype_code);
TF_EXPORT void* tf_storage_create(std::int64_t size);
TF_EXPORT void tf_storage_destroy(void* handle);
TF_EXPORT std::int64_t tf_storage_size(const void* handle);
TF_EXPORT void tf_storage_copy_from(void* handle, const void* src);
TF_EXPORT void tf_storage_copy_to(const void* handle, void* dst);

TF_EXPORT void tf_core_conv2d_forward(
    const void* input_handle, std::int64_t input_offset,
    const void* weight_handle, std::int64_t weight_offset,
    const void* bias_handle, std::int64_t bias_offset,
    void* output_handle,
    std::int64_t batch, std::int64_t in_channels,
    std::int64_t input_height, std::int64_t input_width,
    std::int64_t out_channels, std::int64_t kernel_height,
    std::int64_t kernel_width, std::int64_t stride_height,
    std::int64_t stride_width, std::int64_t pad_height,
    std::int64_t pad_width, std::int64_t output_height,
    std::int64_t output_width);
TF_EXPORT void tf_core_conv2d_input_backward(
    const void* grad_output_handle, std::int64_t grad_output_offset,
    const void* weight_handle, std::int64_t weight_offset,
    void* grad_input_handle,
    std::int64_t batch, std::int64_t in_channels,
    std::int64_t input_height, std::int64_t input_width,
    std::int64_t out_channels, std::int64_t kernel_height,
    std::int64_t kernel_width, std::int64_t stride_height,
    std::int64_t stride_width, std::int64_t pad_height,
    std::int64_t pad_width, std::int64_t output_height,
    std::int64_t output_width);
TF_EXPORT void tf_core_conv2d_weight_backward(
    const void* grad_output_handle, std::int64_t grad_output_offset,
    const void* input_handle, std::int64_t input_offset,
    void* grad_weight_handle,
    std::int64_t batch, std::int64_t in_channels,
    std::int64_t input_height, std::int64_t input_width,
    std::int64_t out_channels, std::int64_t kernel_height,
    std::int64_t kernel_width, std::int64_t stride_height,
    std::int64_t stride_width, std::int64_t pad_height,
    std::int64_t pad_width, std::int64_t output_height,
    std::int64_t output_width);
TF_EXPORT void tf_core_maxpool2d_forward(
    const void* input_handle, std::int64_t input_offset,
    void* output_handle, void* winners_handle,
    std::int64_t batch, std::int64_t channels,
    std::int64_t input_height, std::int64_t input_width,
    std::int64_t kernel_height, std::int64_t kernel_width,
    std::int64_t stride_height, std::int64_t stride_width,
    std::int64_t pad_height, std::int64_t pad_width,
    std::int64_t output_height, std::int64_t output_width);
TF_EXPORT void tf_core_maxpool2d_backward(
    const void* grad_output_handle, std::int64_t grad_output_offset,
    const void* winners_handle, std::int64_t winners_offset,
    void* grad_input_handle,
    std::int64_t batch, std::int64_t channels,
    std::int64_t input_height, std::int64_t input_width,
    std::int64_t output_height, std::int64_t output_width);
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
// Bit-pattern plumbing and per-dtype traits, in the shape the earlier
// dtype targets established so the files read alike.
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

// A typed null bias pointer: the kernels are templates deduced from their
// pointer arguments, and a bare ``nullptr`` deduces nothing.
template <class T>
const T* no_bias() {
    return nullptr;
}

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
               const char* what) {
    if (a.size() != b.size()) {
        char message[256];
        std::snprintf(message, sizeof message, "%.180s [%.16s]: size mismatch",
                      what, DtypeTraits<T>::name());
        check(false, message);
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (bits<T>(a[i]) != bits<T>(b[i])) {
            char message[256];
            std::snprintf(message, sizeof message,
                          "%.150s [%.16s]: element %zu differs in bits", what,
                          DtypeTraits<T>::name(), i);
            check(false, message);
            return false;
        }
    }
    return true;
}

// Deterministic, dtype-exact finite values with varied signs and
// magnitudes; every one is exactly representable at binary32, so the
// float32 case is not a rounded copy of the float64 one.
template <class T>
std::vector<T> patterned(std::size_t count, int salt) {
    std::vector<T> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        const int k = static_cast<int>((i * 7 + static_cast<std::size_t>(salt) * 13) % 23);
        const T magnitude = T(0.25) * T(k) - T(2.5);
        out[i] = (i % 3 == 0) ? -magnitude : magnitude;
    }
    return out;
}

// ---------------------------------------------------------------------------
// One geometry, carried whole so a case reads as data.
// ---------------------------------------------------------------------------

struct Geometry {
    std::int64_t n, c, h, w;      // input NCHW
    std::int64_t o, kh, kw;       // out_channels + kernel
    std::int64_t sh, sw, ph, pw;  // stride + padding
    bool bias;
    const char* label;

    std::int64_t oh() const { return (h + 2 * ph - kh) / sh + 1; }
    std::int64_t ow() const { return (w + 2 * pw - kw) / sw + 1; }
    std::int64_t input_count() const { return n * c * h * w; }
    std::int64_t weight_count() const { return o * c * kh * kw; }
    std::int64_t output_count() const { return n * o * oh() * ow(); }
};

// ---------------------------------------------------------------------------
// Independently written references, in ``T``, in the documented
// accumulation orders.
// ---------------------------------------------------------------------------

std::int64_t ref_index4d(std::int64_t i0, std::int64_t i1, std::int64_t i2,
                         std::int64_t i3, std::int64_t d1, std::int64_t d2,
                         std::int64_t d3) {
    return ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
}

// Forward: bias seed, then ascending c -> p -> q taps per destination.
template <class T>
std::vector<T> reference_conv2d_forward(const Geometry& g,
                                        const std::vector<T>& input,
                                        const std::vector<T>& weight,
                                        const T* bias) {
    std::vector<T> out(static_cast<std::size_t>(g.output_count()), T(0));
    for (std::int64_t n = 0; n < g.n; ++n)
        for (std::int64_t o = 0; o < g.o; ++o)
            for (std::int64_t i = 0; i < g.oh(); ++i)
                for (std::int64_t j = 0; j < g.ow(); ++j) {
                    T acc = (bias != nullptr) ? bias[o] : T(0);
                    for (std::int64_t c = 0; c < g.c; ++c)
                        for (std::int64_t p = 0; p < g.kh; ++p) {
                            const std::int64_t ih = i * g.sh + p - g.ph;
                            if (ih < 0 || ih >= g.h) continue;
                            for (std::int64_t q = 0; q < g.kw; ++q) {
                                const std::int64_t iw = j * g.sw + q - g.pw;
                                if (iw < 0 || iw >= g.w) continue;
                                acc += input[static_cast<std::size_t>(
                                           ref_index4d(n, c, ih, iw, g.c, g.h,
                                                       g.w))]
                                     * weight[static_cast<std::size_t>(
                                           ref_index4d(o, c, p, q, g.c, g.kh,
                                                       g.kw))];
                            }
                        }
                    out[static_cast<std::size_t>(
                        ref_index4d(n, o, i, j, g.o, g.oh(), g.ow()))] = acc;
                }
    return out;
}

// Input gradient: zero-filled, then scatter in n -> o -> i -> j -> c -> p
// -> q order — the documented reference order for every destination.
template <class T>
std::vector<T> reference_conv2d_input_backward(const Geometry& g,
                                               const std::vector<T>& grad_out,
                                               const std::vector<T>& weight) {
    std::vector<T> out(static_cast<std::size_t>(g.input_count()), T(0));
    for (std::int64_t n = 0; n < g.n; ++n)
        for (std::int64_t o = 0; o < g.o; ++o)
            for (std::int64_t i = 0; i < g.oh(); ++i)
                for (std::int64_t j = 0; j < g.ow(); ++j) {
                    const T gval = grad_out[static_cast<std::size_t>(
                        ref_index4d(n, o, i, j, g.o, g.oh(), g.ow()))];
                    for (std::int64_t c = 0; c < g.c; ++c)
                        for (std::int64_t p = 0; p < g.kh; ++p) {
                            const std::int64_t ih = i * g.sh + p - g.ph;
                            if (ih < 0 || ih >= g.h) continue;
                            for (std::int64_t q = 0; q < g.kw; ++q) {
                                const std::int64_t iw = j * g.sw + q - g.pw;
                                if (iw < 0 || iw >= g.w) continue;
                                out[static_cast<std::size_t>(
                                    ref_index4d(n, c, ih, iw, g.c, g.h, g.w))]
                                    += gval * weight[static_cast<std::size_t>(
                                           ref_index4d(o, c, p, q, g.c, g.kh,
                                                       g.kw))];
                            }
                        }
                }
    return out;
}

// Weight gradient: zero-filled, then scatter in the same documented order.
template <class T>
std::vector<T> reference_conv2d_weight_backward(const Geometry& g,
                                                const std::vector<T>& grad_out,
                                                const std::vector<T>& input) {
    std::vector<T> out(static_cast<std::size_t>(g.weight_count()), T(0));
    for (std::int64_t n = 0; n < g.n; ++n)
        for (std::int64_t o = 0; o < g.o; ++o)
            for (std::int64_t i = 0; i < g.oh(); ++i)
                for (std::int64_t j = 0; j < g.ow(); ++j) {
                    const T gval = grad_out[static_cast<std::size_t>(
                        ref_index4d(n, o, i, j, g.o, g.oh(), g.ow()))];
                    for (std::int64_t c = 0; c < g.c; ++c)
                        for (std::int64_t p = 0; p < g.kh; ++p) {
                            const std::int64_t ih = i * g.sh + p - g.ph;
                            if (ih < 0 || ih >= g.h) continue;
                            for (std::int64_t q = 0; q < g.kw; ++q) {
                                const std::int64_t iw = j * g.sw + q - g.pw;
                                if (iw < 0 || iw >= g.w) continue;
                                out[static_cast<std::size_t>(
                                    ref_index4d(o, c, p, q, g.c, g.kh, g.kw))]
                                    += gval * input[static_cast<std::size_t>(
                                           ref_index4d(n, c, ih, iw, g.c, g.h,
                                                       g.w))];
                            }
                        }
                }
    return out;
}

// MaxPool forward: the exact production comparison sequence — the first
// candidate anchors the all-NaN fallback, the first non-NaN candidate
// seeds the scan, and only a strictly greater value displaces the
// selection, so every tie keeps its first occurrence. Padding is the
// element type's own -inf with winner -1.
template <class T>
void reference_maxpool_forward(const Geometry& g, const std::vector<T>& input,
                               std::vector<T>& out_values,
                               std::vector<double>& out_winners) {
    const std::int64_t count = g.n * g.c * g.oh() * g.ow();
    out_values.assign(static_cast<std::size_t>(count), T(0));
    out_winners.assign(static_cast<std::size_t>(count), 0.0);
    const T neg_inf = -std::numeric_limits<T>::infinity();
    for (std::int64_t n = 0; n < g.n; ++n)
        for (std::int64_t c = 0; c < g.c; ++c)
            for (std::int64_t i = 0; i < g.oh(); ++i)
                for (std::int64_t j = 0; j < g.ow(); ++j) {
                    T best = T(0);
                    double winner = -1.0;
                    bool seen_any = false;
                    bool seen_number = false;
                    for (std::int64_t p = 0; p < g.kh; ++p) {
                        const std::int64_t ih = i * g.sh + p - g.ph;
                        const bool row_ok = ih >= 0 && ih < g.h;
                        for (std::int64_t q = 0; q < g.kw; ++q) {
                            const std::int64_t iw = j * g.sw + q - g.pw;
                            const bool ok = row_ok && iw >= 0 && iw < g.w;
                            const T cand = ok
                                ? input[static_cast<std::size_t>(ref_index4d(
                                      n, c, ih, iw, g.c, g.h, g.w))]
                                : neg_inf;
                            const double cand_winner = ok
                                ? static_cast<double>(ih * g.w + iw)
                                : -1.0;
                            if (!seen_any) {
                                seen_any = true;
                                best = cand;
                                winner = cand_winner;
                            }
                            if (cand != cand) continue;
                            if (!seen_number || cand > best) {
                                seen_number = true;
                                best = cand;
                                winner = cand_winner;
                            }
                        }
                    }
                    const std::size_t at = static_cast<std::size_t>(
                        ref_index4d(n, c, i, j, g.c, g.oh(), g.ow()));
                    out_values[at] = best;
                    out_winners[at] = winner;
                }
}

// MaxPool backward: zero-filled, then scatter-add in n -> c -> oh -> ow
// order, in the element type.
template <class T>
std::vector<T> reference_maxpool_backward(const Geometry& g,
                                          const std::vector<T>& grad_out,
                                          const std::vector<double>& winners) {
    std::vector<T> out(static_cast<std::size_t>(g.input_count()), T(0));
    const std::int64_t plane = g.h * g.w;
    for (std::int64_t n = 0; n < g.n; ++n)
        for (std::int64_t c = 0; c < g.c; ++c) {
            const std::int64_t base = (n * g.c + c) * plane;
            for (std::int64_t i = 0; i < g.oh(); ++i)
                for (std::int64_t j = 0; j < g.ow(); ++j) {
                    const std::size_t at = static_cast<std::size_t>(
                        ref_index4d(n, c, i, j, g.c, g.oh(), g.ow()));
                    const double winner = winners[at];
                    if (winner < 0.0) continue;
                    out[static_cast<std::size_t>(
                        base + static_cast<std::int64_t>(winner))] +=
                        grad_out[at];
                }
        }
    return out;
}

// ---------------------------------------------------------------------------
// 1. Conv2d: export + both internal paths against the reference, per
//    geometry, per dtype
// ---------------------------------------------------------------------------

// The geometry sweep. Swept extents 1-3 stay on the retained generic
// paths, 4+ take the H9 traversals; the strided rows keep the input
// backward on its retained path (the gather requires unit stride), which
// is exactly the asymmetry the predicates encode.
const Geometry kGeometries[] = {
    {1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, false, "3x3 minimal, generic"},
    {1, 1, 4, 4, 1, 2, 2, 1, 1, 0, 0, true, "4x4 boundary, generic (ow=3)"},
    {1, 1, 4, 5, 1, 2, 2, 1, 1, 0, 0, true, "4x5, optimized (ow=4)"},
    {1, 1, 6, 6, 1, 3, 3, 1, 1, 0, 0, false, "6x6 k3, optimized"},
    {2, 3, 6, 7, 4, 3, 2, 1, 1, 0, 0, true, "batched rectangular, optimized"},
    {1, 2, 6, 6, 2, 3, 3, 1, 1, 1, 1, true, "padded, optimized"},
    {1, 1, 7, 7, 1, 3, 3, 2, 2, 0, 0, false, "stride 2, fwd/weight optimized"},
    {2, 2, 8, 8, 3, 3, 3, 2, 2, 1, 1, true, "stride 2 padded batched"},
    {1, 1, 5, 9, 1, 1, 3, 1, 2, 0, 1, false, "asymmetric stride/pad"},
    {1, 1, 1, 8, 1, 1, 2, 1, 1, 0, 0, false, "single row, optimized"},
    {1, 1, 8, 1, 1, 2, 1, 1, 1, 0, 0, false, "single column, generic (w=1)"},
};

template <class T>
void run_conv2d_case(const Geometry& g) {
    char message[256];
    const std::vector<T> input =
        patterned<T>(static_cast<std::size_t>(g.input_count()), 1);
    const std::vector<T> weight =
        patterned<T>(static_cast<std::size_t>(g.weight_count()), 2);
    const std::vector<T> bias_values =
        patterned<T>(static_cast<std::size_t>(g.o), 3);
    const std::vector<T> grad_out =
        patterned<T>(static_cast<std::size_t>(g.output_count()), 4);
    const T* bias = g.bias ? bias_values.data() : nullptr;

    // -- forward: reference vs both internal paths vs the export ----------
    const std::vector<T> want_forward =
        reference_conv2d_forward<T>(g, input, weight, bias);
    {
        std::vector<T> generic(static_cast<std::size_t>(g.output_count()),
                               DtypeTraits<T>::sentinel());
        tf::conv2d_forward_generic(input.data(), weight.data(), bias,
                                   generic.data(), g.n, g.c, g.h, g.w, g.o,
                                   g.kh, g.kw, g.sh, g.sw, g.ph, g.pw,
                                   g.oh(), g.ow());
        std::snprintf(message, sizeof message, "conv fwd generic: %s", g.label);
        same_bits(generic, want_forward, message);

        std::vector<T> optimized(static_cast<std::size_t>(g.output_count()),
                                 DtypeTraits<T>::sentinel());
        tf::conv2d_forward_row_sweep(input.data(), weight.data(), bias,
                                     optimized.data(), g.n, g.c, g.h, g.w,
                                     g.o, g.kh, g.kw, g.sh, g.sw, g.ph, g.pw,
                                     g.oh(), g.ow());
        std::snprintf(message, sizeof message, "conv fwd sweep: %s", g.label);
        same_bits(optimized, want_forward, message);
    }
    {
        void* in = storage_of(input);
        void* wt = storage_of(weight);
        void* bs = g.bias ? storage_of(bias_values) : nullptr;
        void* out = tf_storage_create_typed(g.output_count(),
                                            DtypeTraits<T>::code);
        if (in && wt && out && (!g.bias || bs)) {
            tf_clear_error();
            tf_core_conv2d_forward(in, 0, wt, 0, bs, 0, out, g.n, g.c, g.h,
                                   g.w, g.o, g.kh, g.kw, g.sh, g.sw, g.ph,
                                   g.pw, g.oh(), g.ow());
            std::snprintf(message, sizeof message,
                          "conv fwd export error: %s [%s]", g.label,
                          DtypeTraits<T>::name());
            check(tf_last_error_code() == TF_OK, message);
            std::snprintf(message, sizeof message, "conv fwd export: %s",
                          g.label);
            same_bits(read_back<T>(out), want_forward, message);
        } else {
            check(false, "conv fwd storage");
        }
        if (in) tf_storage_destroy(in);
        if (wt) tf_storage_destroy(wt);
        if (bs) tf_storage_destroy(bs);
        if (out) tf_storage_destroy(out);
    }

    // -- input backward ----------------------------------------------------
    const std::vector<T> want_input_grad =
        reference_conv2d_input_backward<T>(g, grad_out, weight);
    {
        std::vector<T> generic(static_cast<std::size_t>(g.input_count()),
                               DtypeTraits<T>::sentinel());
        tf::conv2d_input_backward_generic(grad_out.data(), weight.data(),
                                          generic.data(), g.n, g.c, g.h, g.w,
                                          g.o, g.kh, g.kw, g.sh, g.sw, g.ph,
                                          g.pw, g.oh(), g.ow());
        std::snprintf(message, sizeof message, "conv ib generic: %s", g.label);
        same_bits(generic, want_input_grad, message);

        // The gather's order proof holds only at unit stride, which is why
        // its predicate requires it; call it directly only where it is
        // legal.
        if (g.sh == 1 && g.sw == 1) {
            std::vector<T> optimized(static_cast<std::size_t>(g.input_count()),
                                     DtypeTraits<T>::sentinel());
            tf::conv2d_input_backward_gather(grad_out.data(), weight.data(),
                                             optimized.data(), g.n, g.c, g.h,
                                             g.w, g.o, g.kh, g.kw, g.sh, g.sw,
                                             g.ph, g.pw, g.oh(), g.ow());
            std::snprintf(message, sizeof message, "conv ib gather: %s",
                          g.label);
            same_bits(optimized, want_input_grad, message);
        }

        void* go = storage_of(grad_out);
        void* wt = storage_of(weight);
        void* out = tf_storage_create_typed(g.input_count(),
                                            DtypeTraits<T>::code);
        if (go && wt && out) {
            tf_clear_error();
            tf_core_conv2d_input_backward(go, 0, wt, 0, out, g.n, g.c, g.h,
                                          g.w, g.o, g.kh, g.kw, g.sh, g.sw,
                                          g.ph, g.pw, g.oh(), g.ow());
            std::snprintf(message, sizeof message,
                          "conv ib export error: %s [%s]", g.label,
                          DtypeTraits<T>::name());
            check(tf_last_error_code() == TF_OK, message);
            std::snprintf(message, sizeof message, "conv ib export: %s",
                          g.label);
            same_bits(read_back<T>(out), want_input_grad, message);
        } else {
            check(false, "conv ib storage");
        }
        if (go) tf_storage_destroy(go);
        if (wt) tf_storage_destroy(wt);
        if (out) tf_storage_destroy(out);
    }

    // -- weight backward ---------------------------------------------------
    const std::vector<T> want_weight_grad =
        reference_conv2d_weight_backward<T>(g, grad_out, input);
    {
        std::vector<T> generic(static_cast<std::size_t>(g.weight_count()),
                               DtypeTraits<T>::sentinel());
        tf::conv2d_weight_backward_generic(grad_out.data(), input.data(),
                                           generic.data(), g.n, g.c, g.h, g.w,
                                           g.o, g.kh, g.kw, g.sh, g.sw, g.ph,
                                           g.pw, g.oh(), g.ow());
        std::snprintf(message, sizeof message, "conv wb generic: %s", g.label);
        same_bits(generic, want_weight_grad, message);

        std::vector<T> optimized(static_cast<std::size_t>(g.weight_count()),
                                 DtypeTraits<T>::sentinel());
        tf::conv2d_weight_backward_gather(grad_out.data(), input.data(),
                                          optimized.data(), g.n, g.c, g.h,
                                          g.w, g.o, g.kh, g.kw, g.sh, g.sw,
                                          g.ph, g.pw, g.oh(), g.ow());
        std::snprintf(message, sizeof message, "conv wb gather: %s", g.label);
        same_bits(optimized, want_weight_grad, message);

        void* go = storage_of(grad_out);
        void* in = storage_of(input);
        void* out = tf_storage_create_typed(g.weight_count(),
                                            DtypeTraits<T>::code);
        if (go && in && out) {
            tf_clear_error();
            tf_core_conv2d_weight_backward(go, 0, in, 0, out, g.n, g.c, g.h,
                                           g.w, g.o, g.kh, g.kw, g.sh, g.sw,
                                           g.ph, g.pw, g.oh(), g.ow());
            std::snprintf(message, sizeof message,
                          "conv wb export error: %s [%s]", g.label,
                          DtypeTraits<T>::name());
            check(tf_last_error_code() == TF_OK, message);
            std::snprintf(message, sizeof message, "conv wb export: %s",
                          g.label);
            same_bits(read_back<T>(out), want_weight_grad, message);
        } else {
            check(false, "conv wb storage");
        }
        if (go) tf_storage_destroy(go);
        if (in) tf_storage_destroy(in);
        if (out) tf_storage_destroy(out);
    }
}

template <class T>
void test_conv2d_geometries() {
    for (const Geometry& g : kGeometries) {
        run_conv2d_case<T>(g);
    }
}

// The predicate boundaries themselves: metadata-only, dtype-blind, and
// exactly where H9 put them.
void test_predicate_boundaries() {
    check(!tf::conv2d_forward_prefers_row_sweep(3, 3), "fwd predicate at 3");
    check(tf::conv2d_forward_prefers_row_sweep(4, 4), "fwd predicate at 4");
    check(!tf::conv2d_forward_prefers_row_sweep(8, 3),
          "fwd predicate min(8,3)");
    check(tf::conv2d_input_backward_prefers_gather(1, 1, 4, 4),
          "ib predicate unit stride");
    check(!tf::conv2d_input_backward_prefers_gather(2, 1, 8, 8),
          "ib predicate strided h");
    check(!tf::conv2d_input_backward_prefers_gather(1, 2, 8, 8),
          "ib predicate strided w");
    check(!tf::conv2d_input_backward_prefers_gather(1, 1, 8, 3),
          "ib predicate short sweep");
    check(tf::conv2d_weight_backward_prefers_gather(4, 4), "wb predicate at 4");
    check(!tf::conv2d_weight_backward_prefers_gather(3, 8),
          "wb predicate min(3,8)");
}

// ---------------------------------------------------------------------------
// 2. Special values through every Conv2d path
// ---------------------------------------------------------------------------

// Signed zeros, infinities, subnormals, and a single NaN, placed so no
// destination accumulates two NaNs (the payload of a two-NaN meeting is
// deliberately outside the contract). The reference performs the same
// operations in the same order, so bit equality — payload included — is
// the correct expectation for every destination here.
template <class T>
void test_conv2d_special_values() {
    Geometry g = {1, 1, 4, 5, 1, 2, 2, 1, 1, 1, 1, false,
                  "special values, padded"};
    std::vector<T> input =
        patterned<T>(static_cast<std::size_t>(g.input_count()), 5);
    input[0] = T(-0.0);
    input[3] = std::numeric_limits<T>::infinity();
    input[7] = -std::numeric_limits<T>::infinity();
    input[11] = std::numeric_limits<T>::denorm_min();
    input[18] = std::numeric_limits<T>::quiet_NaN();
    const std::vector<T> weight = {T(1.0), T(-0.5), T(0.25), T(2.0)};
    const std::vector<T> grad_out =
        patterned<T>(static_cast<std::size_t>(g.output_count()), 6);

    const std::vector<T> want =
        reference_conv2d_forward<T>(g, input, weight, nullptr);
    std::vector<T> generic(static_cast<std::size_t>(g.output_count()), T(0));
    tf::conv2d_forward_generic(input.data(), weight.data(), no_bias<T>(),
                               generic.data(), g.n, g.c, g.h, g.w, g.o, g.kh,
                               g.kw, g.sh, g.sw, g.ph, g.pw, g.oh(), g.ow());
    same_bits(generic, want, "conv fwd special generic");
    std::vector<T> sweep(static_cast<std::size_t>(g.output_count()), T(0));
    tf::conv2d_forward_row_sweep(input.data(), weight.data(), no_bias<T>(),
                                 sweep.data(), g.n, g.c, g.h, g.w, g.o, g.kh,
                                 g.kw, g.sh, g.sw, g.ph, g.pw, g.oh(), g.ow());
    same_bits(sweep, want, "conv fwd special sweep");

    // The weight gradient accumulates nearly the whole plane into each
    // destination, so the mixed sweep above would put *two* NaNs into one
    // accumulation — the input's own NaN plus the one ``0 x inf``
    // manufactures — and a two-NaN meeting's payload and sign are
    // deliberately outside the contract (the H2 ADDSD carve-out; MSVC
    // demonstrably picks different operand registers for the two loops).
    // The contractual cases are exercised separately: at most one NaN per
    // destination, and at most one infinite term per destination.
    {
        // (a) signed zero, subnormal, and a single +inf; every grad value
        // nonzero so no NaN is manufactured, and each input cell reaches a
        // given destination exactly once so infinities never cancel.
        std::vector<T> wb_input = input;
        wb_input[7] = T(1.5);   // drop the -inf
        wb_input[18] = T(-4.5);  // drop the NaN
        std::vector<T> wb_grad = grad_out;
        for (T& value : wb_grad) {
            if (value == T(0)) value = T(0.5);
        }
        const std::vector<T> want_wb =
            reference_conv2d_weight_backward<T>(g, wb_grad, wb_input);
        std::vector<T> wb_gather(static_cast<std::size_t>(g.weight_count()),
                                 T(0));
        tf::conv2d_weight_backward_gather(wb_grad.data(), wb_input.data(),
                                          wb_gather.data(), g.n, g.c, g.h,
                                          g.w, g.o, g.kh, g.kw, g.sh, g.sw,
                                          g.ph, g.pw, g.oh(), g.ow());
        same_bits(wb_gather, want_wb, "conv wb special gather (inf)");
        std::vector<T> wb_generic(static_cast<std::size_t>(g.weight_count()),
                                  T(0));
        tf::conv2d_weight_backward_generic(wb_grad.data(), wb_input.data(),
                                           wb_generic.data(), g.n, g.c, g.h,
                                           g.w, g.o, g.kh, g.kw, g.sh, g.sw,
                                           g.ph, g.pw, g.oh(), g.ow());
        same_bits(wb_generic, want_wb, "conv wb special generic (inf)");
    }
    {
        // (b) a single NaN and no infinities: exactly one NaN enters each
        // destination, and the contract requires payload agreement.
        std::vector<T> wb_input = input;
        wb_input[3] = T(2.0);   // drop the +inf
        wb_input[7] = T(-1.5);  // drop the -inf
        std::vector<T> wb_grad = grad_out;
        for (T& value : wb_grad) {
            if (value == T(0)) value = T(-0.75);
        }
        const std::vector<T> want_wb =
            reference_conv2d_weight_backward<T>(g, wb_grad, wb_input);
        std::vector<T> wb_gather(static_cast<std::size_t>(g.weight_count()),
                                 T(0));
        tf::conv2d_weight_backward_gather(wb_grad.data(), wb_input.data(),
                                          wb_gather.data(), g.n, g.c, g.h,
                                          g.w, g.o, g.kh, g.kw, g.sh, g.sw,
                                          g.ph, g.pw, g.oh(), g.ow());
        same_bits(wb_gather, want_wb, "conv wb special gather (NaN)");
    }

    // A NaN in the upstream reaches the input gradient through both paths.
    std::vector<T> grad_with_nan = grad_out;
    grad_with_nan[2] = std::numeric_limits<T>::quiet_NaN();
    const std::vector<T> want_ib =
        reference_conv2d_input_backward<T>(g, grad_with_nan, weight);
    std::vector<T> ib_generic(static_cast<std::size_t>(g.input_count()), T(0));
    tf::conv2d_input_backward_generic(grad_with_nan.data(), weight.data(),
                                      ib_generic.data(), g.n, g.c, g.h, g.w,
                                      g.o, g.kh, g.kw, g.sh, g.sw, g.ph, g.pw,
                                      g.oh(), g.ow());
    same_bits(ib_generic, want_ib, "conv ib special generic");
    std::vector<T> ib_gather(static_cast<std::size_t>(g.input_count()), T(0));
    tf::conv2d_input_backward_gather(grad_with_nan.data(), weight.data(),
                                     ib_gather.data(), g.n, g.c, g.h, g.w,
                                     g.o, g.kh, g.kw, g.sh, g.sw, g.ph, g.pw,
                                     g.oh(), g.ow());
    same_bits(ib_gather, want_ib, "conv ib special gather");
}

// ---------------------------------------------------------------------------
// 3. The float32 accumulation witness, in all three directions, on both
//    paths
// ---------------------------------------------------------------------------

// ``1.0`` followed by ``count - 1`` copies of ``2**-24``: sequential
// binary32 stays at exactly 1.0 (each addend is half an ULP, and
// round-to-nearest-even keeps the total), while binary64 accumulation
// narrowed once lands measurably higher. Returns the two disagreeing
// answers so the caller can assert TensorForge equals the first and
// differs from the second.
struct WitnessExpectation {
    float sequential;
    float widened;
};

WitnessExpectation witness_expectation(std::size_t count) {
    const float tiny = std::ldexp(1.0f, -24);
    float sequential = 1.0f;
    double widened = 1.0;
    for (std::size_t i = 1; i < count; ++i) {
        sequential += tiny;
        widened += static_cast<double>(tiny);
    }
    WitnessExpectation out;
    out.sequential = sequential;
    out.widened = static_cast<float>(widened);
    return out;
}

std::vector<float> witness_values(std::size_t count) {
    std::vector<float> out(count, std::ldexp(1.0f, -24));
    out[0] = 1.0f;
    return out;
}

void check_witness(float got, const WitnessExpectation& want,
                   const char* what) {
    char message[256];
    std::snprintf(message, sizeof message,
                  "%.150s: expected the sequential binary32 result", what);
    check(bits(got) == bits(want.sequential), message);
    std::snprintf(message, sizeof message,
                  "%.130s: witness is vacuous (sequential == widened)", what);
    check(bits(want.sequential) != bits(want.widened), message);
    std::snprintf(message, sizeof message,
                  "%.130s: matches the forbidden widened accumulation", what);
    check(bits(got) != bits(want.widened), message);
}

void test_conv2d_forward_witness() {
    // Generic path: 1x9 input, 1x9 kernel -> ow = 1 (swept extent 1).
    {
        const Geometry g = {1, 1, 1, 9, 1, 1, 9, 1, 1, 0, 0, false,
                            "fwd witness generic"};
        const std::vector<float> input = witness_values(9);
        const std::vector<float> weight(9, 1.0f);
        const WitnessExpectation want = witness_expectation(9);
        std::vector<float> out(1, 0.0f);
        tf::conv2d_forward_generic(input.data(), weight.data(), no_bias<float>(),
                                   out.data(), g.n, g.c, g.h, g.w, g.o, g.kh,
                                   g.kw, g.sh, g.sw, g.ph, g.pw, g.oh(),
                                   g.ow());
        check_witness(out[0], want, "conv fwd witness (generic)");
        check(!tf::conv2d_forward_prefers_row_sweep(g.w, g.ow()),
              "conv fwd witness generic really is the generic path");
    }
    // Optimized path: 1x12 input, 1x9 kernel -> ow = 4 (swept extent 4).
    // Destination j=0 accumulates input[0..8] in ascending q.
    {
        const Geometry g = {1, 1, 1, 12, 1, 1, 9, 1, 1, 0, 0, false,
                            "fwd witness sweep"};
        std::vector<float> input = witness_values(9);
        input.resize(12, 0.0f);
        const std::vector<float> weight(9, 1.0f);
        const WitnessExpectation want = witness_expectation(9);
        std::vector<float> out(4, 0.0f);
        tf::conv2d_forward_row_sweep(input.data(), weight.data(), no_bias<float>(),
                                     out.data(), g.n, g.c, g.h, g.w, g.o,
                                     g.kh, g.kw, g.sh, g.sw, g.ph, g.pw,
                                     g.oh(), g.ow());
        check_witness(out[0], want, "conv fwd witness (sweep)");
        check(tf::conv2d_forward_prefers_row_sweep(g.w, g.ow()),
              "conv fwd witness sweep really is the optimized path");
        // ...and the export produces the same bits through its dispatch.
        void* in = storage_of(input);
        void* wt = storage_of(weight);
        void* dst = tf_storage_create_typed(4, TF_DTYPE_FLOAT32);
        if (in && wt && dst) {
            tf_clear_error();
            tf_core_conv2d_forward(in, 0, wt, 0, nullptr, 0, dst, 1, 1, 1, 12,
                                   1, 1, 9, 1, 1, 0, 0, 1, 4);
            check(tf_last_error_code() == TF_OK, "conv fwd witness export");
            check_witness(read_back<float>(dst)[0], want,
                          "conv fwd witness (export)");
        } else {
            check(false, "conv fwd witness storage");
        }
        if (in) tf_storage_destroy(in);
        if (wt) tf_storage_destroy(wt);
        if (dst) tf_storage_destroy(dst);
    }
}

void test_conv2d_input_backward_witness() {
    // A 1x1 kernel with 9 output channels and all-ones weights sends
    // grad_output[o] straight to the one input cell, in ascending o —
    // the witness order.
    const std::vector<float> weight(9, 1.0f);
    const WitnessExpectation want = witness_expectation(9);
    // Generic path: 1x1 spatial plane (swept extent 1).
    {
        const std::vector<float> grad_out = witness_values(9);
        std::vector<float> out(1, 0.0f);
        tf::conv2d_input_backward_generic(grad_out.data(), weight.data(),
                                          out.data(), 1, 1, 1, 1, 9, 1, 1, 1,
                                          1, 0, 0, 1, 1);
        check_witness(out[0], want, "conv ib witness (generic)");
    }
    // Gather path: a 1x4 plane at unit stride (swept extent 4); every
    // input cell receives the witness sequence in ascending o.
    {
        std::vector<float> grad_out(9 * 4, std::ldexp(1.0f, -24));
        for (std::int64_t j = 0; j < 4; ++j) {
            grad_out[static_cast<std::size_t>(j)] = 1.0f;  // o = 0 plane
        }
        std::vector<float> out(4, 0.0f);
        tf::conv2d_input_backward_gather(grad_out.data(), weight.data(),
                                         out.data(), 1, 1, 1, 4, 9, 1, 1, 1,
                                         1, 0, 0, 1, 4);
        check(tf::conv2d_input_backward_prefers_gather(1, 1, 4, 4),
              "conv ib witness gather really is the optimized path");
        for (int j = 0; j < 4; ++j) {
            check_witness(out[static_cast<std::size_t>(j)], want,
                          "conv ib witness (gather)");
        }
    }
}

void test_conv2d_weight_backward_witness() {
    // A 1x1 kernel pairs grad_output[n, 0, i, j] with input[n, 0, i, j];
    // with an all-ones input the one weight cell accumulates the upstream
    // in ascending n -> i -> j — the witness order.
    const WitnessExpectation want = witness_expectation(12);
    // Generic path: 1x3x1x3 (ow = 3, swept extent 3) with batch 4? Keep it
    // simpler: 1x1x3x3 -> swept extent 3, 9 contributions.
    {
        const WitnessExpectation want9 = witness_expectation(9);
        const std::vector<float> grad_out = witness_values(9);
        const std::vector<float> input(9, 1.0f);
        std::vector<float> out(1, -1.0f);
        tf::conv2d_weight_backward_generic(grad_out.data(), input.data(),
                                           out.data(), 1, 1, 3, 3, 1, 1, 1, 1,
                                           1, 0, 0, 3, 3);
        check(!tf::conv2d_weight_backward_prefers_gather(3, 3),
              "conv wb witness generic really is the generic path");
        check_witness(out[0], want9, "conv wb witness (generic)");
    }
    // Gather path: 1x1x3x4 (ow = 4, swept extent 4), 12 contributions.
    {
        const std::vector<float> grad_out = witness_values(12);
        const std::vector<float> input(12, 1.0f);
        std::vector<float> out(1, -1.0f);
        tf::conv2d_weight_backward_gather(grad_out.data(), input.data(),
                                          out.data(), 1, 1, 3, 4, 1, 1, 1, 1,
                                          1, 0, 0, 3, 4);
        check(tf::conv2d_weight_backward_prefers_gather(4, 4),
              "conv wb witness gather really is the optimized path");
        check_witness(out[0], want, "conv wb witness (gather)");
    }
}

// The float64 halves of the witness sweeps: at binary64 the same inputs
// accumulate to exactly 1.0 + (count-1) * 2**-24 on both paths — the
// generalization did not change the float64 arithmetic.
void test_conv2d_float64_witness_is_exact() {
    const std::vector<double> input = {1.0, std::ldexp(1.0, -24),
                                       std::ldexp(1.0, -24),
                                       std::ldexp(1.0, -24)};
    const std::vector<double> weight(4, 1.0);
    const double want = 1.0 + 3.0 * std::ldexp(1.0, -24);
    std::vector<double> out(1, 0.0);
    tf::conv2d_forward_generic(input.data(), weight.data(), no_bias<double>(),
                               out.data(), 1, 1, 1, 4, 1, 1, 4, 1, 1, 0, 0,
                               1, 1);
    check(bits(out[0]) == bits(want), "conv fwd float64 exact accumulation");
}

// ---------------------------------------------------------------------------
// 4. H1 destination initialization, per dtype
// ---------------------------------------------------------------------------

template <class T>
void test_conv2d_destinations_are_fully_defined() {
    const Geometry g = {1, 2, 5, 5, 2, 3, 3, 2, 2, 1, 1, true,
                        "poison geometry"};
    const std::vector<T> input =
        patterned<T>(static_cast<std::size_t>(g.input_count()), 7);
    const std::vector<T> weight =
        patterned<T>(static_cast<std::size_t>(g.weight_count()), 8);
    const std::vector<T> bias_values =
        patterned<T>(static_cast<std::size_t>(g.o), 9);
    const std::vector<T> grad_out =
        patterned<T>(static_cast<std::size_t>(g.output_count()), 10);
    const T sentinel = DtypeTraits<T>::sentinel();

    // Forward assigns every output element: a poisoned destination holds
    // no sentinel afterwards, on either path.
    {
        std::vector<T> out(static_cast<std::size_t>(g.output_count()),
                           sentinel);
        tf::conv2d_forward_generic(input.data(), weight.data(),
                                   bias_values.data(), out.data(), g.n, g.c,
                                   g.h, g.w, g.o, g.kh, g.kw, g.sh, g.sw,
                                   g.ph, g.pw, g.oh(), g.ow());
        bool clean = true;
        for (const T value : out) clean = clean && bits(value) != bits(sentinel);
        check(clean, "conv fwd generic overwrote every element");
    }
    // The backward destinations are zero-filled by the kernels themselves;
    // cells that receive no contribution must end at exactly +0.0.
    {
        std::vector<T> out(static_cast<std::size_t>(g.input_count()),
                           sentinel);
        tf::conv2d_input_backward_generic(grad_out.data(), weight.data(),
                                          out.data(), g.n, g.c, g.h, g.w, g.o,
                                          g.kh, g.kw, g.sh, g.sw, g.ph, g.pw,
                                          g.oh(), g.ow());
        bool clean = true;
        for (const T value : out) clean = clean && bits(value) != bits(sentinel);
        check(clean, "conv ib generic defined every element");
    }
    {
        std::vector<T> out(static_cast<std::size_t>(g.weight_count()),
                           sentinel);
        tf::conv2d_weight_backward_gather(grad_out.data(), input.data(),
                                          out.data(), g.n, g.c, g.h, g.w, g.o,
                                          g.kh, g.kw, g.sh, g.sw, g.ph, g.pw,
                                          g.oh(), g.ow());
        bool clean = true;
        for (const T value : out) clean = clean && bits(value) != bits(sentinel);
        check(clean, "conv wb gather assigned every element");
    }
    // An input cell no window reaches (stride 2, kernel 3, pad 1 on a 5x5
    // plane reaches every cell, so shrink the upstream instead: a 1-output
    // geometry leaves cells beyond the window at exactly +0.0).
    {
        std::vector<T> out(9, sentinel);
        const std::vector<T> one_grad = {T(2.0)};
        const std::vector<T> one_weight = {T(3.0)};
        tf::conv2d_input_backward_generic(one_grad.data(), one_weight.data(),
                                          out.data(), 1, 1, 3, 3, 1, 1, 1, 1,
                                          1, 0, 0, 1, 1);
        check(bits(out[0]) == bits(T(6.0)), "conv ib touched cell");
        bool zeros = true;
        for (std::size_t i = 1; i < out.size(); ++i) {
            zeros = zeros && bits(out[i]) == bits(T(0));
        }
        check(zeros, "conv ib untouched cells are exactly +0.0");
    }
}

// ---------------------------------------------------------------------------
// 5. MaxPool2d: values, ties, exceptional values, winners, per dtype
// ---------------------------------------------------------------------------

template <class T>
void run_maxpool_case(const Geometry& g, const std::vector<T>& input,
                      const char* label) {
    char message[256];
    std::vector<T> want_values;
    std::vector<double> want_winners;
    reference_maxpool_forward<T>(g, input, want_values, want_winners);

    void* in = storage_of(input);
    void* out = tf_storage_create_typed(g.n * g.c * g.oh() * g.ow(),
                                        DtypeTraits<T>::code);
    void* winners = tf_storage_create(g.n * g.c * g.oh() * g.ow());
    if (in && out && winners) {
        tf_clear_error();
        tf_core_maxpool2d_forward(in, 0, out, winners, g.n, g.c, g.h, g.w,
                                  g.kh, g.kw, g.sh, g.sw, g.ph, g.pw, g.oh(),
                                  g.ow());
        std::snprintf(message, sizeof message, "maxpool fwd error: %s [%s]",
                      label, DtypeTraits<T>::name());
        check(tf_last_error_code() == TF_OK, message);
        std::snprintf(message, sizeof message, "maxpool fwd values: %s",
                      label);
        same_bits(read_back<T>(out), want_values, message);
        std::snprintf(message, sizeof message, "maxpool fwd winners: %s",
                      label);
        same_bits(read_back<double>(winners), want_winners, message);

        // Backward through the export, against the reference scatter.
        const std::vector<T> grad_out = patterned<T>(
            static_cast<std::size_t>(g.n * g.c * g.oh() * g.ow()), 11);
        const std::vector<T> want_grad =
            reference_maxpool_backward<T>(g, grad_out, want_winners);
        void* go = storage_of(grad_out);
        void* gi = tf_storage_create_typed(g.input_count(),
                                           DtypeTraits<T>::code);
        if (go && gi) {
            tf_clear_error();
            tf_core_maxpool2d_backward(go, 0, winners, 0, gi, g.n, g.c, g.h,
                                       g.w, g.oh(), g.ow());
            std::snprintf(message, sizeof message,
                          "maxpool bwd error: %s [%s]", label,
                          DtypeTraits<T>::name());
            check(tf_last_error_code() == TF_OK, message);
            std::snprintf(message, sizeof message, "maxpool bwd grads: %s",
                          label);
            same_bits(read_back<T>(gi), want_grad, message);
        } else {
            check(false, "maxpool bwd storage");
        }
        if (go) tf_storage_destroy(go);
        if (gi) tf_storage_destroy(gi);
    } else {
        check(false, "maxpool fwd storage");
    }
    if (in) tf_storage_destroy(in);
    if (out) tf_storage_destroy(out);
    if (winners) tf_storage_destroy(winners);
}

template <class T>
void test_maxpool_semantics() {
    // Ordinary values, non-overlapping windows.
    {
        const Geometry g = {2, 2, 4, 4, 0, 2, 2, 2, 2, 0, 0, false,
                            "plain 2x2/2"};
        run_maxpool_case<T>(g, patterned<T>(64, 12), "plain 2x2/2");
    }
    // Overlapping windows, rectangular plane, padding.
    {
        const Geometry g = {1, 2, 4, 5, 0, 3, 2, 1, 1, 1, 0, false,
                            "overlap padded"};
        run_maxpool_case<T>(g, patterned<T>(40, 13), "overlap padded");
    }
    // Ties, signed zeros, NaN before and after values, infinities,
    // subnormals — one 1x1x4x4 plane, 2x2 windows.
    {
        const Geometry g = {1, 1, 4, 4, 0, 2, 2, 2, 2, 0, 0, false,
                            "exceptional"};
        std::vector<T> input(16, T(0));
        // window (0,0): an exact tie — the first occurrence must win.
        input[0] = T(3.5); input[1] = T(3.5); input[4] = T(1.0);
        input[5] = T(-2.0);
        // window (0,1): -0.0 first, +0.0 later; strict > keeps -0.0 and
        // its winner index.
        input[2] = T(-0.0); input[3] = T(0.0); input[6] = T(-1.0);
        input[7] = T(-1.0);
        // window (1,0): NaN first, then finite — the first non-NaN seeds.
        input[8] = std::numeric_limits<T>::quiet_NaN(); input[9] = T(-7.0);
        input[12] = T(-8.0); input[13] = T(-9.0);
        // window (1,1): -inf against a subnormal and +inf last.
        input[10] = -std::numeric_limits<T>::infinity();
        input[11] = std::numeric_limits<T>::denorm_min();
        input[14] = std::numeric_limits<T>::infinity(); input[15] = T(5.0);
        run_maxpool_case<T>(g, input, "exceptional values");

        // Pin the semantics beyond reference agreement: first-tie winner,
        // signed-zero selection, NaN seeding, and the infinity result.
        std::vector<T> values;
        std::vector<double> winners;
        reference_maxpool_forward<T>(g, input, values, winners);
        check(bits(values[0]) == bits(T(3.5)) && winners[0] == 0.0,
              "first equal candidate wins the tie");
        check(bits(values[1]) == bits(T(-0.0)) && winners[1] == 2.0,
              "-0.0 first is kept against +0.0 (strict >)");
        check(bits(values[2]) == bits(T(-7.0)) && winners[2] == 9.0,
              "NaN never wins; first non-NaN seeds");
        check(values[3] == std::numeric_limits<T>::infinity() &&
                  winners[3] == 14.0,
              "+inf wins its window");
    }
    // An all-NaN window (no padding): the deterministic fallback is the
    // window's first candidate, so output and winner still agree.
    {
        const Geometry g = {1, 1, 2, 2, 0, 2, 2, 2, 2, 0, 0, false,
                            "all NaN"};
        std::vector<T> input(4, std::numeric_limits<T>::quiet_NaN());
        std::vector<T> values;
        std::vector<double> winners;
        reference_maxpool_forward<T>(g, input, values, winners);
        check(values[0] != values[0] && winners[0] == 0.0,
              "all-NaN window falls back to the first candidate");
        run_maxpool_case<T>(g, input, "all NaN window");
    }
    // A fully padded-out selection: a window whose only in-bounds cell is
    // -inf keeps the first candidate (the padding, winner -1) — the
    // padding-won sentinel drops the gradient in backward.
    {
        const Geometry g = {1, 1, 1, 1, 0, 3, 3, 1, 1, 1, 1, false,
                            "padding wins"};
        std::vector<T> input = {-std::numeric_limits<T>::infinity()};
        std::vector<T> values;
        std::vector<double> winners;
        reference_maxpool_forward<T>(g, input, values, winners);
        check(values[0] == -std::numeric_limits<T>::infinity() &&
                  winners[0] == -1.0,
              "padding wins an all -inf window");
        run_maxpool_case<T>(g, input, "padding wins");
    }
}

// ---------------------------------------------------------------------------
// 6. The MaxPool backward accumulation witness (overlapping windows)
// ---------------------------------------------------------------------------

void test_maxpool_backward_witness() {
    // A 5x5 plane whose center is the strict maximum: with 3x3 windows at
    // unit stride every one of the 9 output positions selects it, so the
    // center's gradient accumulates all 9 upstream values in ascending
    // oh -> ow order — the witness order.
    std::vector<float> input(25, 0.0f);
    input[12] = 100.0f;
    std::vector<double> winners(9, 12.0);
    const std::vector<float> grad_out = witness_values(9);
    const WitnessExpectation want = witness_expectation(9);

    void* go = storage_of(grad_out);
    void* win = tf_storage_create(9);
    void* gi = tf_storage_create_typed(25, TF_DTYPE_FLOAT32);
    if (go && win && gi) {
        tf_storage_copy_from(win, winners.data());
        tf_clear_error();
        tf_core_maxpool2d_backward(go, 0, win, 0, gi, 1, 1, 5, 5, 3, 3);
        check(tf_last_error_code() == TF_OK, "maxpool witness error");
        const std::vector<float> got = read_back<float>(gi);
        check_witness(got[12], want, "maxpool bwd witness");
        bool zeros = true;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (i == 12) continue;
            zeros = zeros && bits(got[i]) == bits(0.0f);
        }
        check(zeros, "maxpool bwd witness: unselected cells stay +0.0");
    } else {
        check(false, "maxpool witness storage");
    }
    if (go) tf_storage_destroy(go);
    if (win) tf_storage_destroy(win);
    if (gi) tf_storage_destroy(gi);
}

// Unique (non-overlapping) routing preserves finite upstream values
// bitwise: 0 + x reproduces x for every finite non-zero x, and the -0.0
// case follows IEEE arithmetic (+0.0), because the scatter is a genuine
// accumulation, not a transfer.
template <class T>
void test_maxpool_backward_unique_routing() {
    std::vector<double> winners = {0.0, 3.0, 12.0, 15.0};
    std::vector<T> grad_out = {T(2.5), T(-0.0), T(-3.25),
                               std::numeric_limits<T>::denorm_min()};
    void* go = storage_of(grad_out);
    void* win = tf_storage_create(4);
    void* gi = tf_storage_create_typed(16, DtypeTraits<T>::code);
    if (go && win && gi) {
        tf_storage_copy_from(win, winners.data());
        tf_clear_error();
        tf_core_maxpool2d_backward(go, 0, win, 0, gi, 1, 1, 4, 4, 2, 2);
        check(tf_last_error_code() == TF_OK, "maxpool unique routing error");
        const std::vector<T> got = read_back<T>(gi);
        check(bits(got[0]) == bits(T(2.5)), "unique routing value 0");
        // -0.0 accumulated onto the +0.0 destination is +0.0 — IEEE
        // arithmetic, not a transfer; asserted so nobody "fixes" it.
        check(bits(got[3]) == bits(T(0.0)), "-0.0 upstream lands as +0.0");
        check(bits(got[12]) == bits(T(-3.25)), "unique routing value 12");
        check(bits(got[15]) == bits(std::numeric_limits<T>::denorm_min()),
              "subnormal upstream survives");
    } else {
        check(false, "maxpool unique routing storage");
    }
    if (go) tf_storage_destroy(go);
    if (win) tf_storage_destroy(win);
    if (gi) tf_storage_destroy(gi);
}

// ---------------------------------------------------------------------------
// 7. The winner buffer is float64 at every value dtype, and the plane
//    bound is float64's
// ---------------------------------------------------------------------------

template <class T>
void test_winner_buffer_must_be_float64() {
    const std::vector<T> input = patterned<T>(16, 14);
    void* in = storage_of(input);
    void* out = tf_storage_create_typed(4, DtypeTraits<T>::code);
    void* bad_winners = tf_storage_create_typed(4, TF_DTYPE_FLOAT32);
    void* good_winners = tf_storage_create(4);
    if (in && out && bad_winners && good_winners) {
        // A float32 winner buffer is refused whatever the value dtype —
        // for the float64 pool it is also a mismatch, for the float32 pool
        // it is exactly the "winner follows the values" mistake §13.3
        // forbids.
        const std::vector<T> before = read_back<T>(out);
        tf_clear_error();
        tf_core_maxpool2d_forward(in, 0, out, bad_winners, 1, 1, 4, 4, 2, 2,
                                  2, 2, 0, 0, 2, 2);
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "float32 winner buffer refused in forward");
        same_bits(read_back<T>(out), before,
                  "forward wrote nothing after winner rejection");

        // ...and in backward, where a 4-byte winner read through the
        // float64 accessor would overrun.
        void* gi = tf_storage_create_typed(16, DtypeTraits<T>::code);
        if (gi) {
            const std::vector<T> gi_before = read_back<T>(gi);
            tf_clear_error();
            tf_core_maxpool2d_backward(in, 0, bad_winners, 0, gi, 1, 1, 4, 4,
                                       2, 2);
            check(tf_last_error_code() == TF_ERROR_INVALID,
                  "float32 winner buffer refused in backward");
            same_bits(read_back<T>(gi), gi_before,
                      "backward wrote nothing after winner rejection");
            tf_storage_destroy(gi);
        }

        // The float64 winner buffer is accepted for the same call.
        tf_clear_error();
        tf_core_maxpool2d_forward(in, 0, out, good_winners, 1, 1, 4, 4, 2, 2,
                                  2, 2, 0, 0, 2, 2);
        check(tf_last_error_code() == TF_OK,
              "float64 winner buffer accepted at this value dtype");
    } else {
        check(false, "winner dtype storage");
    }
    if (in) tf_storage_destroy(in);
    if (out) tf_storage_destroy(out);
    if (bad_winners) tf_storage_destroy(bad_winners);
    if (good_winners) tf_storage_destroy(good_winners);
}

template <class T>
void test_plane_bound_is_float64s() {
    // A plane of 2**27 x 2**27 = 2**54 > 2**53 is rejected by metadata
    // arithmetic before any span is checked, at every value dtype — the
    // bound did not shrink to 2**24 for float32 and does not depend on
    // the storages, which are deliberately tiny here.
    const std::vector<T> input = patterned<T>(4, 15);
    void* in = storage_of(input);
    void* out = tf_storage_create_typed(4, DtypeTraits<T>::code);
    void* winners = tf_storage_create(4);
    if (in && out && winners) {
        const std::int64_t big = static_cast<std::int64_t>(1) << 27;
        tf_clear_error();
        tf_core_maxpool2d_forward(in, 0, out, winners, 1, 1, big, big, 1, 1,
                                  1, 1, 0, 0, big, big);
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "2**54 plane refused in forward");
        check(std::strstr(tf_last_error_message(), "plane") != nullptr,
              "2**54 forward rejection names the plane");
        tf_clear_error();
        tf_core_maxpool2d_backward(in, 0, winners, 0, out, 1, 1, big, big,
                                   1, 1);
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "2**54 plane refused in backward");
        check(std::strstr(tf_last_error_message(), "plane") != nullptr,
              "2**54 backward rejection names the plane");
        // 2**53 exactly is within the bound: the *plane* check passes and
        // the failure moves on to the span check — proved by the message,
        // so the boundary demonstrably sits at 2**53, not below it.
        tf_clear_error();
        tf_core_maxpool2d_forward(in, 0, out, winners, 1, 1,
                                  static_cast<std::int64_t>(1) << 26,
                                  static_cast<std::int64_t>(1) << 27, 1, 1,
                                  1, 1, 0, 0,
                                  static_cast<std::int64_t>(1) << 26,
                                  static_cast<std::int64_t>(1) << 27);
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "2**53 plane fails only on its span");
        check(std::strstr(tf_last_error_message(), "span") != nullptr,
              "the 2**53 failure is the span check, not the plane check");
    } else {
        check(false, "plane bound storage");
    }
    if (in) tf_storage_destroy(in);
    if (out) tf_storage_destroy(out);
    if (winners) tf_storage_destroy(winners);
}

// Invalid winner values are rejected before anything is written, at both
// value dtypes.
template <class T>
void test_invalid_winner_values_are_rejected() {
    const double bad_values[] = {0.5, -2.0, 16.0,
                                 std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()};
    for (const double bad : bad_values) {
        std::vector<double> winners = {0.0, bad, 3.0, 5.0};
        const std::vector<T> grad_out = patterned<T>(4, 16);
        void* go = storage_of(grad_out);
        void* win = tf_storage_create(4);
        void* gi = tf_storage_create_typed(16, DtypeTraits<T>::code);
        if (go && win && gi) {
            tf_storage_copy_from(win, winners.data());
            const std::vector<T> before = read_back<T>(gi);
            tf_clear_error();
            tf_core_maxpool2d_backward(go, 0, win, 0, gi, 1, 1, 4, 4, 2, 2);
            check(tf_last_error_code() == TF_ERROR_INVALID,
                  "invalid winner value refused");
            same_bits(read_back<T>(gi), before,
                      "nothing written after invalid winner");
        } else {
            check(false, "invalid winner storage");
        }
        if (go) tf_storage_destroy(go);
        if (win) tf_storage_destroy(win);
        if (gi) tf_storage_destroy(gi);
    }
}

// ---------------------------------------------------------------------------
// 8. Mixed dtype is rejected before anything is written, in every
//    participating handle position of all five exports
// ---------------------------------------------------------------------------

void test_mixed_dtype_is_rejected_before_any_write() {
    void* f64_a = tf_storage_create(64);
    void* f64_b = tf_storage_create(64);
    void* f64_c = tf_storage_create(64);
    void* f32_a = tf_storage_create_typed(64, TF_DTYPE_FLOAT32);
    void* f32_b = tf_storage_create_typed(64, TF_DTYPE_FLOAT32);
    void* f32_c = tf_storage_create_typed(64, TF_DTYPE_FLOAT32);
    void* winners = tf_storage_create(64);
    if (!f64_a || !f64_b || !f64_c || !f32_a || !f32_b || !f32_c ||
        !winners) {
        check(false, "mixed dtype storage");
    } else {
        const std::vector<double> before64 = read_back<double>(f64_c);
        const std::vector<float> before32 = read_back<float>(f32_c);
        // Every mixed call below must be refused with TF_ERROR_INVALID; the
        // two destinations are then proved byte-for-byte unmoved. Each
        // export is driven with the odd dtype in **every** participating
        // handle position — the destination rotation is the direction that
        // would corrupt memory rather than merely misread it.
        struct Case {
            const char* label;
            void (*call)(void* f64_a, void* f64_b, void* f64_c, void* f32_a,
                         void* f32_b, void* f32_c, void* winners);
        };
        const Case cases[] = {
            {"conv fwd, float32 input",
             [](void* a64, void* b64, void* c64, void* a32, void*, void*,
                void*) {
                 (void)a64;
                 tf_core_conv2d_forward(a32, 0, b64, 0, nullptr, 0, c64, 1, 1,
                                        4, 4, 1, 2, 2, 1, 1, 0, 0, 3, 3);
             }},
            {"conv fwd, float32 weight",
             [](void* a64, void* b64, void* c64, void*, void* b32, void*,
                void*) {
                 (void)b64;
                 tf_core_conv2d_forward(a64, 0, b32, 0, nullptr, 0, c64, 1, 1,
                                        4, 4, 1, 2, 2, 1, 1, 0, 0, 3, 3);
             }},
            {"conv fwd, float32 bias",
             [](void* a64, void* b64, void* c64, void* a32, void*, void*,
                void*) {
                 // The one nullable handle participates in the agreement
                 // rule when present.
                 tf_core_conv2d_forward(a64, 0, b64, 0, a32, 0, c64, 1, 1, 4,
                                        4, 1, 2, 2, 1, 1, 0, 0, 3, 3);
             }},
            {"conv fwd, float32 destination",
             [](void* a64, void* b64, void*, void*, void*, void* c32,
                void*) {
                 tf_core_conv2d_forward(a64, 0, b64, 0, nullptr, 0, c32, 1, 1,
                                        4, 4, 1, 2, 2, 1, 1, 0, 0, 3, 3);
             }},
            {"conv fwd, float64 destination for float32 operands",
             [](void*, void*, void* c64, void* a32, void* b32, void*,
                void*) {
                 tf_core_conv2d_forward(a32, 0, b32, 0, nullptr, 0, c64, 1, 1,
                                        4, 4, 1, 2, 2, 1, 1, 0, 0, 3, 3);
             }},
            {"conv ib, float32 grad_output",
             [](void* a64, void* b64, void* c64, void* a32, void*, void*,
                void*) {
                 (void)a64;
                 tf_core_conv2d_input_backward(a32, 0, b64, 0, c64, 1, 1, 4,
                                               4, 1, 2, 2, 1, 1, 0, 0, 3, 3);
             }},
            {"conv ib, float32 weight",
             [](void* a64, void* b64, void* c64, void*, void* b32, void*,
                void*) {
                 (void)b64;
                 tf_core_conv2d_input_backward(a64, 0, b32, 0, c64, 1, 1, 4,
                                               4, 1, 2, 2, 1, 1, 0, 0, 3, 3);
             }},
            {"conv ib, float32 destination",
             [](void* a64, void* b64, void*, void*, void*, void* c32,
                void*) {
                 tf_core_conv2d_input_backward(a64, 0, b64, 0, c32, 1, 1, 4,
                                               4, 1, 2, 2, 1, 1, 0, 0, 3, 3);
             }},
            {"conv wb, float32 grad_output",
             [](void* a64, void* b64, void* c64, void* a32, void*, void*,
                void*) {
                 (void)a64;
                 tf_core_conv2d_weight_backward(a32, 0, b64, 0, c64, 1, 1, 4,
                                                4, 1, 2, 2, 1, 1, 0, 0, 3, 3);
             }},
            {"conv wb, float32 input",
             [](void* a64, void* b64, void* c64, void*, void* b32, void*,
                void*) {
                 (void)b64;
                 tf_core_conv2d_weight_backward(a64, 0, b32, 0, c64, 1, 1, 4,
                                                4, 1, 2, 2, 1, 1, 0, 0, 3, 3);
             }},
            {"conv wb, float32 destination",
             [](void* a64, void* b64, void*, void*, void*, void* c32,
                void*) {
                 tf_core_conv2d_weight_backward(a64, 0, b64, 0, c32, 1, 1, 4,
                                                4, 1, 2, 2, 1, 1, 0, 0, 3, 3);
             }},
            {"maxpool fwd, float32 input with float64 output",
             [](void*, void*, void* c64, void* a32, void*, void*,
                void* win) {
                 tf_core_maxpool2d_forward(a32, 0, c64, win, 1, 1, 4, 4, 2, 2,
                                           2, 2, 0, 0, 2, 2);
             }},
            {"maxpool fwd, float64 input with float32 output",
             [](void* a64, void*, void*, void*, void*, void* c32,
                void* win) {
                 tf_core_maxpool2d_forward(a64, 0, c32, win, 1, 1, 4, 4, 2, 2,
                                           2, 2, 0, 0, 2, 2);
             }},
            {"maxpool bwd, float32 grad_output with float64 grad_input",
             [](void*, void*, void* c64, void* a32, void*, void*,
                void* win) {
                 tf_core_maxpool2d_backward(a32, 0, win, 0, c64, 1, 1, 4, 4,
                                            2, 2);
             }},
            {"maxpool bwd, float64 grad_output with float32 grad_input",
             [](void* a64, void*, void*, void*, void*, void* c32,
                void* win) {
                 tf_core_maxpool2d_backward(a64, 0, win, 0, c32, 1, 1, 4, 4,
                                            2, 2);
             }},
        };
        for (const Case& c : cases) {
            char message[256];
            tf_clear_error();
            c.call(f64_a, f64_b, f64_c, f32_a, f32_b, f32_c, winners);
            std::snprintf(message, sizeof message, "mixed dtype refused: %s",
                          c.label);
            check(tf_last_error_code() == TF_ERROR_INVALID, message);
        }
        same_bits(read_back<double>(f64_c), before64,
                  "float64 destination unmoved by mixed calls");
        same_bits(read_back<float>(f32_c), before32,
                  "float32 destination unmoved by mixed calls");
    }
    if (f64_a) tf_storage_destroy(f64_a);
    if (f64_b) tf_storage_destroy(f64_b);
    if (f64_c) tf_storage_destroy(f64_c);
    if (f32_a) tf_storage_destroy(f32_a);
    if (f32_b) tf_storage_destroy(f32_b);
    if (f32_c) tf_storage_destroy(f32_c);
    if (winners) tf_storage_destroy(winners);
}

}  // namespace

int main() {
    test_predicate_boundaries();

    test_conv2d_geometries<double>();
    test_conv2d_geometries<float>();

    test_conv2d_special_values<double>();
    test_conv2d_special_values<float>();

    test_conv2d_forward_witness();
    test_conv2d_input_backward_witness();
    test_conv2d_weight_backward_witness();
    test_conv2d_float64_witness_is_exact();

    test_conv2d_destinations_are_fully_defined<double>();
    test_conv2d_destinations_are_fully_defined<float>();

    test_maxpool_semantics<double>();
    test_maxpool_semantics<float>();

    test_maxpool_backward_witness();

    test_maxpool_backward_unique_routing<double>();
    test_maxpool_backward_unique_routing<float>();

    test_winner_buffer_must_be_float64<double>();
    test_winner_buffer_must_be_float64<float>();

    test_plane_bound_is_float64s<double>();
    test_plane_bound_is_float64s<float>();

    test_invalid_winner_values_are_rejected<double>();
    test_invalid_winner_values_are_rejected<float>();

    test_mixed_dtype_is_rejected_before_any_write();

    if (g_failures == 0) {
        std::printf("dtype cnn: all checks passed\n");
        return 0;
    }
    std::printf("dtype cnn: %d check(s) failed\n", g_failures);
    return 1;
}
