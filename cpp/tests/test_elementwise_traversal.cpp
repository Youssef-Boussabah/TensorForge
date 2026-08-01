// Dependency-free C++ test for the elementwise plan builders and the
// templated traversals they feed (Phase H, milestone H8). No GoogleTest /
// Catch2 — a plain executable that prints failures and returns a nonzero
// exit code if any check fails, so CTest reports pass/fail.
//
// This binary compiles cpp/src/elementwise.cpp (plus storage.cpp and
// error.cpp) directly, so it can reach BOTH the hidden
// ``tf::build_unary_plan`` / ``tf::build_binary_plan`` builders and the
// templated traversals, as well as the exported wrappers they live inside.
//
// What it proves, at the layer where the property is actually decided —
// i.e. without the Python wrapper, the ctypes boundary, or NumPy anywhere in
// the picture:
//
//   1. The builders are **total**: they answer for rank 0, unit extents,
//      contiguous layouts, transposes, narrows, non-unit strides, negative
//      strides, broadcast stride-0 layouts, ranks 1-5, and degenerate
//      metadata alike, and never allocate, never set the error slot, and
//      never mutate their inputs.
//   2. The collapse is exactly the documented two transformations — drop
//      unit axes, merge adjacent axes whose address progressions form one
//      run for **every** operand at once — with the collapsed extents'
//      product always equal to the logical element count.
//   3. A rejected plan is a **fallback**, never an error: the exported
//      wrapper still produces the odometer's answer.
//   4. The planned traversal and the retained odometer write **identical
//      bits** over an IEEE-754 sweep covering signed zeros, infinities,
//      denormals, the smallest normal, the largest finite magnitudes, quiet
//      NaNs of both signs with distinct payloads, and signaling NaNs, in
//      every operand position — with exactly one qualification, measured
//      rather than assumed and stated in full in
//      tf_elementwise_internal.h:
//
//      * **Subtraction and the ReLU backward are bit-exact**, two-NaN pairs
//        included. Neither is commutative, so the compiler has no freedom
//        over which operand reaches the destination register.
//      * **Addition and multiplication are bit-exact for every pair with at
//        most one NaN**, and their NaN positions and quietness are exact
//        everywhere, but when **both** operands are NaN the surviving
//        payload is asserted in **neither** direction. That is not something
//        the plan introduced: the pre-H8 library's own flat kernel and its
//        own odometer already disagreed on 30 of 196 such pairs, and the
//        post-H8 paths disagree on 5.
//
//      Note this is a *different* qualification from H2's and H6's, which
//      concerned NaNs meeting inside an accumulation. Here there is no
//      accumulation at all — every output element is a function of exactly
//      one element of each source — only operand order inside a single
//      commutative instruction.
//
//   5. Every destination element is written exactly once (H1's contract),
//      proved by poisoning the destination and checking no poison survives
//      and no element outside the logical span is touched.
//
// Bit comparison, never a tolerance: the entire question is whether the
// optimized traversal reproduces the reference's bits, and ``==`` on doubles
// cannot see -0.0 versus +0.0 and calls every NaN unequal to itself.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "tf_elementwise_internal.h"
#include "tf_internal.h"

TF_EXPORT void* tf_storage_create(std::int64_t size);
TF_EXPORT void tf_storage_destroy(void* handle);
// Retyped at Phase I, milestone I2: the host positions are ``void*`` and
// the storage handle's dtype tag decides how they are read. A source-level
// retype of an existing export — same symbol, same slots, no ABI change.
TF_EXPORT void tf_storage_copy_from(void* handle, const void* src);
TF_EXPORT void tf_storage_copy_to(const void* handle, void* dst);
TF_EXPORT void tf_core_add(const void* a, const void* b, void* dst,
                           const std::int64_t* shape,
                           const std::int64_t* a_strides,
                           const std::int64_t* b_strides,
                           std::int64_t a_offset, std::int64_t b_offset,
                           std::int64_t ndim);
TF_EXPORT void tf_core_relu(const void* src, void* dst,
                            const std::int64_t* shape,
                            const std::int64_t* strides,
                            std::int64_t offset, std::int64_t ndim);

using std::int64_t;
using std::uint64_t;

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

static double from_bits(uint64_t b) { double d; std::memcpy(&d, &b, 8); return d; }
static uint64_t to_bits(double d) { uint64_t b; std::memcpy(&b, &d, 8); return b; }

// ---------------------------------------------------------------------------
// The retained odometers, transcribed here so the test compares the planned
// traversal against an independent statement of the reference semantics
// rather than against whichever branch the production wrapper happened to
// take. If production ever silently changed the reference, this copy would
// disagree with it.
// ---------------------------------------------------------------------------

template <class Op>
static void reference_unary(const double* src, double* dst,
                            const int64_t* shape, const int64_t* strides,
                            int64_t offset, int64_t ndim) {
    if (ndim == 0) { dst[0] = Op::apply(src[offset]); return; }
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) total *= shape[d];
    std::vector<int64_t> counter(static_cast<size_t>(ndim), 0);
    int64_t pos = offset;
    for (int64_t out = 0; out < total; ++out) {
        dst[out] = Op::apply(src[pos]);
        for (int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[d];
            pos += strides[d];
            if (counter[d] < shape[d]) break;
            counter[d] = 0;
            pos -= shape[d] * strides[d];
        }
    }
}

template <class Op>
static void reference_binary(const double* a, const double* b, double* dst,
                             const int64_t* shape, const int64_t* as,
                             const int64_t* bs, int64_t ao, int64_t bo,
                             int64_t ndim) {
    if (ndim == 0) { dst[0] = Op::apply(a[ao], b[bo]); return; }
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) total *= shape[d];
    std::vector<int64_t> counter(static_cast<size_t>(ndim), 0);
    int64_t ap = ao, bp = bo;
    for (int64_t out = 0; out < total; ++out) {
        dst[out] = Op::apply(a[ap], b[bp]);
        for (int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[d];
            ap += as[d];
            bp += bs[d];
            if (counter[d] < shape[d]) break;
            counter[d] = 0;
            ap -= shape[d] * as[d];
            bp -= shape[d] * bs[d];
        }
    }
}

// ---------------------------------------------------------------------------
// 1-2. The builders: totality, purity, and the collapse rule.
// ---------------------------------------------------------------------------

struct LayoutCase {
    const char* label;
    int64_t ndim;
    int64_t shape[6];
    int64_t a[6];
    int64_t b[6];
    bool expect_accept;
    int64_t expect_rank;  // -1 when not asserted
};

static void test_builders() {
    std::printf("-- builders --\n");
    const LayoutCase cases[] = {
        // contiguous layouts collapse all the way to one axis
        {"1-D contiguous", 1, {16}, {1}, {1}, true, 1},
        {"2-D contiguous", 2, {8, 4}, {4, 1}, {4, 1}, true, 1},
        {"4-D NCHW contiguous", 4, {2, 3, 4, 5}, {60, 20, 5, 1},
         {60, 20, 5, 1}, true, 1},
        // a fully broadcast operand collapses with a contiguous one
        {"scalar broadcast", 2, {8, 4}, {4, 1}, {0, 0}, true, 1},
        // a per-column operand cannot merge with the row axis
        {"row broadcast", 2, {8, 4}, {4, 1}, {0, 1}, true, 2},
        {"column broadcast", 2, {8, 4}, {4, 1}, {1, 0}, true, 2},
        // NCHW per-channel statistics: H and W merge, N and C do not
        {"NCHW channel broadcast", 4, {2, 3, 4, 5}, {60, 20, 5, 1},
         {0, 1, 0, 0}, true, 3},
        // unit axes vanish entirely
        {"unit axes dropped", 4, {1, 8, 1, 4}, {32, 4, 4, 1}, {32, 4, 4, 1},
         true, 1},
        {"all unit axes", 3, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, true, 0},
        // transposes and narrows keep their axes
        {"2-D transposed", 2, {4, 8}, {1, 4}, {8, 1}, true, 2},
        {"last-axis narrow", 2, {8, 2}, {4, 1}, {2, 1}, true, 2},
        // Non-unit strides are described, not rejected — and they collapse
        // whenever the progressions really do form one run: a row stride of
        // 8 over 4 columns of stride 2 *is* one arithmetic run of 16, and
        // the partner is contiguous, so both merge and the rank is 1.
        {"stride-2 rows", 2, {4, 4}, {8, 2}, {4, 1}, true, 1},
        // ...while a partner that does not share the progression blocks it.
        {"stride-2 rows, blocked", 2, {4, 4}, {8, 2}, {0, 1}, true, 2},
        // negative strides are legal metadata and describable
        {"negative stride", 1, {8}, {-1}, {1}, true, 1},
        // rank 0 declines (the odometer's own branch handles it)
        {"rank 0", 0, {0}, {0}, {0}, false, -1},
        // a non-positive extent declines
        {"zero extent", 1, {0}, {1}, {1}, false, -1},
        {"negative extent", 2, {4, -1}, {1, 1}, {1, 1}, false, -1},
        // rank 5 that cannot collapse below 5 declines
        {"rank-5 reversed", 5, {2, 2, 2, 2, 2}, {1, 2, 4, 8, 16},
         {16, 8, 4, 2, 1}, false, -1},
        // ...but a rank-5 that *does* collapse is accepted
        {"rank-5 contiguous", 5, {2, 2, 2, 2, 2}, {16, 8, 4, 2, 1},
         {16, 8, 4, 2, 1}, true, 1},
        // an unrepresentable element count declines rather than wrap
        {"overflowing element count", 2, {2, INT64_MAX}, {INT64_MAX, 1},
         {INT64_MAX, 1}, false, -1},
        {"overflowing element count, unary-only", 3,
         {INT64_MAX, INT64_MAX, 2}, {1, 1, 1}, {1, 1, 1}, false, -1},
    };

    for (const LayoutCase& c : cases) {
        // purity: the inputs must be untouched
        int64_t shape_copy[6], a_copy[6], b_copy[6];
        std::memcpy(shape_copy, c.shape, sizeof(shape_copy));
        std::memcpy(a_copy, c.a, sizeof(a_copy));
        std::memcpy(b_copy, c.b, sizeof(b_copy));

        tf::ElementwiseBinaryPlan plan;
        const bool ok = tf::build_binary_plan(c.shape, c.a, c.b, c.ndim, plan);
        char msg[160];
        std::snprintf(msg, sizeof(msg), "%s: accepted == %d", c.label,
                      int(c.expect_accept));
        check(ok == c.expect_accept, msg);
        check(std::memcmp(shape_copy, c.shape, sizeof(shape_copy)) == 0,
              "builder mutated shape");
        check(std::memcmp(a_copy, c.a, sizeof(a_copy)) == 0,
              "builder mutated a strides");
        check(std::memcmp(b_copy, c.b, sizeof(b_copy)) == 0,
              "builder mutated b strides");
        check(tf::last_error_code() == TF_OK, "builder set the error slot");

        if (ok && c.expect_rank >= 0) {
            std::snprintf(msg, sizeof(msg), "%s: collapsed rank %lld == %lld",
                          c.label, (long long)plan.ndim,
                          (long long)c.expect_rank);
            check(plan.ndim == c.expect_rank, msg);
            // the collapsed extents' product is the logical element count
            int64_t logical = 1;
            for (int64_t d = 0; d < c.ndim; ++d) logical *= c.shape[d];
            int64_t planned = 1;
            for (int64_t d = 0; d < plan.ndim; ++d) planned *= plan.shape[d];
            std::snprintf(msg, sizeof(msg), "%s: element count preserved",
                          c.label);
            check(planned == logical, msg);
        }

        // the unary builder agrees with the binary one on the a-operand
        // whenever the binary one accepted for structural reasons
        tf::ElementwiseUnaryPlan uplan;
        const bool uok = tf::build_unary_plan(c.shape, c.a, c.ndim, uplan);
        if (!c.expect_accept && c.ndim <= 0) {
            check(!uok, "unary builder declines rank 0 too");
        }
        if (uok) {
            int64_t logical = 1;
            for (int64_t d = 0; d < c.ndim; ++d) logical *= c.shape[d];
            int64_t planned = 1;
            for (int64_t d = 0; d < uplan.ndim; ++d) planned *= uplan.shape[d];
            std::snprintf(msg, sizeof(msg), "%s: unary count preserved",
                          c.label);
            check(planned == logical, msg);
        }
    }

    // A unary layout the binary builder would reject only because of the
    // *other* operand still collapses on its own: the merge rule is a
    // conjunction over the operands, so dropping one can only accept more.
    {
        const int64_t shape[2] = {8, 4};
        const int64_t contiguous[2] = {4, 1};
        const int64_t percolumn[2] = {0, 1};
        tf::ElementwiseUnaryPlan uplan;
        check(tf::build_unary_plan(shape, contiguous, 2, uplan)
                  && uplan.ndim == 1,
              "unary contiguous collapses to rank 1");
        tf::ElementwiseBinaryPlan bplan;
        check(tf::build_binary_plan(shape, contiguous, percolumn, 2, bplan)
                  && bplan.ndim == 2,
              "the same layout stays rank 2 when the partner blocks the merge");
    }
}

// ---------------------------------------------------------------------------
// 4. Bit identity between the planned traversal and the odometer.
// ---------------------------------------------------------------------------

static const uint64_t PATTERNS[] = {
    0x0000000000000000ull,  // +0
    0x8000000000000000ull,  // -0
    0x3FF0000000000000ull,  // 1.0
    0xBFF0000000000000ull,  // -1.0
    0x4008000000000000ull,  // 3.0
    0x7FF0000000000000ull,  // +inf
    0xFFF0000000000000ull,  // -inf
    0x7FF8000000000000ull,  // quiet NaN
    0xFFF8000000000000ull,  // negative quiet NaN
    0x7FF8000ABCDEF123ull,  // quiet NaN, nontrivial payload
    0xFFFDEADBEEF00000ull,  // negative quiet NaN, another payload
    0x7FF0000000000001ull,  // signaling NaN
    0xFFF0000000000001ull,  // negative signaling NaN
    0x0000000000000001ull,  // smallest subnormal
    0x800FFFFFFFFFFFFFull,  // -largest subnormal
    0x0010000000000000ull,  // smallest normal
    0x7FEFFFFFFFFFFFFFull,  // largest finite
    0xFFEFFFFFFFFFFFFFull,  // -largest finite
};
static const int NPAT = int(sizeof(PATTERNS) / sizeof(PATTERNS[0]));

// ``exact`` selects which of the two contract levels this operation is held
// to. Subtraction is not commutative, so every bit matches. Addition and
// multiplication are, so when BOTH operands are a NaN the payload is an
// instruction-selection decision and is asserted in neither direction — see
// part (4) of the contract in tf_elementwise_internal.h. Everything else,
// including every pair with at most one NaN, is still exact.
// ``quiets`` says whether this operation is arithmetic. Addition,
// subtraction, and multiplication quiet a signaling operand because they
// *compute*; the ReLU backward **selects** an operand and copies it
// verbatim, so a signaling NaN legitimately survives it — exactly as H5
// established for the identity gather. Both behaviours are identical on
// both traversals, which is what this file is here to check.
template <class Op>
static void sweep_binary_at(const char* name, bool exact, bool quiets = true) {
    const int n = NPAT * NPAT;
    const int64_t shape[2] = {NPAT, NPAT};
    const int64_t as[2] = {NPAT, 1};
    const int64_t bs[2] = {0, 1};
    std::vector<double> a(static_cast<size_t>(n)), b(static_cast<size_t>(NPAT));
    std::vector<double> ref(static_cast<size_t>(n)), got(static_cast<size_t>(n));
    for (int i = 0; i < NPAT; ++i)
        for (int j = 0; j < NPAT; ++j)
            a[size_t(i * NPAT + j)] = from_bits(PATTERNS[i]);
    for (int j = 0; j < NPAT; ++j) b[size_t(j)] = from_bits(PATTERNS[j]);

    reference_binary<Op>(a.data(), b.data(), ref.data(), shape, as, bs, 0, 0, 2);
    tf::ElementwiseBinaryPlan plan;
    check(tf::build_binary_plan(shape, as, bs, 2, plan), "sweep plan accepted");
    tf::binary_plan_walk<Op>(a.data(), b.data(), got.data(), plan, 0, 0);

    int total = 0, outside_two_nan = 0, not_quiet = 0, wrong_position = 0;
    for (int i = 0; i < n; ++i) {
        const double x = ref[size_t(i)], y = got[size_t(i)];
        const bool nx = (x != x), ny = (y != y);
        if (nx != ny) ++wrong_position;
        if (quiets && ny && (to_bits(y) & 0x0008000000000000ull) == 0)
            ++not_quiet;
        // Whichever the operation does, the two paths must do the same.
        if (nx != ny) { /* counted above */ }
        else if (nx && ((to_bits(x) & 0x0008000000000000ull) == 0)
                    != ((to_bits(y) & 0x0008000000000000ull) == 0))
            ++not_quiet;
        if (to_bits(x) == to_bits(y)) continue;
        ++total;
        const double la = a[size_t(i)], lb = b[size_t(i % NPAT)];
        const bool both_nan = (la != la) && (lb != lb);
        if (!both_nan) ++outside_two_nan;
    }
    char msg[192];
    std::snprintf(msg, sizeof(msg), "%s: %d NaN positions differ", name,
                  wrong_position);
    check(wrong_position == 0, msg);
    std::snprintf(msg, sizeof(msg), "%s: %d produced NaNs are signaling", name,
                  not_quiet);
    check(not_quiet == 0, msg);
    std::snprintf(msg, sizeof(msg),
                  "%s: %d differing results outside two-NaN pairs (must be 0)",
                  name, outside_two_nan);
    check(outside_two_nan == 0, msg);
    if (exact) {
        std::snprintf(msg, sizeof(msg),
                      "%s: %d of %d pairs differ, but this operation is "
                      "contracted exact", name, total, n);
        check(total == 0, msg);
    }
}

template <class Op>
static void sweep_binary(const char* name) {
    // a[i][j] = pattern i, b[j] = pattern j -> every ordered pair, read
    // through a real broadcast (b's row stride is 0), which is also the
    // layout the normalization modules actually produce.
    const int n = NPAT * NPAT;
    const int64_t shape[2] = {NPAT, NPAT};
    const int64_t as[2] = {NPAT, 1};
    const int64_t bs[2] = {0, 1};
    std::vector<double> a(static_cast<size_t>(n)), b(static_cast<size_t>(NPAT));
    std::vector<double> ref(static_cast<size_t>(n)), got(static_cast<size_t>(n));
    for (int i = 0; i < NPAT; ++i)
        for (int j = 0; j < NPAT; ++j)
            a[size_t(i * NPAT + j)] = from_bits(PATTERNS[i]);
    for (int j = 0; j < NPAT; ++j) b[size_t(j)] = from_bits(PATTERNS[j]);

    reference_binary<Op>(a.data(), b.data(), ref.data(), shape, as, bs, 0, 0, 2);
    tf::ElementwiseBinaryPlan plan;
    check(tf::build_binary_plan(shape, as, bs, 2, plan), "sweep plan accepted");
    tf::binary_plan_walk<Op>(a.data(), b.data(), got.data(), plan, 0, 0);

    int diff = 0;
    for (int i = 0; i < n; ++i)
        if (to_bits(ref[size_t(i)]) != to_bits(got[size_t(i)])) ++diff;
    char msg[128];
    std::snprintf(msg, sizeof(msg),
                  "%s: %d of %d ordered pairs differ in bits (must be 0)",
                  name, diff, n);
    check(diff == 0, msg);

    // ...and again with the operands swapped in position, so a payload rule
    // that depended on which side a NaN arrived from would be caught.
    const int64_t as2[2] = {0, 1};
    const int64_t bs2[2] = {NPAT, 1};
    reference_binary<Op>(b.data(), a.data(), ref.data(), shape, as2, bs2, 0, 0, 2);
    check(tf::build_binary_plan(shape, as2, bs2, 2, plan), "swapped plan");
    tf::binary_plan_walk<Op>(b.data(), a.data(), got.data(), plan, 0, 0);
    diff = 0;
    for (int i = 0; i < n; ++i)
        if (to_bits(ref[size_t(i)]) != to_bits(got[size_t(i)])) ++diff;
    std::snprintf(msg, sizeof(msg), "%s (operands swapped): %d differ", name,
                  diff);
    check(diff == 0, msg);
}

template <class Op>
static void sweep_unary(const char* name) {
    const int64_t shape[1] = {NPAT};
    const int64_t st[1] = {1};
    std::vector<double> src(static_cast<size_t>(NPAT)), ref(static_cast<size_t>(NPAT)), got(static_cast<size_t>(NPAT));
    for (int i = 0; i < NPAT; ++i) src[size_t(i)] = from_bits(PATTERNS[i]);
    reference_unary<Op>(src.data(), ref.data(), shape, st, 0, 1);
    tf::ElementwiseUnaryPlan plan;
    check(tf::build_unary_plan(shape, st, 1, plan), "unary sweep plan");
    tf::unary_plan_walk<Op>(src.data(), got.data(), plan, 0);
    int diff = 0;
    for (int i = 0; i < NPAT; ++i)
        if (to_bits(ref[size_t(i)]) != to_bits(got[size_t(i)])) ++diff;
    char msg[128];
    std::snprintf(msg, sizeof(msg), "%s: %d of %d patterns differ in bits",
                  name, diff, NPAT);
    check(diff == 0, msg);

    // ...and through a reversed (negative-stride) view, the layout the
    // generic row branch owns.
    const int64_t back[1] = {-1};
    reference_unary<Op>(src.data() + (NPAT - 1), ref.data(), shape, back, 0, 1);
    check(tf::build_unary_plan(shape, back, 1, plan), "negative-stride plan");
    tf::unary_plan_walk<Op>(src.data() + (NPAT - 1), got.data(), plan, 0);
    diff = 0;
    for (int i = 0; i < NPAT; ++i)
        if (to_bits(ref[size_t(i)]) != to_bits(got[size_t(i)])) ++diff;
    std::snprintf(msg, sizeof(msg), "%s (reversed): %d differ", name, diff);
    check(diff == 0, msg);
}

// A structural sweep: every layout in the builder table, compared
// element-for-element between the odometer and the plan on ordinary data.
static void test_layouts_agree() {
    std::printf("-- layout agreement --\n");
    struct L { const char* label; int64_t ndim, shape[5], a[5], b[5]; };
    const L layouts[] = {
        {"contiguous 2-D", 2, {6, 5}, {5, 1}, {5, 1}},
        {"scalar broadcast", 2, {6, 5}, {5, 1}, {0, 0}},
        {"row broadcast", 2, {6, 5}, {5, 1}, {0, 1}},
        {"column broadcast", 2, {6, 5}, {5, 1}, {1, 0}},
        {"both transposed", 2, {5, 6}, {1, 5}, {1, 5}},
        {"one transposed", 2, {5, 6}, {1, 5}, {6, 1}},
        {"rank-3 middle broadcast", 3, {3, 4, 5}, {20, 5, 1}, {20, 0, 1}},
        {"rank-4 channel", 4, {2, 3, 4, 5}, {60, 20, 5, 1}, {0, 1, 0, 0}},
        {"rank-4 contiguous", 4, {2, 3, 4, 5}, {60, 20, 5, 1}, {60, 20, 5, 1}},
        {"unit axes", 4, {1, 6, 1, 5}, {30, 5, 5, 1}, {0, 1, 1, 0}},
        {"stride-2 source", 2, {3, 4}, {8, 2}, {4, 1}},
        {"prime extents", 2, {7, 11}, {11, 1}, {0, 1}},
        {"rank-5 contiguous", 5, {2, 3, 2, 3, 2}, {36, 12, 6, 2, 1},
         {36, 12, 6, 2, 1}},
    };
    for (const L& l : layouts) {
        int64_t total = 1;
        for (int64_t d = 0; d < l.ndim; ++d) total *= l.shape[d];
        // one buffer large enough for any of these strides
        const size_t big = 4096;
        std::vector<double> a(big), b(big), ref(static_cast<size_t>(total)), got(static_cast<size_t>(total));
        for (size_t i = 0; i < big; ++i) {
            a[i] = 0.5 + double(i) * 0.125;
            b[i] = -0.25 + double(i) * 0.0625;
        }
        reference_binary<tf::MultiplyOp>(a.data(), b.data(), ref.data(),
                                         l.shape, l.a, l.b, 0, 0, l.ndim);
        tf::ElementwiseBinaryPlan plan;
        char msg[128];
        if (!tf::build_binary_plan(l.shape, l.a, l.b, l.ndim, plan)) {
            std::snprintf(msg, sizeof(msg), "%s: unexpectedly rejected",
                          l.label);
            check(false, msg);
            continue;
        }
        tf::binary_plan_walk<tf::MultiplyOp>(a.data(), b.data(), got.data(),
                                             plan, 0, 0);
        int diff = 0;
        for (int64_t i = 0; i < total; ++i)
            if (to_bits(ref[size_t(i)]) != to_bits(got[size_t(i)])) ++diff;
        std::snprintf(msg, sizeof(msg), "%s: %d of %lld elements differ",
                      l.label, diff, (long long)total);
        check(diff == 0, msg);

        // ...and with a nonzero offset on both operands
        reference_binary<tf::AddOp>(a.data(), b.data(), ref.data(), l.shape,
                                    l.a, l.b, 3, 7, l.ndim);
        tf::binary_plan_walk<tf::AddOp>(a.data(), b.data(), got.data(), plan,
                                        3, 7);
        diff = 0;
        for (int64_t i = 0; i < total; ++i)
            if (to_bits(ref[size_t(i)]) != to_bits(got[size_t(i)])) ++diff;
        std::snprintf(msg, sizeof(msg), "%s (offsets): %d differ", l.label,
                      diff);
        check(diff == 0, msg);
    }
}

// ---------------------------------------------------------------------------
// 5. H1: every destination element is written exactly once, and nothing
//    outside the logical span is touched.
// ---------------------------------------------------------------------------

static void test_full_write() {
    std::printf("-- full-write (H1) --\n");
    const uint64_t poisons[2] = {0x7FF8ABCDEF012345ull,   // a quiet NaN
                                 0x4160000000000000ull};  // 8388608.0
    struct L { const char* label; int64_t ndim, shape[4], a[4], b[4]; };
    const L layouts[] = {
        {"contiguous", 2, {6, 5}, {5, 1}, {5, 1}},
        {"scalar broadcast", 2, {6, 5}, {5, 1}, {0, 0}},
        {"row broadcast", 2, {6, 5}, {5, 1}, {0, 1}},
        {"transposed", 2, {5, 6}, {1, 5}, {6, 1}},
        {"rank-4 channel", 4, {2, 3, 4, 5}, {60, 20, 5, 1}, {0, 1, 0, 0}},
        {"unit axes", 4, {1, 6, 1, 5}, {30, 5, 5, 1}, {0, 1, 1, 0}},
    };
    for (const L& l : layouts) {
        int64_t total = 1;
        for (int64_t d = 0; d < l.ndim; ++d) total *= l.shape[d];
        for (uint64_t poison : poisons) {
            std::vector<double> a(4096), b(4096);
            for (size_t i = 0; i < a.size(); ++i) {
                a[i] = 1.0 + double(i) * 0.25;
                b[i] = 2.0 - double(i) * 0.125;
            }
            // one guard element on each side of the logical destination
            std::vector<double> dst(size_t(total) + 2);
            for (size_t i = 0; i < dst.size(); ++i) dst[i] = from_bits(poison);
            tf::ElementwiseBinaryPlan plan;
            check(tf::build_binary_plan(l.shape, l.a, l.b, l.ndim, plan),
                  "full-write plan accepted");
            tf::binary_plan_walk<tf::AddOp>(a.data(), b.data(), dst.data() + 1,
                                            plan, 0, 0);
            int survivors = 0;
            for (int64_t i = 0; i < total; ++i)
                if (to_bits(dst[size_t(i) + 1]) == poison) ++survivors;
            char msg[160];
            std::snprintf(msg, sizeof(msg),
                          "%s: %d poison survivors inside the output",
                          l.label, survivors);
            check(survivors == 0, msg);
            std::snprintf(msg, sizeof(msg), "%s: wrote outside the output",
                          l.label);
            check(to_bits(dst[0]) == poison
                      && to_bits(dst[size_t(total) + 1]) == poison, msg);
        }
    }
    // A negative control: the detector must be able to fail. Leaving one
    // element unwritten must be visible.
    {
        std::vector<double> dst(8, from_bits(poisons[0]));
        std::vector<double> a(8, 1.0), b(8, 2.0);
        const int64_t shape[1] = {7};  // deliberately one short of 8
        const int64_t st[1] = {1};
        tf::ElementwiseBinaryPlan plan;
        check(tf::build_binary_plan(shape, st, st, 1, plan), "control plan");
        tf::binary_plan_walk<tf::AddOp>(a.data(), b.data(), dst.data(), plan,
                                        0, 0);
        int survivors = 0;
        for (int i = 0; i < 8; ++i)
            if (to_bits(dst[size_t(i)]) == poisons[0]) ++survivors;
        check(survivors == 1,
              "negative control: an unwritten element must survive as poison");
    }
}

// ---------------------------------------------------------------------------
// 3. A rejected plan is a fallback, through the real exported wrappers.
// ---------------------------------------------------------------------------

static void test_rejected_plan_still_computes() {
    std::printf("-- rejected plan falls back --\n");
    // rank-5 fully reversed: the builder declines, the odometer answers.
    const int64_t shape[5] = {2, 2, 2, 2, 2};
    const int64_t rev[5] = {1, 2, 4, 8, 16};
    const int64_t fwd[5] = {16, 8, 4, 2, 1};
    tf::ElementwiseBinaryPlan plan;
    check(!tf::build_binary_plan(shape, rev, fwd, 5, plan),
          "rank-5 reversed is declined");

    void* a = tf_storage_create(32);
    void* b = tf_storage_create(32);
    void* dst = tf_storage_create(32);
    std::vector<double> av(32), bv(32), out(32), expected(32);
    for (int i = 0; i < 32; ++i) { av[size_t(i)] = i + 1.0; bv[size_t(i)] = -0.5 * i; }
    tf_storage_copy_from(a, av.data());
    tf_storage_copy_from(b, bv.data());
    tf_core_add(a, b, dst, shape, rev, fwd, 0, 0, 5);
    check(tf::last_error_code() == TF_OK, "declined plan is not an error");
    tf_storage_copy_to(dst, out.data());
    reference_binary<tf::AddOp>(av.data(), bv.data(), expected.data(), shape,
                                rev, fwd, 0, 0, 5);
    int diff = 0;
    for (int i = 0; i < 32; ++i)
        if (to_bits(out[size_t(i)]) != to_bits(expected[size_t(i)])) ++diff;
    check(diff == 0, "fallback produces the odometer's bits");

    // the unary wrapper too
    tf_core_relu(a, dst, shape, rev, 0, 5);
    check(tf::last_error_code() == TF_OK, "declined unary plan is not an error");
    tf_storage_copy_to(dst, out.data());
    reference_unary<tf::ReluOp>(av.data(), expected.data(), shape, rev, 0, 5);
    diff = 0;
    for (int i = 0; i < 32; ++i)
        if (to_bits(out[size_t(i)]) != to_bits(expected[size_t(i)])) ++diff;
    check(diff == 0, "unary fallback produces the odometer's bits");

    tf_storage_destroy(a);
    tf_storage_destroy(b);
    tf_storage_destroy(dst);
}

int main() {
    test_builders();
    // Subtraction and the ReLU backward select an operand rather than
    // combining two, so both are contracted bit-exact. Addition and
    // multiplication are commutative and get the two-NaN qualification.
    sweep_binary_at<tf::AddOp>("add", false);
    sweep_binary_at<tf::SubtractOp>("subtract", true);
    sweep_binary_at<tf::MultiplyOp>("multiply", false);
    sweep_binary_at<tf::ReluBackwardOp>("relu_backward", true, false);
    sweep_binary<tf::SubtractOp>("subtract (swapped-operand sweep)");
    sweep_binary<tf::ReluBackwardOp>("relu_backward (swapped-operand sweep)");
    sweep_unary<tf::ReluOp>("relu");
    sweep_unary<tf::SqrtOp>("sqrt");
    sweep_unary<tf::ReciprocalOp>("reciprocal");
    sweep_unary<tf::IdentityOp>("identity");
    test_layouts_agree();
    test_full_write();
    test_rejected_plan_still_computes();
    if (failures == 0) {
        std::printf("elementwise traversal: all checks passed\n");
        return 0;
    }
    std::printf("elementwise traversal: %d check(s) failed\n", failures);
    return 1;
}
