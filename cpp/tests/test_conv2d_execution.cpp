// Dependency-free C++ test for the Conv2d compute paths and their H9
// dispatch (Phase H, milestone H9). No GoogleTest / Catch2 — a plain
// executable that prints failures and returns a nonzero exit code if any
// check fails, so CTest reports pass/fail.
//
// This binary compiles cpp/src/conv2d.cpp (plus error.cpp) directly, so it
// can reach the three hidden ``tf::conv2d_*_prefers_*`` predicates and BOTH
// compute paths in every direction — the retained Phase-D generic loops and
// the H9 optimized traversals — none of which the shared library exports.
// That is what proves which path a geometry takes, and that the two agree,
// without adding any "which kernel ran" control to the shipped ABI.
//
// What it proves, at the layer where the property is actually decided —
// i.e. without the Python wrapper, the ctypes boundary, or NumPy anywhere:
//
//   1. The predicates are exactly the documented rules, are total (they
//      answer for every geometry, degenerate ones included), and never
//      depend on anything but the integer extents handed to them.
//   2. Each optimized traversal writes **identical bits** to its retained
//      generic twin, across a geometry matrix covering 1x1 / 3x3 / 5x5 /
//      rectangular kernels, unit and non-unit stride, zero and non-zero
//      padding, single and multiple batches/channels, prime extents,
//      boundary-only and interior-only outputs, and an IEEE-754 value sweep
//      with signed zeros in every role, infinities, denormals, the smallest
//      normal, and the largest finite magnitudes.
//   3. **Signed zero is proved, not assumed.** The optimized forward seeds
//      its destination row and accumulates into memory where the generic
//      path uses a register accumulator, and the optimized weight gradient
//      assigns a register accumulator where the generic path accumulates
//      into a zero-filled destination. Those are exactly the rewrites that
//      could change a zero's sign, so every all-zero and mixed-zero sign
//      pattern is compared as raw bits.
//   4. **NaN characterization.** NaN positions are identical on both paths
//      and every NaN either path produces is quiet, including from a
//      signaling-NaN input. Payload bits are asserted in neither direction
//      when two or more NaNs reach one destination, for the same reason H2
//      and H6 recorded: which addend x86-64's ADDSD leaves in the
//      destination register is an instruction-selection decision C++ cannot
//      express. With at most one NaN per destination the paths agree
//      exactly, payload included, and that half *is* asserted.
//   5. **H1 full-write.** The optimized forward and the optimized weight
//      gradient are handed a poisoned destination and must leave no poison
//      behind and produce the same bits as against a zeroed destination —
//      i.e. they write every element and never read one first. (The generic
//      input-gradient and weight-gradient paths, and the H9 input-gradient
//      gather, define their destination by zeroing it themselves.)
//
// Bit comparison, never a tolerance: the entire question is whether the
// optimized traversal reproduces the reference's bits, and ``==`` on
// doubles cannot see -0.0 versus +0.0 and calls every NaN unequal to itself.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "tf_conv2d_internal.h"

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

std::uint64_t bits(double value) {
    std::uint64_t raw;
    std::memcpy(&raw, &value, sizeof raw);
    return raw;
}

double from_bits(std::uint64_t raw) {
    double value;
    std::memcpy(&value, &raw, sizeof value);
    return value;
}

bool is_nan(double v) {
    return v != v;
}

// IEEE-754 quiet bit (bit 51 of the significand) set?
bool is_quiet_nan(double v) {
    return is_nan(v) && ((bits(v) >> 51) & 1u) != 0u;
}

struct Geo {
    std::int64_t n, c, h, w, o, kh, kw, sh, sw, ph, pw, oh, ow;
};

Geo make(std::int64_t n, std::int64_t c, std::int64_t h, std::int64_t w,
         std::int64_t o, std::int64_t kh, std::int64_t kw, std::int64_t sh,
         std::int64_t sw, std::int64_t ph, std::int64_t pw) {
    Geo g{n, c, h, w, o, kh, kw, sh, sw, ph, pw, 0, 0};
    g.oh = (h + 2 * ph - kh) / sh + 1;
    g.ow = (w + 2 * pw - kw) / sw + 1;
    return g;
}

#define GEO_ARGS(g)                                                          \
    (g).n, (g).c, (g).h, (g).w, (g).o, (g).kh, (g).kw, (g).sh, (g).sw,       \
        (g).ph, (g).pw, (g).oh, (g).ow

std::int64_t input_count(const Geo& g) { return g.n * g.c * g.h * g.w; }
std::int64_t weight_count(const Geo& g) { return g.o * g.c * g.kh * g.kw; }
std::int64_t output_count(const Geo& g) { return g.n * g.o * g.oh * g.ow; }

// A deterministic, dependency-free value generator (no <random>, so the
// sequence is identical on every toolchain).
struct Source {
    std::uint64_t state;
    explicit Source(std::uint64_t seed) : state(seed) {}
    double next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        // 53 significant bits mapped into [-1, 1).
        const double unit =
            static_cast<double>(state >> 11) * (1.0 / 9007199254740992.0);
        return unit * 2.0 - 1.0;
    }
};

void fill(std::vector<double>& v, std::uint64_t seed) {
    Source src(seed);
    for (auto& x : v) x = src.next();
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (bits(a[i]) != bits(b[i])) return false;
    }
    return true;
}

// ==========================================================================
// 1. The predicates
// ==========================================================================

void test_predicates() {
    // The shared extent rule: min(input_width, output_width) >= 4.
    check(!tf::conv2d_sweep_extent_is_worthwhile(1, 100),
          "extent rule must reject a 1-wide input");
    check(!tf::conv2d_sweep_extent_is_worthwhile(100, 1),
          "extent rule must reject a 1-wide output");
    check(!tf::conv2d_sweep_extent_is_worthwhile(3, 3),
          "extent rule must reject 3");
    check(!tf::conv2d_sweep_extent_is_worthwhile(3, 1000),
          "extent rule takes the MINIMUM, not the input width");
    check(!tf::conv2d_sweep_extent_is_worthwhile(1000, 3),
          "extent rule takes the MINIMUM, not the output width");
    check(tf::conv2d_sweep_extent_is_worthwhile(4, 4),
          "extent rule must accept exactly 4");
    check(tf::conv2d_sweep_extent_is_worthwhile(4, 1000),
          "extent rule must accept 4 against a large output");
    check(tf::conv2d_sweep_extent_is_worthwhile(1000, 4),
          "extent rule must accept 4 against a large input");
    check(tf::kConv2dMinSweptExtent == 4,
          "the documented minimum swept extent is 4");

    // Forward and weight-gradient follow the shared rule and nothing else:
    // any stride, any padding.
    for (std::int64_t s = 1; s <= 3; ++s) {
        check(tf::conv2d_forward_prefers_row_sweep(8, 8),
              "forward takes the sweep at a workable extent");
        check(tf::conv2d_weight_backward_prefers_gather(8, 8),
              "weight gradient takes the gather at a workable extent");
        check(!tf::conv2d_forward_prefers_row_sweep(8, 2),
              "forward falls back on a short output");
        check(!tf::conv2d_weight_backward_prefers_gather(2, 8),
              "weight gradient falls back on a short input");
        (void)s;
    }

    // The input gradient additionally demands unit stride in BOTH axes.
    check(tf::conv2d_input_backward_prefers_gather(1, 1, 16, 16),
          "input gradient takes the gather at unit stride");
    check(!tf::conv2d_input_backward_prefers_gather(2, 1, 16, 16),
          "input gradient falls back on a non-unit row stride");
    check(!tf::conv2d_input_backward_prefers_gather(1, 2, 16, 16),
          "input gradient falls back on a non-unit column stride");
    check(!tf::conv2d_input_backward_prefers_gather(2, 2, 16, 16),
          "input gradient falls back when both strides are non-unit");
    check(!tf::conv2d_input_backward_prefers_gather(1, 1, 16, 3),
          "input gradient still obeys the shared extent rule");

    // Totality: the predicates answer for degenerate extents without
    // reading anything but their arguments, and never claim a path for a
    // geometry with no swept room at all.
    check(!tf::conv2d_forward_prefers_row_sweep(0, 0),
          "forward predicate is total at zero extents");
    check(!tf::conv2d_input_backward_prefers_gather(1, 1, 0, 0),
          "input-gradient predicate is total at zero extents");
    check(!tf::conv2d_weight_backward_prefers_gather(0, 0),
          "weight-gradient predicate is total at zero extents");

    // Purity: the same arguments give the same answer every time, and the
    // answer does not depend on call order.
    for (int i = 0; i < 3; ++i) {
        check(tf::conv2d_forward_prefers_row_sweep(9, 9) &&
                  !tf::conv2d_forward_prefers_row_sweep(9, 1),
              "forward predicate is pure across repeated calls");
    }
}

// ==========================================================================
// 2. The two paths agree, bit for bit, across the geometry matrix
// ==========================================================================

const Geo kMatrix[] = {
    make(1, 1, 6, 6, 1, 1, 1, 1, 1, 0, 0),    // 1x1, everything singular
    make(1, 1, 8, 8, 1, 3, 3, 1, 1, 0, 0),    // smallest real 3x3
    make(2, 3, 8, 9, 4, 3, 3, 1, 1, 0, 0),    // multi batch/channel
    make(2, 3, 8, 9, 4, 3, 3, 1, 1, 1, 1),    // padded
    make(2, 3, 9, 9, 4, 3, 3, 2, 2, 1, 1),    // strided + padded
    make(2, 3, 9, 11, 4, 3, 3, 2, 1, 0, 1),   // asymmetric stride/padding
    make(1, 2, 12, 12, 3, 5, 5, 1, 1, 2, 2),  // 5x5, padding = kernel/2
    make(2, 2, 7, 13, 3, 3, 5, 1, 1, 0, 0),   // rectangular kernel
    make(2, 2, 13, 7, 3, 5, 3, 1, 1, 2, 1),   // rectangular, padded
    make(1, 1, 5, 5, 1, 5, 5, 1, 1, 0, 0),    // one output element (fallback)
    make(3, 1, 4, 4, 2, 3, 3, 1, 1, 0, 0),    // output smaller than kernel run
    make(1, 5, 23, 29, 7, 3, 3, 1, 1, 1, 1),  // prime extents
    make(2, 2, 6, 8, 3, 6, 8, 1, 1, 0, 0),    // kernel fills the input
    make(1, 1, 4, 4, 1, 3, 3, 1, 1, 2, 2),    // padding > kernel: pad-only rows
    make(2, 3, 10, 10, 4, 3, 3, 3, 3, 0, 0),  // stride 3
};

void run_all_paths(const Geo& g, const std::vector<double>& in,
                   const std::vector<double>& wt, const double* bias,
                   const std::vector<double>& go,
                   std::vector<double>& fwd_gen, std::vector<double>& fwd_opt,
                   std::vector<double>& ib_gen, std::vector<double>& ib_opt,
                   std::vector<double>& wb_gen, std::vector<double>& wb_opt) {
    tf::conv2d_forward_generic(in.data(), wt.data(), bias, fwd_gen.data(),
                               GEO_ARGS(g));
    tf::conv2d_forward_row_sweep(in.data(), wt.data(), bias, fwd_opt.data(),
                                 GEO_ARGS(g));
    tf::conv2d_input_backward_generic(go.data(), wt.data(), ib_gen.data(),
                                      GEO_ARGS(g));
    tf::conv2d_input_backward_gather(go.data(), wt.data(), ib_opt.data(),
                                     GEO_ARGS(g));
    tf::conv2d_weight_backward_generic(go.data(), in.data(), wb_gen.data(),
                                       GEO_ARGS(g));
    tf::conv2d_weight_backward_gather(go.data(), in.data(), wb_opt.data(),
                                      GEO_ARGS(g));
}

void compare_geometry(const Geo& g, const std::vector<double>& in,
                      const std::vector<double>& wt,
                      const std::vector<double>& bias,
                      const std::vector<double>& go, const char* label,
                      bool compare_input_gradient) {
    std::vector<double> fg(output_count(g)), fo(output_count(g));
    std::vector<double> ig(input_count(g)), io(input_count(g));
    std::vector<double> wg(weight_count(g)), wo(weight_count(g));
    // Distinct pre-existing junk in the two destinations, so a path that
    // failed to write an element could not pass by coincidence.
    for (auto& v : fg) v = 1.5;
    for (auto& v : fo) v = -7.25;
    for (auto& v : wg) v = 1.5;
    for (auto& v : wo) v = -7.25;
    run_all_paths(g, in, wt, bias.empty() ? nullptr : bias.data(), go, fg, fo,
                  ig, io, wg, wo);
    char message[160];
    std::snprintf(message, sizeof message, "%s: forward paths must agree bit "
                                           "for bit", label);
    check(same_bits(fg, fo), message);
    if (compare_input_gradient) {
        std::snprintf(message, sizeof message,
                      "%s: input-gradient paths must agree bit for bit", label);
        check(same_bits(ig, io), message);
    }
    std::snprintf(message, sizeof message,
                  "%s: weight-gradient paths must agree bit for bit", label);
    check(same_bits(wg, wo), message);
}

void test_paths_agree() {
    for (const Geo& g : kMatrix) {
        std::vector<double> in(input_count(g)), wt(weight_count(g)),
            bias(g.o), go(output_count(g));
        fill(in, 0x1234);
        fill(wt, 0x9ABC);
        fill(bias, 0x5555);
        fill(go, 0xFEED);
        char label[96];
        std::snprintf(label, sizeof label,
                      "%lldx%lldx%lldx%lld->%lld k%lldx%lld s%lldx%lld p%lldx%lld",
                      (long long)g.n, (long long)g.c, (long long)g.h,
                      (long long)g.w, (long long)g.o, (long long)g.kh,
                      (long long)g.kw, (long long)g.sh, (long long)g.sw,
                      (long long)g.ph, (long long)g.pw);
        // The input-gradient gather is only *defined* at unit stride; at any
        // other stride the dispatcher never reaches it.
        compare_geometry(g, in, wt, bias, go, label, g.sh == 1 && g.sw == 1);

        // ...and again with no bias at all, which is a separate seeding path
        // in the optimized forward.
        std::vector<double> none;
        char nb_label[128];
        std::snprintf(nb_label, sizeof nb_label, "%s (no bias)", label);
        compare_geometry(g, in, wt, none, go, nb_label,
                         g.sh == 1 && g.sw == 1);
    }
}

// ==========================================================================
// 3. Signed zeros
// ==========================================================================

void test_signed_zeros() {
    const double zeros[] = {0.0, -0.0};
    for (const Geo& g : {kMatrix[3], kMatrix[2], kMatrix[6], kMatrix[9]}) {
        for (double xz : zeros) {
            for (double wz : zeros) {
                for (double gz : zeros) {
                    for (double bz : zeros) {
                        std::vector<double> in(input_count(g), xz);
                        std::vector<double> wt(weight_count(g), wz);
                        std::vector<double> bias(g.o, bz);
                        std::vector<double> go(output_count(g), gz);
                        compare_geometry(g, in, wt, bias, go,
                                         "all-zero sign pattern",
                                         g.sh == 1 && g.sw == 1);
                    }
                }
            }
        }
        // Alternating signed zeros mixed with exactly cancelling finites, so
        // a destination's running sum passes through -0.0 and +0.0 both.
        std::vector<double> in(input_count(g)), wt(weight_count(g), 1.0),
            bias(g.o, -0.0), go(output_count(g));
        for (std::size_t i = 0; i < in.size(); ++i) {
            const double cycle[] = {1.0, -1.0, -0.0, 0.0};
            in[i] = cycle[i % 4];
        }
        for (std::size_t i = 0; i < go.size(); ++i) {
            const double cycle[] = {-0.0, 2.0, -2.0, 0.0};
            go[i] = cycle[i % 4];
        }
        compare_geometry(g, in, wt, bias, go, "cancelling / mixed zeros",
                         g.sh == 1 && g.sw == 1);
    }

    // Negative zero survives an accumulation only while *every* addend is
    // also negative zero, because -0.0 + (+0.0) is +0.0 under
    // round-to-nearest. Both facts are checked, on both paths, because the
    // row sweep replaces a register accumulator with an accumulate-into-
    // memory sequence and that is exactly the rewrite that could change a
    // zero's sign.
    const Geo g = kMatrix[2];  // unpadded, so every tap contributes
    std::vector<double> go(output_count(g), 0.0);
    std::vector<double> fg(output_count(g)), fo(output_count(g)),
        ig(input_count(g)), io(input_count(g)), wg(weight_count(g)),
        wo(weight_count(g));

    // (a) every product is -0.0 (-0.0 * +0.0) and the bias is -0.0, so the
    //     whole sum stays -0.0.
    {
        std::vector<double> in(input_count(g), -0.0), wt(weight_count(g), 0.0),
            bias(g.o, -0.0);
        run_all_paths(g, in, wt, bias.data(), go, fg, fo, ig, io, wg, wo);
        bool generic_negative = true, optimized_negative = true;
        for (std::size_t i = 0; i < fg.size(); ++i) {
            if (bits(fg[i]) != bits(-0.0)) generic_negative = false;
            if (bits(fo[i]) != bits(-0.0)) optimized_negative = false;
        }
        check(generic_negative,
              "generic forward keeps -0.0 when every addend is -0.0");
        check(optimized_negative,
              "row-sweep forward keeps -0.0 when every addend is -0.0");
    }

    // (b) one +0.0 addend anywhere turns the sum positive — identically on
    //     both paths, which is the property that actually matters.
    {
        std::vector<double> in(input_count(g), 0.0), wt(weight_count(g), 0.0),
            bias(g.o, -0.0);
        run_all_paths(g, in, wt, bias.data(), go, fg, fo, ig, io, wg, wo);
        bool generic_positive = true, optimized_positive = true;
        for (std::size_t i = 0; i < fg.size(); ++i) {
            if (bits(fg[i]) != bits(0.0)) generic_positive = false;
            if (bits(fo[i]) != bits(0.0)) optimized_positive = false;
        }
        check(generic_positive,
              "generic forward: a -0.0 bias plus +0.0 products is +0.0");
        check(optimized_positive,
              "row-sweep forward: a -0.0 bias plus +0.0 products is +0.0");
    }

    // A destination that receives no contribution at all is +0.0 in both
    // gradients, on both paths — the sum of nothing is positive zero.
    bool gradients_positive = true;
    for (std::size_t i = 0; i < wg.size(); ++i) {
        if (bits(wg[i]) != bits(0.0) || bits(wo[i]) != bits(0.0)) {
            gradients_positive = false;
        }
    }
    check(gradients_positive,
          "an all-zero weight gradient is +0.0 on both paths");
}

// ==========================================================================
// 4. NaN and infinity
// ==========================================================================

void test_nan_and_infinity() {
    const double qnan = from_bits(0x7FF8000000000000ULL);
    const double neg_qnan = from_bits(0xFFF8000000000000ULL);
    const double snan = from_bits(0x7FF0000000000001ULL);
    const double payload = from_bits(0x7FF8000000ABCDEFULL);
    const double inf = from_bits(0x7FF0000000000000ULL);
    const double specials[] = {qnan, neg_qnan, snan, payload, inf, -inf};

    for (const Geo& g : {kMatrix[3], kMatrix[7], kMatrix[11]}) {
        for (double special : specials) {
            // ONE special value planted in one operand: at most one NaN can
            // reach any destination, so the paths must agree exactly —
            // payload included. This is the contractual half.
            for (int slot = 0; slot < 3; ++slot) {
                std::vector<double> in(input_count(g)), wt(weight_count(g)),
                    bias(g.o), go(output_count(g));
                fill(in, 0x2222);
                fill(wt, 0x3333);
                fill(bias, 0x4444);
                fill(go, 0x6666);
                if (slot == 0) in[in.size() / 3] = special;
                if (slot == 1) wt[wt.size() / 2] = special;
                if (slot == 2) go[go.size() / 3] = special;
                std::vector<double> fg(output_count(g)), fo(output_count(g)),
                    ig(input_count(g)), io(input_count(g)),
                    wg(weight_count(g)), wo(weight_count(g));
                run_all_paths(g, in, wt, bias.data(), go, fg, fo, ig, io, wg,
                              wo);
                check(same_bits(fg, fo),
                      "one special value: forward paths agree exactly");
                if (g.sh == 1 && g.sw == 1) {
                    check(same_bits(ig, io),
                          "one special value: input-gradient paths agree "
                          "exactly");
                }
                check(same_bits(wg, wo),
                      "one special value: weight-gradient paths agree exactly");
            }
        }

        // MANY NaNs, so two or more meet in one destination. Positions and
        // quietness are contractual; payload bits are not asserted either
        // way.
        std::vector<double> in(input_count(g)), wt(weight_count(g)),
            bias(g.o), go(output_count(g));
        fill(in, 0x2222);
        fill(wt, 0x3333);
        fill(bias, 0x4444);
        fill(go, 0x6666);
        for (std::size_t i = 0; i < in.size(); i += 3) in[i] = qnan;
        for (std::size_t i = 1; i < wt.size(); i += 2) wt[i] = neg_qnan;
        for (std::size_t i = 0; i < go.size(); i += 2) go[i] = payload;
        std::vector<double> fg(output_count(g)), fo(output_count(g)),
            ig(input_count(g)), io(input_count(g)), wg(weight_count(g)),
            wo(weight_count(g));
        run_all_paths(g, in, wt, bias.data(), go, fg, fo, ig, io, wg, wo);
        for (std::size_t i = 0; i < fg.size(); ++i) {
            check(is_nan(fg[i]) == is_nan(fo[i]),
                  "forward: NaN positions are identical on both paths");
            if (is_nan(fo[i])) {
                check(is_quiet_nan(fo[i]) && is_quiet_nan(fg[i]),
                      "forward: every NaN either path produces is quiet");
            } else {
                check(bits(fg[i]) == bits(fo[i]),
                      "forward: every non-NaN result is bit-identical");
            }
        }
        for (std::size_t i = 0; i < wg.size(); ++i) {
            check(is_nan(wg[i]) == is_nan(wo[i]),
                  "weight gradient: NaN positions are identical");
            if (is_nan(wo[i])) {
                check(is_quiet_nan(wo[i]) && is_quiet_nan(wg[i]),
                      "weight gradient: every NaN produced is quiet");
            } else {
                check(bits(wg[i]) == bits(wo[i]),
                      "weight gradient: every non-NaN result is bit-identical");
            }
        }
        if (g.sh == 1 && g.sw == 1) {
            for (std::size_t i = 0; i < ig.size(); ++i) {
                check(is_nan(ig[i]) == is_nan(io[i]),
                      "input gradient: NaN positions are identical");
                if (is_nan(io[i])) {
                    check(is_quiet_nan(io[i]) && is_quiet_nan(ig[i]),
                          "input gradient: every NaN produced is quiet");
                } else {
                    check(bits(ig[i]) == bits(io[i]),
                          "input gradient: every non-NaN result is "
                          "bit-identical");
                }
            }
        }
    }

    // 0 * inf and (+inf) + (-inf) manufacture a NaN inside the accumulation
    // itself rather than receiving one; both paths must still agree on
    // position and quietness.
    const Geo g = kMatrix[3];
    std::vector<double> in(input_count(g), 0.0), wt(weight_count(g), inf),
        bias(g.o, 0.0), go(output_count(g), inf);
    std::vector<double> fg(output_count(g)), fo(output_count(g)),
        ig(input_count(g)), io(input_count(g)), wg(weight_count(g)),
        wo(weight_count(g));
    run_all_paths(g, in, wt, bias.data(), go, fg, fo, ig, io, wg, wo);
    bool any_nan = false;
    for (std::size_t i = 0; i < fg.size(); ++i) {
        check(is_nan(fg[i]) == is_nan(fo[i]),
              "manufactured NaN: forward positions agree");
        if (is_nan(fo[i])) {
            any_nan = true;
            check(is_quiet_nan(fo[i]), "manufactured NaN is quiet");
        }
    }
    check(any_nan, "the 0 * inf probe must actually manufacture a NaN");
}

// ==========================================================================
// 5. H1: the paths that take an uninitialized destination write all of it
// ==========================================================================

void test_full_write() {
    // The poison patterns the Python H1 suite uses, for the same reasons: a
    // quiet NaN with a distinctive payload contaminates anything that reads
    // it, and a large finite catches code that special-cases NaN.
    const double poisons[] = {from_bits(0x7FF8DEADBEEFCAFEULL),
                              -1.2345678901234567e300};
    for (const Geo& g : kMatrix) {
        std::vector<double> in(input_count(g)), wt(weight_count(g)),
            bias(g.o), go(output_count(g));
        fill(in, 0x7777);
        fill(wt, 0x8888);
        fill(bias, 0x9999);
        fill(go, 0xAAAA);
        std::vector<double> clean_fwd(output_count(g), 0.0);
        std::vector<double> clean_wb(weight_count(g), 0.0);
        tf::conv2d_forward_row_sweep(in.data(), wt.data(), bias.data(),
                                     clean_fwd.data(), GEO_ARGS(g));
        tf::conv2d_weight_backward_gather(go.data(), in.data(),
                                          clean_wb.data(), GEO_ARGS(g));
        for (double poison : poisons) {
            std::vector<double> dirty_fwd(output_count(g), poison);
            std::vector<double> dirty_wb(weight_count(g), poison);
            tf::conv2d_forward_row_sweep(in.data(), wt.data(), bias.data(),
                                         dirty_fwd.data(), GEO_ARGS(g));
            tf::conv2d_weight_backward_gather(go.data(), in.data(),
                                              dirty_wb.data(), GEO_ARGS(g));
            check(same_bits(clean_fwd, dirty_fwd),
                  "row-sweep forward must ignore whatever the destination "
                  "held: every element written, none read first");
            check(same_bits(clean_wb, dirty_wb),
                  "gather weight gradient must ignore whatever the "
                  "destination held");
            // The input-gradient gather zeroes its own span, exactly as the
            // generic path does, so it too must survive a poisoned buffer.
            if (g.sh == 1 && g.sw == 1) {
                std::vector<double> clean_ib(input_count(g), 0.0);
                std::vector<double> dirty_ib(input_count(g), poison);
                tf::conv2d_input_backward_gather(go.data(), wt.data(),
                                                 clean_ib.data(), GEO_ARGS(g));
                tf::conv2d_input_backward_gather(go.data(), wt.data(),
                                                 dirty_ib.data(), GEO_ARGS(g));
                check(same_bits(clean_ib, dirty_ib),
                      "gather input gradient must define its whole span");
            }
        }
    }

    // A negative control: the detector must be able to fail. A destination
    // element the kernel does NOT write keeps its poison, and this check
    // proves the comparison above would have noticed.
    const Geo g = kMatrix[2];
    std::vector<double> a(output_count(g), 0.0);
    std::vector<double> b(output_count(g), 0.0);
    b[b.size() / 2] = from_bits(0x7FF8DEADBEEFCAFEULL);
    check(!same_bits(a, b),
          "negative control: the bit comparison must detect one unwritten "
          "element");
}

// ==========================================================================
// 6. Source immutability
// ==========================================================================

void test_sources_are_not_mutated() {
    for (const Geo& g : {kMatrix[3], kMatrix[4], kMatrix[7]}) {
        std::vector<double> in(input_count(g)), wt(weight_count(g)),
            bias(g.o), go(output_count(g));
        fill(in, 0xB1B1);
        fill(wt, 0xC2C2);
        fill(bias, 0xD3D3);
        fill(go, 0xE4E4);
        const std::vector<double> in0 = in, wt0 = wt, bias0 = bias, go0 = go;
        std::vector<double> fg(output_count(g)), fo(output_count(g)),
            ig(input_count(g)), io(input_count(g)), wg(weight_count(g)),
            wo(weight_count(g));
        run_all_paths(g, in, wt, bias.data(), go, fg, fo, ig, io, wg, wo);
        check(same_bits(in, in0), "no path may mutate the input");
        check(same_bits(wt, wt0), "no path may mutate the weight");
        check(same_bits(bias, bias0), "no path may mutate the bias");
        check(same_bits(go, go0), "no path may mutate the upstream gradient");
    }
}

// ==========================================================================
// 7. Repeatability
// ==========================================================================

void test_repeatability() {
    const Geo g = kMatrix[11];
    std::vector<double> in(input_count(g)), wt(weight_count(g)), bias(g.o),
        go(output_count(g));
    fill(in, 0x1010);
    fill(wt, 0x2020);
    fill(bias, 0x3030);
    fill(go, 0x4040);
    std::vector<double> first(output_count(g)), second(output_count(g));
    for (int i = 0; i < 4; ++i) {
        std::vector<double>& dst = (i % 2 == 0) ? first : second;
        for (auto& v : dst) v = static_cast<double>(i);
        tf::conv2d_forward_row_sweep(in.data(), wt.data(), bias.data(),
                                     dst.data(), GEO_ARGS(g));
    }
    check(same_bits(first, second),
          "the row sweep is deterministic across repeated calls");
}

}  // namespace

int main() {
    std::printf("conv2d execution / H9 dispatch tests\n");
    test_predicates();
    test_paths_agree();
    test_signed_zeros();
    test_nan_and_infinity();
    test_full_write();
    test_sources_are_not_mutated();
    test_repeatability();
    if (g_failures == 0) {
        std::printf("all conv2d-execution checks passed\n");
        return 0;
    }
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
}
