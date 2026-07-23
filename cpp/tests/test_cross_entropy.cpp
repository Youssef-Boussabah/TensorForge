// Dependency-free C++ test for the fused native cross-entropy (Phase E,
// milestone E5). No GoogleTest/Catch2 — a plain executable that prints
// failures and returns a nonzero exit code if any check fails, so CTest
// reports pass/fail.
//
// This binary compiles cpp/src/classification.cpp (and error.cpp)
// directly, so it exercises **both** layers of that translation unit, the
// way test_softmax.cpp and test_log_softmax.cpp do:
//
//   * tf::cross_entropy_forward_contiguous /
//     tf::cross_entropy_backward_contiguous — the internal, hidden
//     compute kernels (valid, pre-validated arguments only: they are the
//     math, not a validation boundary); and
//   * tf_core_cross_entropy_forward / tf_core_cross_entropy_backward —
//     the exported guarded C ABI wrappers, where every trust-boundary
//     argument must be rejected, driven here through plain tf::Storage
//     handles and the thread-local error slot. That includes the
//     C++-side revalidation of every target index, which is the one
//     operand Python cannot be trusted for at this boundary.
//
// The reference values are computed with the SAME maximum-shift /
// log-sum-exp order the kernel uses, not borrowed from another
// framework: the point is that the fused kernel agrees with the
// algorithm it claims to implement, including at IEEE edges where
// frameworks differ. The reference deliberately never forms
// -log(probability[target]).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "tf_classification_internal.h"
#include "tf_internal.h"  // TF_EXPORT, tf::Storage, error accessors, TfStatus

TF_EXPORT void tf_core_cross_entropy_forward(
    const void* logits_handle, int64_t logits_offset,
    const int64_t* targets, int64_t target_count,
    void* loss_handle, void* probabilities_handle,
    int64_t batch_size, int64_t num_classes, int64_t reduction_code);

TF_EXPORT void tf_core_cross_entropy_backward(
    const void* probabilities_handle, int64_t probabilities_offset,
    const int64_t* targets, int64_t target_count,
    const void* upstream_handle, int64_t upstream_offset,
    void* grad_logits_handle,
    int64_t batch_size, int64_t num_classes, int64_t reduction_code);

namespace {

int g_failures = 0;

const double kPosInf = std::numeric_limits<double>::infinity();
const double kNegInf = -std::numeric_limits<double>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kPoison = 7777.5;

const int64_t kMean = tf::kCrossEntropyReductionMean;
const int64_t kSum = tf::kCrossEntropyReductionSum;

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

void check_eq(double got, double want, const char* what) {
    if (got != want) {
        std::printf("FAIL: %s (got %.17g, want %.17g)\n", what, got, want);
        ++g_failures;
    }
}

void check_close(double got, double want, double tol, const char* what) {
    if (!(std::fabs(got - want) <= tol)) {
        std::printf("FAIL: %s (got %.17g, want %.17g, |diff|=%.3g)\n",
                    what, got, want, std::fabs(got - want));
        ++g_failures;
    }
}

// The same maximum-shift / log-sum-exp algorithm, written independently
// here so a mistake in the kernel does not silently define its own
// reference. Note what it is NOT: -log(softmax(x)[target]).
double reference_loss(const std::vector<double>& logits,
                      const std::vector<int64_t>& targets,
                      int64_t batch_size, int64_t num_classes,
                      int64_t reduction_code) {
    double total = 0.0;
    for (int64_t n = 0; n < batch_size; ++n) {
        const int64_t base = n * num_classes;
        double maximum = logits[static_cast<size_t>(base)];
        for (int64_t c = 1; c < num_classes; ++c) {
            const double value = logits[static_cast<size_t>(base + c)];
            if (value > maximum) {
                maximum = value;
            }
        }
        double sum_exp = 0.0;
        for (int64_t c = 0; c < num_classes; ++c) {
            sum_exp += std::exp(logits[static_cast<size_t>(base + c)] - maximum);
        }
        const double shifted_target =
            logits[static_cast<size_t>(base + targets[static_cast<size_t>(n)])] -
            maximum;
        total += std::log(sum_exp) - shifted_target;
    }
    if (reduction_code == kMean) {
        return total / static_cast<double>(batch_size);
    }
    return total;
}

std::vector<double> reference_probabilities(const std::vector<double>& logits,
                                            int64_t batch_size,
                                            int64_t num_classes) {
    std::vector<double> out(logits.size(), 0.0);
    for (int64_t n = 0; n < batch_size; ++n) {
        const int64_t base = n * num_classes;
        double maximum = logits[static_cast<size_t>(base)];
        for (int64_t c = 1; c < num_classes; ++c) {
            const double value = logits[static_cast<size_t>(base + c)];
            if (value > maximum) {
                maximum = value;
            }
        }
        double sum_exp = 0.0;
        for (int64_t c = 0; c < num_classes; ++c) {
            const double shifted =
                std::exp(logits[static_cast<size_t>(base + c)] - maximum);
            out[static_cast<size_t>(base + c)] = shifted;
            sum_exp += shifted;
        }
        for (int64_t c = 0; c < num_classes; ++c) {
            out[static_cast<size_t>(base + c)] /= sum_exp;
        }
    }
    return out;
}

// -- drivers ----------------------------------------------------------

struct Forward {
    double loss;
    std::vector<double> probabilities;
};

Forward run_forward(const std::vector<double>& logits,
                    const std::vector<int64_t>& targets,
                    int64_t batch_size, int64_t num_classes,
                    int64_t reduction_code) {
    Forward result;
    result.loss = kPoison;
    result.probabilities.assign(logits.size(), kPoison);
    tf::cross_entropy_forward_contiguous(
        logits.data(), targets.data(), &result.loss,
        result.probabilities.data(), batch_size, num_classes, reduction_code);
    return result;
}

std::vector<double> run_backward(const std::vector<double>& probabilities,
                                 const std::vector<int64_t>& targets,
                                 double upstream, int64_t batch_size,
                                 int64_t num_classes, int64_t reduction_code) {
    std::vector<double> grad(probabilities.size(), kPoison);
    tf::cross_entropy_backward_contiguous(
        probabilities.data(), targets.data(), &upstream, grad.data(),
        batch_size, num_classes, reduction_code);
    return grad;
}

int run_checked_forward(const std::vector<double>& logits, int64_t logits_offset,
                        const int64_t* targets, int64_t target_count,
                        std::vector<double>& loss,
                        std::vector<double>& probabilities,
                        int64_t batch_size, int64_t num_classes,
                        int64_t reduction_code) {
    tf::Storage in{const_cast<double*>(logits.data()),
                   static_cast<int64_t>(logits.size())};
    tf::Storage loss_storage{loss.data(), static_cast<int64_t>(loss.size())};
    tf::Storage probability_storage{
        probabilities.data(), static_cast<int64_t>(probabilities.size())};
    tf_core_cross_entropy_forward(&in, logits_offset, targets, target_count,
                                  &loss_storage, &probability_storage,
                                  batch_size, num_classes, reduction_code);
    return tf::last_error_code();
}

int run_checked_backward(const std::vector<double>& probabilities,
                         int64_t probabilities_offset,
                         const int64_t* targets, int64_t target_count,
                         const std::vector<double>& upstream,
                         int64_t upstream_offset,
                         std::vector<double>& grad,
                         int64_t batch_size, int64_t num_classes,
                         int64_t reduction_code) {
    tf::Storage probability_storage{const_cast<double*>(probabilities.data()),
                                    static_cast<int64_t>(probabilities.size())};
    tf::Storage upstream_storage{const_cast<double*>(upstream.data()),
                                 static_cast<int64_t>(upstream.size())};
    tf::Storage grad_storage{grad.data(), static_cast<int64_t>(grad.size())};
    tf_core_cross_entropy_backward(&probability_storage, probabilities_offset,
                                   targets, target_count, &upstream_storage,
                                   upstream_offset, &grad_storage, batch_size,
                                   num_classes, reduction_code);
    return tf::last_error_code();
}

void check_eq_or_both_nan(double got, double want, const char* what) {
    if (std::isnan(got) && std::isnan(want)) {
        return;
    }
    check_eq(got, want, what);
}

void expect_matches_reference(const std::vector<double>& logits,
                              const std::vector<int64_t>& targets,
                              int64_t batch_size, int64_t num_classes,
                              int64_t reduction_code, const char* what) {
    const Forward got =
        run_forward(logits, targets, batch_size, num_classes, reduction_code);
    const double want_loss =
        reference_loss(logits, targets, batch_size, num_classes, reduction_code);
    const std::vector<double> want_probabilities =
        reference_probabilities(logits, batch_size, num_classes);
    char label[160];
    std::snprintf(label, sizeof(label), "%s loss", what);
    check_eq_or_both_nan(got.loss, want_loss, label);
    for (size_t i = 0; i < want_probabilities.size(); ++i) {
        std::snprintf(label, sizeof(label), "%s probabilities[%zu]", what, i);
        check_eq_or_both_nan(got.probabilities[i], want_probabilities[i], label);
    }
}

void expect_rows_sum_to_one(const std::vector<double>& probabilities,
                            int64_t batch_size, int64_t num_classes,
                            const char* what) {
    for (int64_t n = 0; n < batch_size; ++n) {
        double total = 0.0;
        for (int64_t c = 0; c < num_classes; ++c) {
            const double value =
                probabilities[static_cast<size_t>(n * num_classes + c)];
            check(value >= 0.0 && value <= 1.0, "a probability is in [0, 1]");
            total += value;
        }
        char label[160];
        std::snprintf(label, sizeof(label), "%s row %lld sums to 1", what,
                      static_cast<long long>(n));
        check_close(total, 1.0, 1e-12, label);
    }
}

// ======================================================================
// Forward — finite correctness
// ======================================================================

// 1. A single example, two classes: the loss is exactly log(1 + e^-d)
// where d is the margin between the target and the other logit.
void test_batch_size_one_two_classes() {
    const std::vector<double> logits = {2.0, 0.5};
    const std::vector<int64_t> targets = {0};
    for (int64_t reduction : {kMean, kSum}) {
        const Forward got = run_forward(logits, targets, 1, 2, reduction);
        // Both reductions agree for a single example (mean divides by 1).
        check_close(got.loss, std::log(1.0 + std::exp(-1.5)), 1e-15,
                    "batch_size_one loss");
        expect_rows_sum_to_one(got.probabilities, 1, 2, "batch_size_one");
    }
    expect_matches_reference(logits, targets, 1, 2, kSum, "batch_size_one");
}

// 2. A multi-class batch under both reductions, with mean * batch == sum.
void test_multi_class_batch_both_reductions() {
    const std::vector<double> logits = {
        1.0, 2.0, 0.5, -1.0,
        -2.0, 0.25, 3.0, 1.5,
        0.0, 0.0, 0.0, 4.0,
    };
    const std::vector<int64_t> targets = {1, 2, 3};
    const Forward mean = run_forward(logits, targets, 3, 4, kMean);
    const Forward sum = run_forward(logits, targets, 3, 4, kSum);
    check_close(mean.loss, reference_loss(logits, targets, 3, 4, kMean), 1e-15,
                "multi_class mean loss");
    check_close(sum.loss, reference_loss(logits, targets, 3, 4, kSum), 1e-15,
                "multi_class sum loss");
    check_close(mean.loss * 3.0, sum.loss, 1e-14,
                "mean * batch_size == sum");
    expect_rows_sum_to_one(mean.probabilities, 3, 4, "multi_class");
    // The probabilities do not depend on the reduction.
    for (size_t i = 0; i < mean.probabilities.size(); ++i) {
        check_eq(mean.probabilities[i], sum.probabilities[i],
                 "probabilities are reduction independent");
    }
    expect_matches_reference(logits, targets, 3, 4, kMean, "multi_class");
}

// 3. Equal logits: every class is equally likely, so the loss is exactly
// log(num_classes) whatever the targets are.
void test_equal_logits() {
    const std::vector<double> logits(12, 3.25);
    const std::vector<int64_t> targets = {0, 3, 1};
    const Forward got = run_forward(logits, targets, 3, 4, kMean);
    check_close(got.loss, std::log(4.0), 1e-15, "equal_logits loss");
    for (size_t i = 0; i < got.probabilities.size(); ++i) {
        check_eq(got.probabilities[i], 0.25, "equal_logits probability");
    }
    const Forward summed = run_forward(logits, targets, 3, 4, kSum);
    check_close(summed.loss, 3.0 * std::log(4.0), 1e-14, "equal_logits sum");
}

// 4. num_classes == 1: the only class always wins, so the loss is 0.
void test_single_class() {
    const std::vector<double> logits = {5.0, -3.0, 100.0};
    const std::vector<int64_t> targets = {0, 0, 0};
    const Forward got = run_forward(logits, targets, 3, 1, kSum);
    check_eq(got.loss, 0.0, "single_class loss is exactly 0");
    for (size_t i = 0; i < got.probabilities.size(); ++i) {
        check_eq(got.probabilities[i], 1.0, "single_class probability is 1");
    }
}

// 5/6. Large positive and negative common offsets must not overflow and
// must not change the loss — the whole point of the maximum shift. A
// naive log(sum(exp(x))) would give inf at +700 and -inf at -700.
void test_large_common_offsets() {
    const std::vector<double> base = {0.0, 1.0, 2.0, 3.0, -1.0, 0.5, 4.0, 2.5};
    const std::vector<int64_t> targets = {2, 3};
    const double want = reference_loss(base, targets, 2, 4, kMean);
    for (double offset : {700.0, -700.0, 1e5, -1e5, 1e10, -1e10}) {
        std::vector<double> shifted;
        for (double value : base) {
            shifted.push_back(value + offset);
        }
        const Forward got = run_forward(shifted, targets, 2, 4, kMean);
        char label[128];
        std::snprintf(label, sizeof(label), "offset %.3g loss invariance",
                      offset);
        // The addition itself rounds slightly at 1e10, so compare to a
        // tolerance rather than exactly.
        check_close(got.loss, want, 1e-6, label);
        check(std::isfinite(got.loss), "offset loss stays finite");
        expect_rows_sum_to_one(got.probabilities, 2, 4, "large_offset");
    }
}

// 7. Adding a per-row constant leaves that row's loss and probabilities
// unchanged (cross-entropy is shift invariant per row).
void test_additive_shift_invariance() {
    const std::vector<double> base = {0.5, -1.5, 2.25, 1.0, 3.0, -0.25};
    const std::vector<int64_t> targets = {2, 0};
    const Forward want = run_forward(base, targets, 2, 3, kSum);
    std::vector<double> shifted = base;
    for (int c = 0; c < 3; ++c) {
        shifted[static_cast<size_t>(c)] += 37.5;          // row 0 only
        shifted[static_cast<size_t>(3 + c)] -= 12.25;     // row 1 only
    }
    const Forward got = run_forward(shifted, targets, 2, 3, kSum);
    check_close(got.loss, want.loss, 1e-13, "shift invariant loss");
    for (size_t i = 0; i < want.probabilities.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "shift_invariance[%zu]", i);
        check_close(got.probabilities[i], want.probabilities[i], 1e-14, label);
    }
}

// 8. The stable form beats -log(probability[target]) exactly where it is
// supposed to: a target 800 below the row maximum.
void test_stable_loss_beats_the_naive_form() {
    const std::vector<double> logits = {0.0, -800.0};
    const std::vector<int64_t> targets = {1};
    const Forward got = run_forward(logits, targets, 1, 2, kSum);
    check(std::isfinite(got.loss), "a tiny target probability keeps a finite loss");
    check_close(got.loss, 800.0, 1e-9, "small-probability regime is accurate");
    // ...and the naive form really would have failed here.
    check_eq(got.probabilities[1], 0.0, "the probability underflowed to 0");
    check(-std::log(got.probabilities[1]) == kPosInf,
          "-log(probability[target]) would give inf");
}

// 9. Determinism, non-mutation, and both destinations fully written.
void test_determinism_and_non_mutation() {
    const std::vector<double> original = {0.5, -1.5, 2.25, 4.0, -0.75, 3.5};
    std::vector<double> logits = original;
    const std::vector<int64_t> original_targets = {1, 2};
    std::vector<int64_t> targets = original_targets;
    const Forward first = run_forward(logits, targets, 2, 3, kMean);
    const Forward second = run_forward(logits, targets, 2, 3, kMean);
    check_eq(first.loss, second.loss, "loss determinism");
    check(first.loss != kPoison, "loss destination written");
    for (size_t i = 0; i < first.probabilities.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "determinism[%zu]", i);
        check_eq(first.probabilities[i], second.probabilities[i], label);
        std::snprintf(label, sizeof(label), "logits_unmodified[%zu]", i);
        check_eq(logits[i], original[i], label);
        std::snprintf(label, sizeof(label), "probabilities_written[%zu]", i);
        check(first.probabilities[i] != kPoison, label);
    }
    for (size_t i = 0; i < targets.size(); ++i) {
        check(targets[i] == original_targets[i], "targets unmodified");
    }
}

// 10. A nonzero logits offset through the checked wrapper.
void test_checked_nonzero_offset() {
    const std::vector<double> padded = {99.0, 99.0, 1.0, 2.0, 0.5, -1.0};
    const std::vector<int64_t> targets = {1, 0};
    std::vector<double> loss(1, kPoison);
    std::vector<double> probabilities(4, kPoison);
    check(run_checked_forward(padded, 2, targets.data(), 2, loss, probabilities,
                              2, 2, kSum) == TF_OK, "offset status");
    const std::vector<double> tail = {1.0, 2.0, 0.5, -1.0};
    check_eq(loss[0], reference_loss(tail, targets, 2, 2, kSum),
             "checked offset loss");
    const std::vector<double> want = reference_probabilities(tail, 2, 2);
    for (size_t i = 0; i < want.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "checked_offset[%zu]", i);
        check_eq(probabilities[i], want[i], label);
    }
    check_eq(padded[0], 99.0, "offset call left the prefix untouched");
}

// ======================================================================
// Backward
// ======================================================================

// 11/12. Sum and mean reductions against the analytic formula.
void test_backward_matches_the_formula() {
    const std::vector<double> logits = {
        1.0, 2.0, 0.5,
        -1.0, 0.25, 3.0,
    };
    const std::vector<int64_t> targets = {1, 2};
    const Forward forward = run_forward(logits, targets, 2, 3, kSum);
    for (int64_t reduction : {kSum, kMean}) {
        for (double upstream : {1.0, 2.5, -0.75, 0.0}) {
            const std::vector<double> grad = run_backward(
                forward.probabilities, targets, upstream, 2, 3, reduction);
            for (int64_t n = 0; n < 2; ++n) {
                for (int64_t c = 0; c < 3; ++c) {
                    const size_t index = static_cast<size_t>(n * 3 + c);
                    double want = forward.probabilities[index];
                    if (c == targets[static_cast<size_t>(n)]) {
                        want -= 1.0;
                    }
                    if (reduction == kMean) {
                        want /= 2.0;
                    }
                    want *= upstream;
                    char label[160];
                    std::snprintf(label, sizeof(label),
                                  "backward reduction %lld upstream %.3g [%zu]",
                                  static_cast<long long>(reduction), upstream,
                                  index);
                    check_eq(grad[index], want, label);
                }
            }
        }
    }
}

// 13. Every row of the gradient sums to (approximately) zero, whatever
// the reduction or upstream: the probabilities sum to 1 and exactly one
// class has 1 subtracted.
void test_backward_rows_sum_to_zero() {
    const std::vector<double> logits = {
        0.5, -1.0, 2.0, 1.25,
        3.0, 0.0, -2.5, 0.75,
    };
    const std::vector<int64_t> targets = {3, 0};
    const Forward forward = run_forward(logits, targets, 2, 4, kMean);
    for (int64_t reduction : {kSum, kMean}) {
        const std::vector<double> grad =
            run_backward(forward.probabilities, targets, 1.5, 2, 4, reduction);
        for (int64_t n = 0; n < 2; ++n) {
            double total = 0.0;
            for (int64_t c = 0; c < 4; ++c) {
                total += grad[static_cast<size_t>(n * 4 + c)];
            }
            check_close(total, 0.0, 1e-15, "gradient row sums to zero");
        }
    }
}

// 14. Zero upstream gives an exactly zero gradient; a negative upstream
// flips every sign.
void test_backward_upstream_scaling() {
    const std::vector<double> logits = {1.0, 2.0, 0.5, -1.0};
    const std::vector<int64_t> targets = {0, 1};
    const Forward forward = run_forward(logits, targets, 2, 2, kSum);
    const std::vector<double> zero =
        run_backward(forward.probabilities, targets, 0.0, 2, 2, kSum);
    for (size_t i = 0; i < zero.size(); ++i) {
        check_eq(zero[i], 0.0, "zero upstream gives a zero gradient");
    }
    const std::vector<double> positive =
        run_backward(forward.probabilities, targets, 2.0, 2, 2, kSum);
    const std::vector<double> negative =
        run_backward(forward.probabilities, targets, -2.0, 2, 2, kSum);
    for (size_t i = 0; i < positive.size(); ++i) {
        check_eq(negative[i], -positive[i], "negative upstream flips the sign");
    }
}

// 15. Determinism, and neither the probabilities nor the upstream is
// mutated. The destination is fully written.
void test_backward_determinism_and_non_mutation() {
    const std::vector<double> logits = {0.25, 1.5, -0.5, 2.0, 0.75, -1.25};
    const std::vector<int64_t> targets = {2, 1};
    const Forward forward = run_forward(logits, targets, 2, 3, kMean);
    const std::vector<double> original = forward.probabilities;
    std::vector<double> probabilities = forward.probabilities;
    const std::vector<double> first =
        run_backward(probabilities, targets, 1.0, 2, 3, kMean);
    const std::vector<double> second =
        run_backward(probabilities, targets, 1.0, 2, 3, kMean);
    for (size_t i = 0; i < first.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "backward determinism[%zu]", i);
        check_eq(first[i], second[i], label);
        std::snprintf(label, sizeof(label), "probabilities untouched[%zu]", i);
        check_eq(probabilities[i], original[i], label);
        std::snprintf(label, sizeof(label), "gradient written[%zu]", i);
        check(first[i] != kPoison, label);
    }
}

// 16. Central finite differences of the forward loss agree with the
// backward gradient — the strongest available cross-check that the two
// kernels describe the same function.
void test_backward_agrees_with_finite_differences() {
    std::vector<double> logits = {0.5, -1.0, 2.0, 1.25, 3.0, 0.0};
    const std::vector<int64_t> targets = {2, 0};
    for (int64_t reduction : {kSum, kMean}) {
        const Forward forward = run_forward(logits, targets, 2, 3, reduction);
        const std::vector<double> grad =
            run_backward(forward.probabilities, targets, 1.0, 2, 3, reduction);
        const double eps = 1e-6;
        for (size_t i = 0; i < logits.size(); ++i) {
            const double original = logits[i];
            logits[i] = original + eps;
            const double plus =
                reference_loss(logits, targets, 2, 3, reduction);
            logits[i] = original - eps;
            const double minus =
                reference_loss(logits, targets, 2, 3, reduction);
            logits[i] = original;
            const double numeric = (plus - minus) / (2.0 * eps);
            char label[128];
            std::snprintf(label, sizeof(label), "finite difference[%zu]", i);
            check_close(grad[i], numeric, 1e-7, label);
        }
    }
}

// ======================================================================
// Exceptional values (plain IEEE, no special-casing)
// ======================================================================

// 17. A NaN poisons its own row and only its own row.
void test_nan_row() {
    const std::vector<double> logits = {1.0, kNaN, 2.0,   1.0, 2.0, 3.0};
    const std::vector<int64_t> targets = {0, 2};
    const Forward got = run_forward(logits, targets, 2, 3, kSum);
    check(std::isnan(got.loss), "a NaN row makes the loss NaN");
    for (size_t i = 0; i < 3; ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "nan_row probability[%zu]", i);
        check(std::isnan(got.probabilities[i]), label);
    }
    const std::vector<double> clean = {1.0, 2.0, 3.0};
    const std::vector<double> want = reference_probabilities(clean, 1, 3);
    for (size_t i = 0; i < 3; ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "clean row unaffected[%zu]", i);
        check_eq(got.probabilities[3 + i], want[i], label);
    }
    expect_matches_reference(logits, targets, 2, 3, kSum, "nan_reference");
}

// 18. +inf makes its row NaN (inf - inf); an all -inf row is NaN; -inf
// beside finite values simply takes zero probability.
void test_infinities() {
    const std::vector<double> pos = {kPosInf, 1.0, 2.0};
    const std::vector<int64_t> one = {0};
    const Forward pos_result = run_forward(pos, one, 1, 3, kSum);
    check(std::isnan(pos_result.loss), "+inf row gives a NaN loss");
    for (size_t i = 0; i < 3; ++i) {
        check(std::isnan(pos_result.probabilities[i]), "+inf row is NaN");
    }

    const std::vector<double> neg = {kNegInf, 1.0, 2.0};
    const Forward neg_result = run_forward(neg, one, 1, 3, kSum);
    check_eq(neg_result.probabilities[0], 0.0,
             "-inf member gets zero probability");
    check(std::isfinite(neg_result.probabilities[1]) &&
              std::isfinite(neg_result.probabilities[2]),
          "the rest of a -inf row stays finite");
    check_close(neg_result.probabilities[1] + neg_result.probabilities[2], 1.0,
                1e-15, "the finite part of a -inf row still normalizes");
    check(neg_result.loss == kPosInf,
          "a -inf target logit gives an infinite loss");

    const std::vector<int64_t> finite_target = {2};
    const Forward finite_result = run_forward(neg, finite_target, 1, 3, kSum);
    check(std::isfinite(finite_result.loss),
          "a finite target beside -inf keeps a finite loss");

    const std::vector<double> all_neg = {kNegInf, kNegInf};
    const Forward all_neg_result = run_forward(all_neg, one, 1, 2, kSum);
    check(std::isnan(all_neg_result.loss), "an all -inf row gives NaN");
    for (size_t i = 0; i < 2; ++i) {
        check(std::isnan(all_neg_result.probabilities[i]),
              "an all -inf row is NaN");  // -inf - (-inf) == NaN
    }

    expect_matches_reference(pos, one, 1, 3, kSum, "pos_inf_reference");
    expect_matches_reference(neg, one, 1, 3, kSum, "neg_inf_reference");
    expect_matches_reference(all_neg, one, 1, 2, kSum, "all_neg_inf_reference");
}

// 19. A numerically exceptional but structurally valid call is NOT an
// ABI failure: the error slot stays TF_OK and both destinations are
// still fully written.
void test_exceptional_values_leave_status_ok() {
    const std::vector<double> logits = {kNaN, kPosInf, kNegInf};
    const std::vector<int64_t> targets = {1};
    std::vector<double> loss(1, kPoison);
    std::vector<double> probabilities(3, kPoison);
    check(run_checked_forward(logits, 0, targets.data(), 1, loss, probabilities,
                              1, 3, kMean) == TF_OK,
          "exceptional values leave the error slot clear");
    check(loss[0] != kPoison, "loss still written");
    for (size_t i = 0; i < 3; ++i) {
        check(probabilities[i] != kPoison, "probabilities still written");
    }
    // The backward over an exceptional probability row is a value too.
    std::vector<double> grad(3, kPoison);
    const std::vector<double> upstream = {1.0};
    check(run_checked_backward(probabilities, 0, targets.data(), 1, upstream, 0,
                               grad, 1, 3, kMean) == TF_OK,
          "exceptional backward leaves the error slot clear");
    for (size_t i = 0; i < 3; ++i) {
        check(grad[i] != kPoison, "gradient still written");
    }
}

// ======================================================================
// Forward validation at the trust boundary
// ======================================================================

void expect_forward_rejected(int status, const std::vector<double>& loss,
                             const std::vector<double>& probabilities,
                             const char* what) {
    char label[192];
    std::snprintf(label, sizeof(label), "%s rejected with TF_ERROR_INVALID",
                  what);
    check(status == TF_ERROR_INVALID, label);
    for (size_t i = 0; i < loss.size(); ++i) {
        std::snprintf(label, sizeof(label), "%s left loss[%zu] untouched", what,
                      i);
        check_eq(loss[i], kPoison, label);
    }
    for (size_t i = 0; i < probabilities.size(); ++i) {
        std::snprintf(label, sizeof(label),
                      "%s left probabilities[%zu] untouched", what, i);
        check_eq(probabilities[i], kPoison, label);
    }
    tf::clear_error();
}

// 20. Null handles and a null target pointer.
void test_forward_rejects_null_pointers() {
    std::vector<double> logits = {1.0, 2.0, 0.5, -1.0};
    std::vector<double> loss(1, kPoison);
    std::vector<double> probabilities(4, kPoison);
    const std::vector<int64_t> targets = {0, 1};
    tf::Storage in{logits.data(), 4};
    tf::Storage loss_storage{loss.data(), 1};
    tf::Storage probability_storage{probabilities.data(), 4};

    tf_core_cross_entropy_forward(nullptr, 0, targets.data(), 2, &loss_storage,
                                  &probability_storage, 2, 2, kMean);
    expect_forward_rejected(tf::last_error_code(), loss, probabilities,
                            "null logits handle");
    tf_core_cross_entropy_forward(&in, 0, targets.data(), 2, nullptr,
                                  &probability_storage, 2, 2, kMean);
    expect_forward_rejected(tf::last_error_code(), loss, probabilities,
                            "null loss handle");
    tf_core_cross_entropy_forward(&in, 0, targets.data(), 2, &loss_storage,
                                  nullptr, 2, 2, kMean);
    expect_forward_rejected(tf::last_error_code(), loss, probabilities,
                            "null probability handle");
    tf_core_cross_entropy_forward(&in, 0, nullptr, 2, &loss_storage,
                                  &probability_storage, 2, 2, kMean);
    expect_forward_rejected(tf::last_error_code(), loss, probabilities,
                            "null target pointer");
}

// 21. Malformed dimensions, offsets, target counts, and reduction codes.
void test_forward_rejects_malformed_arguments() {
    const std::vector<double> logits = {1.0, 2.0, 0.5, -1.0};
    const std::vector<int64_t> targets = {0, 1};
    std::vector<double> loss(1, kPoison);
    std::vector<double> probabilities(4, kPoison);

    check(run_checked_forward(logits, 0, targets.data(), 2, loss, probabilities,
                              2, 2, kMean) == TF_OK, "sane baseline");
    std::fill(loss.begin(), loss.end(), kPoison);
    std::fill(probabilities.begin(), probabilities.end(), kPoison);

    expect_forward_rejected(
        run_checked_forward(logits, -1, targets.data(), 2, loss, probabilities,
                            2, 2, kMean), loss, probabilities,
        "negative logits offset");
    expect_forward_rejected(
        run_checked_forward(logits, 0, targets.data(), 2, loss, probabilities,
                            0, 2, kMean), loss, probabilities,
        "batch_size == 0");
    expect_forward_rejected(
        run_checked_forward(logits, 0, targets.data(), 2, loss, probabilities,
                            -2, 2, kMean), loss, probabilities,
        "negative batch_size");
    expect_forward_rejected(
        run_checked_forward(logits, 0, targets.data(), 2, loss, probabilities,
                            2, 0, kMean), loss, probabilities,
        "num_classes == 0");
    expect_forward_rejected(
        run_checked_forward(logits, 0, targets.data(), 2, loss, probabilities,
                            2, -2, kMean), loss, probabilities,
        "negative num_classes");
    expect_forward_rejected(
        run_checked_forward(logits, 0, targets.data(), 1, loss, probabilities,
                            2, 2, kMean), loss, probabilities,
        "target count too small");
    expect_forward_rejected(
        run_checked_forward(logits, 0, targets.data(), 3, loss, probabilities,
                            2, 2, kMean), loss, probabilities,
        "target count too large");
    for (int64_t code : {int64_t(-1), int64_t(2), int64_t(99)}) {
        expect_forward_rejected(
            run_checked_forward(logits, 0, targets.data(), 2, loss,
                                probabilities, 2, 2, code), loss, probabilities,
            "invalid reduction code");
    }
}

// 22. Overflow, spans, and capacities.
void test_forward_rejects_bad_spans() {
    const std::vector<double> logits = {1.0, 2.0, 0.5, -1.0};
    const std::vector<int64_t> targets = {0, 1, 0, 1};
    std::vector<double> loss(1, kPoison);
    std::vector<double> probabilities(4, kPoison);

    const int64_t huge = INT64_MAX / 2 + 4;
    expect_forward_rejected(
        run_checked_forward(logits, 0, targets.data(), huge, loss,
                            probabilities, huge, huge, kMean), loss,
        probabilities, "batch * classes overflow");
    expect_forward_rejected(
        run_checked_forward(logits, INT64_MAX, targets.data(), 2, loss,
                            probabilities, 2, 2, kMean), loss, probabilities,
        "offset + numel overflow");
    // numel 6 > logits capacity 4.
    expect_forward_rejected(
        run_checked_forward(logits, 0, targets.data(), 3, loss, probabilities,
                            3, 2, kMean), loss, probabilities,
        "logits span too short");
    // The offset pushes the run past the end.
    expect_forward_rejected(
        run_checked_forward(logits, 2, targets.data(), 2, loss, probabilities,
                            2, 2, kMean), loss, probabilities,
        "offset + numel past the logits end");
    // A destination smaller than required.
    std::vector<double> small_probabilities(2, kPoison);
    expect_forward_rejected(
        run_checked_forward(logits, 0, targets.data(), 2, loss,
                            small_probabilities, 2, 2, kMean), loss,
        small_probabilities, "probability capacity too short");
    // A loss storage that cannot even hold one double.
    tf::Storage in{const_cast<double*>(logits.data()), 4};
    tf::Storage loss_storage{nullptr, 0};
    tf::Storage probability_storage{probabilities.data(), 4};
    tf_core_cross_entropy_forward(&in, 0, targets.data(), 2, &loss_storage,
                                  &probability_storage, 2, 2, kMean);
    expect_forward_rejected(tf::last_error_code(), loss, probabilities,
                            "loss capacity too short");
}

// 23. Out-of-range targets are caught in C++, whatever Python believed.
void test_forward_rejects_bad_targets() {
    const std::vector<double> logits = {1.0, 2.0, 0.5, -1.0};
    std::vector<double> loss(1, kPoison);
    std::vector<double> probabilities(4, kPoison);
    const std::vector<int64_t> negative = {0, -1};
    const std::vector<int64_t> too_large = {2, 0};
    const std::vector<int64_t> way_too_large = {0, 1000000};

    expect_forward_rejected(
        run_checked_forward(logits, 0, negative.data(), 2, loss, probabilities,
                            2, 2, kMean), loss, probabilities,
        "negative target");
    expect_forward_rejected(
        run_checked_forward(logits, 0, too_large.data(), 2, loss, probabilities,
                            2, 2, kMean), loss, probabilities,
        "target == num_classes");
    expect_forward_rejected(
        run_checked_forward(logits, 0, way_too_large.data(), 2, loss,
                            probabilities, 2, 2, kMean), loss, probabilities,
        "target far above num_classes");
}

// 24. An aliasing destination is rejected rather than silently computing
// a wrong loss.
void test_forward_rejects_aliasing() {
    std::vector<double> logits = {1.0, 2.0, 0.5, -1.0};
    const std::vector<int64_t> targets = {0, 1};
    std::vector<double> loss(1, kPoison);
    tf::Storage in{logits.data(), 4};
    tf::Storage loss_storage{loss.data(), 1};
    tf_core_cross_entropy_forward(&in, 0, targets.data(), 2, &loss_storage, &in,
                                  2, 2, kMean);
    check(tf::last_error_code() == TF_ERROR_INVALID,
          "probabilities aliasing the logits is rejected");
    check_eq(loss[0], kPoison, "the aliasing rejection wrote no loss");
    check_eq(logits[0], 1.0, "the aliasing rejection wrote no probability");
    tf::clear_error();
}

// 25. The guard clears the slot on entry, so neither a previous
// rejection nor an unrelated stale error contaminates a later call.
void test_forward_guard_clears_previous_error() {
    const std::vector<double> logits = {1.0, 2.0};
    const std::vector<int64_t> targets = {1};
    std::vector<double> loss(1, kPoison);
    std::vector<double> probabilities(2, kPoison);
    expect_forward_rejected(
        run_checked_forward(logits, 0, targets.data(), 1, loss, probabilities,
                            1, 2, 42), loss, probabilities, "priming failure");
    tf::set_error(TF_ERROR_RUNTIME, "stale error from somewhere else");
    check(run_checked_forward(logits, 0, targets.data(), 1, loss, probabilities,
                              1, 2, kSum) == TF_OK,
          "a valid call clears the stale error");
    const std::vector<double> want = {1.0, 2.0};
    check_eq(loss[0], reference_loss(want, targets, 1, 2, kSum),
             "post-failure call still computes");
    check_close(probabilities[0] + probabilities[1], 1.0, 1e-15,
                "post-failure probabilities normalized");
}

// ======================================================================
// Backward validation at the trust boundary
// ======================================================================

void expect_backward_rejected(int status, const std::vector<double>& grad,
                              const char* what) {
    char label[192];
    std::snprintf(label, sizeof(label), "%s rejected with TF_ERROR_INVALID",
                  what);
    check(status == TF_ERROR_INVALID, label);
    for (size_t i = 0; i < grad.size(); ++i) {
        std::snprintf(label, sizeof(label), "%s left grad[%zu] untouched", what,
                      i);
        check_eq(grad[i], kPoison, label);
    }
    tf::clear_error();
}

// 26. Null handles and a null target pointer.
void test_backward_rejects_null_pointers() {
    std::vector<double> probabilities = {0.25, 0.75, 0.5, 0.5};
    std::vector<double> upstream = {1.0};
    std::vector<double> grad(4, kPoison);
    const std::vector<int64_t> targets = {0, 1};
    tf::Storage probability_storage{probabilities.data(), 4};
    tf::Storage upstream_storage{upstream.data(), 1};
    tf::Storage grad_storage{grad.data(), 4};

    tf_core_cross_entropy_backward(nullptr, 0, targets.data(), 2,
                                   &upstream_storage, 0, &grad_storage, 2, 2,
                                   kMean);
    expect_backward_rejected(tf::last_error_code(), grad,
                             "null probability handle");
    tf_core_cross_entropy_backward(&probability_storage, 0, targets.data(), 2,
                                   nullptr, 0, &grad_storage, 2, 2, kMean);
    expect_backward_rejected(tf::last_error_code(), grad, "null upstream handle");
    tf_core_cross_entropy_backward(&probability_storage, 0, targets.data(), 2,
                                   &upstream_storage, 0, nullptr, 2, 2, kMean);
    expect_backward_rejected(tf::last_error_code(), grad, "null gradient handle");
    tf_core_cross_entropy_backward(&probability_storage, 0, nullptr, 2,
                                   &upstream_storage, 0, &grad_storage, 2, 2,
                                   kMean);
    expect_backward_rejected(tf::last_error_code(), grad, "null target pointer");
}

// 27. Malformed dimensions, offsets, counts, reduction codes, spans,
// capacities, and targets.
void test_backward_rejects_malformed_arguments() {
    const std::vector<double> probabilities = {0.25, 0.75, 0.5, 0.5};
    const std::vector<int64_t> targets = {0, 1};
    const std::vector<double> upstream = {1.0};
    std::vector<double> grad(4, kPoison);

    check(run_checked_backward(probabilities, 0, targets.data(), 2, upstream, 0,
                               grad, 2, 2, kSum) == TF_OK, "sane baseline");
    std::fill(grad.begin(), grad.end(), kPoison);

    expect_backward_rejected(
        run_checked_backward(probabilities, -1, targets.data(), 2, upstream, 0,
                             grad, 2, 2, kSum), grad,
        "negative probability offset");
    expect_backward_rejected(
        run_checked_backward(probabilities, 0, targets.data(), 2, upstream, -1,
                             grad, 2, 2, kSum), grad, "negative upstream offset");
    expect_backward_rejected(
        run_checked_backward(probabilities, 0, targets.data(), 2, upstream, 0,
                             grad, 0, 2, kSum), grad, "batch_size == 0");
    expect_backward_rejected(
        run_checked_backward(probabilities, 0, targets.data(), 2, upstream, 0,
                             grad, -1, 2, kSum), grad, "negative batch_size");
    expect_backward_rejected(
        run_checked_backward(probabilities, 0, targets.data(), 2, upstream, 0,
                             grad, 2, 0, kSum), grad, "num_classes == 0");
    expect_backward_rejected(
        run_checked_backward(probabilities, 0, targets.data(), 2, upstream, 0,
                             grad, 2, -3, kSum), grad, "negative num_classes");
    expect_backward_rejected(
        run_checked_backward(probabilities, 0, targets.data(), 1, upstream, 0,
                             grad, 2, 2, kSum), grad, "target count mismatch");
    expect_backward_rejected(
        run_checked_backward(probabilities, 0, targets.data(), 2, upstream, 0,
                             grad, 2, 2, 7), grad, "invalid reduction code");
    const int64_t huge = INT64_MAX / 2 + 4;
    expect_backward_rejected(
        run_checked_backward(probabilities, 0, targets.data(), huge, upstream, 0,
                             grad, huge, huge, kSum), grad, "product overflow");
    expect_backward_rejected(
        run_checked_backward(probabilities, 0, targets.data(), 3, upstream, 0,
                             grad, 3, 2, kSum), grad,
        "probability span too short");
    expect_backward_rejected(
        run_checked_backward(probabilities, 2, targets.data(), 2, upstream, 0,
                             grad, 2, 2, kSum), grad,
        "probability offset past the end");
    expect_backward_rejected(
        run_checked_backward(probabilities, 0, targets.data(), 2, upstream, 1,
                             grad, 2, 2, kSum), grad, "upstream span too short");
    std::vector<double> small_grad(2, kPoison);
    expect_backward_rejected(
        run_checked_backward(probabilities, 0, targets.data(), 2, upstream, 0,
                             small_grad, 2, 2, kSum), small_grad,
        "gradient capacity too short");
    const std::vector<int64_t> negative = {-1, 0};
    const std::vector<int64_t> too_large = {0, 2};
    expect_backward_rejected(
        run_checked_backward(probabilities, 0, negative.data(), 2, upstream, 0,
                             grad, 2, 2, kSum), grad, "negative target");
    expect_backward_rejected(
        run_checked_backward(probabilities, 0, too_large.data(), 2, upstream, 0,
                             grad, 2, 2, kSum), grad, "out-of-range target");

    // An aliasing gradient destination is rejected too.
    tf::Storage probability_storage{const_cast<double*>(probabilities.data()), 4};
    tf::Storage upstream_storage{const_cast<double*>(upstream.data()), 1};
    tf_core_cross_entropy_backward(&probability_storage, 0, targets.data(), 2,
                                   &upstream_storage, 0, &probability_storage,
                                   2, 2, kSum);
    check(tf::last_error_code() == TF_ERROR_INVALID,
          "a gradient aliasing the probabilities is rejected");
    tf::clear_error();
}

// 28. A stale error does not survive the next valid backward call, and a
// nonzero upstream offset is honored.
void test_backward_guard_clears_previous_error() {
    const std::vector<double> probabilities = {0.25, 0.75};
    const std::vector<int64_t> targets = {1};
    const std::vector<double> upstream = {99.0, 2.0};
    std::vector<double> grad(2, kPoison);
    expect_backward_rejected(
        run_checked_backward(probabilities, 0, targets.data(), 1, upstream, 0,
                             grad, 1, 2, 55), grad, "priming failure");
    tf::set_error(TF_ERROR_RUNTIME, "stale error from somewhere else");
    check(run_checked_backward(probabilities, 0, targets.data(), 1, upstream, 1,
                               grad, 1, 2, kSum) == TF_OK,
          "a valid call clears the stale error");
    check_eq(grad[0], 2.0 * 0.25, "post-failure backward still computes");
    check_eq(grad[1], 2.0 * (0.75 - 1.0), "upstream offset was honored");
}

}  // namespace

int main() {
    test_batch_size_one_two_classes();
    test_multi_class_batch_both_reductions();
    test_equal_logits();
    test_single_class();
    test_large_common_offsets();
    test_additive_shift_invariance();
    test_stable_loss_beats_the_naive_form();
    test_determinism_and_non_mutation();
    test_checked_nonzero_offset();
    test_backward_matches_the_formula();
    test_backward_rows_sum_to_zero();
    test_backward_upstream_scaling();
    test_backward_determinism_and_non_mutation();
    test_backward_agrees_with_finite_differences();
    test_nan_row();
    test_infinities();
    test_exceptional_values_leave_status_ok();
    test_forward_rejects_null_pointers();
    test_forward_rejects_malformed_arguments();
    test_forward_rejects_bad_spans();
    test_forward_rejects_bad_targets();
    test_forward_rejects_aliasing();
    test_forward_guard_clears_previous_error();
    test_backward_rejects_null_pointers();
    test_backward_rejects_malformed_arguments();
    test_backward_guard_clears_previous_error();

    if (g_failures == 0) {
        std::printf("OK: all cross_entropy tests passed\n");
        return 0;
    }
    std::printf("FAILED: %d cross_entropy check(s)\n", g_failures);
    return 1;
}
