// TensorForge experimental C++ backend — the kernels.
//
// Deliberately tiny: plain C-ABI functions over raw float64 buffers,
// so Python can load them with ctypes. No Python C-API, no pybind11,
// no NumPy headers — the wrapper on the Python side handles arrays,
// shapes, and validation; this file only does the arithmetic.

#include <cstdint>

#ifdef _WIN32
#define TF_EXPORT extern "C" __declspec(dllexport)
#else
#define TF_EXPORT extern "C"
#endif

TF_EXPORT void tf_elementwise_add(
    const double* a, const double* b, double* out, int64_t n
) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

TF_EXPORT void tf_elementwise_subtract(
    const double* a, const double* b, double* out, int64_t n
) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
}

TF_EXPORT void tf_elementwise_multiply(
    const double* a, const double* b, double* out, int64_t n
) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

// IEEE float64 division: x/0 gives +-inf and 0/0 gives NaN, the same
// values NumPy produces (NumPy additionally warns; this kernel does not).
TF_EXPORT void tf_elementwise_divide(
    const double* a, const double* b, double* out, int64_t n
) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] / b[i];
    }
}

TF_EXPORT void tf_relu(const double* a, double* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] > 0.0 ? a[i] : 0.0;
    }
}

// Naive matrix multiplication: out (m x p) = a (m x n) @ b (n x p),
// all row-major. The textbook triple loop — deliberately unoptimized
// (no blocking, no SIMD, no BLAS); this stays as the reference kernel
// that tf_matmul_tiled is measured against.
TF_EXPORT void tf_matmul(
    const double* a, const double* b, double* out,
    int64_t m, int64_t n, int64_t p
) {
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (int64_t k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * p + j];
            }
            out[i * p + j] = sum;
        }
    }
}

// Tiled (blocked) matrix multiplication — the classic cache-locality
// optimization. The naive kernel streams entire rows/columns for every
// output value, so for matrices bigger than cache the same data is
// reloaded from memory over and over. Tiling processes small
// block x block sub-matrices that stay resident in cache while they
// are reused. Within a block, the i-k-j loop order walks b and out
// row-by-row (sequential memory), instead of striding down b's
// columns like the naive j-inner order does.
//
// Still single-threaded, no SIMD, no BLAS: an experiment in one idea,
// not a performance product.
TF_EXPORT void tf_matmul_tiled(
    const double* a, const double* b, double* out,
    int64_t m, int64_t n, int64_t p, int64_t block
) {
    for (int64_t i = 0; i < m * p; ++i) {
        out[i] = 0.0;  // accumulate into out, so it must start at zero
    }
    for (int64_t i0 = 0; i0 < m; i0 += block) {
        const int64_t i_end = i0 + block < m ? i0 + block : m;
        for (int64_t k0 = 0; k0 < n; k0 += block) {
            const int64_t k_end = k0 + block < n ? k0 + block : n;
            for (int64_t j0 = 0; j0 < p; j0 += block) {
                const int64_t j_end = j0 + block < p ? j0 + block : p;
                // One block x block tile; the min-clamped ends handle
                // dimensions that are not multiples of the block size.
                for (int64_t i = i0; i < i_end; ++i) {
                    for (int64_t k = k0; k < k_end; ++k) {
                        const double a_ik = a[i * n + k];
                        for (int64_t j = j0; j < j_end; ++j) {
                            out[i * p + j] += a_ik * b[k * p + j];
                        }
                    }
                }
            }
        }
    }
}
