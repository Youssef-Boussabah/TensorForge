// TensorForge experimental C++ backend — the first kernel.
//
// Deliberately tiny: a plain C-ABI function over raw float64 buffers,
// so Python can load it with ctypes. No Python C-API, no pybind11,
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
