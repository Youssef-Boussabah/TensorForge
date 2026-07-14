// Native storage: a C++-owned float64 buffer behind an opaque handle,
// plus the contiguous-materialization kernel that reads a strided view
// out of it. Python holds the handle only, moves data in and out through
// copy_from/copy_to, and destroys the handle exactly once (the wrapper's
// close() guarantees that). This is the storage half of the tensor
// runtime; the shape/stride metadata layer lives on the Python side.

#include <memory>

#include "tf_internal.h"

using tf::Storage;
using tf::as_storage;

// Returns an opaque handle, or null (with the thread-local error set) if
// allocation fails. The buffer is zero-initialized for predictable
// behavior. Allocation is RAII-ordered so the failure scenario the ABI
// contract calls out — data allocated, then metadata allocation throws,
// leaking the data — cannot happen: the unique_ptr owns the buffer until
// the Storage node has successfully adopted it.
TF_EXPORT void* tf_storage_create(int64_t size) {
    TF_GUARD_BEGIN
    if (size <= 0) {
        tf::set_error(TF_ERROR_INVALID, "storage size must be positive");
        return nullptr;
    }
    if (tf::should_fail_alloc()) {  // test-only injected failure (one point)
        throw std::bad_alloc();
    }
    std::unique_ptr<double[]> data(
        new (std::nothrow) double[static_cast<size_t>(size)]());
    if (!data) {
        tf::set_error(TF_ERROR_ALLOC, "could not allocate native storage");
        return nullptr;
    }
    // If this metadata allocation fails for real, the unique_ptr frees the
    // buffer — the "data allocated, then metadata throws, buffer leaks"
    // scenario the ABI contract calls out cannot happen.
    Storage* storage = new (std::nothrow) Storage{data.get(), size};
    if (storage == nullptr) {
        tf::set_error(TF_ERROR_ALLOC, "could not allocate native storage");
        return nullptr;  // data freed by unique_ptr — no leak
    }
    data.release();  // ownership transferred to the Storage node
    return storage;
    TF_GUARD_END(nullptr)
}

// Safe to call with null (does nothing), and must be called exactly once
// per successful create — the Python wrapper guarantees that.
TF_EXPORT void tf_storage_destroy(void* handle) {
    if (handle == nullptr) {
        return;
    }
    Storage* storage = as_storage(handle);
    delete[] storage->data;
    delete storage;
}

TF_EXPORT int64_t tf_storage_size(const void* handle) {
    return as_storage(handle)->size;
}

TF_EXPORT void tf_storage_fill(void* handle, double value) {
    Storage* storage = as_storage(handle);
    for (int64_t i = 0; i < storage->size; ++i) {
        storage->data[i] = value;
    }
}

// Multiply every element by a scalar factor, in place — a small storage
// primitive alongside fill, used by the mean reduction to scale a freshly
// summed output by 1/count in float64.
TF_EXPORT void tf_storage_scale(void* handle, double factor) {
    Storage* storage = as_storage(handle);
    for (int64_t i = 0; i < storage->size; ++i) {
        storage->data[i] *= factor;
    }
}

TF_EXPORT void tf_storage_copy_from(void* handle, const double* src) {
    Storage* storage = as_storage(handle);
    for (int64_t i = 0; i < storage->size; ++i) {
        storage->data[i] = src[i];
    }
}

TF_EXPORT void tf_storage_copy_to(const void* handle, double* dst) {
    const Storage* storage = as_storage(handle);
    for (int64_t i = 0; i < storage->size; ++i) {
        dst[i] = storage->data[i];
    }
}

// Materialize a strided view of the storage into a contiguous output
// buffer — the canonical tensor-runtime loop. The logical indices are
// walked like an odometer (last dimension fastest, i.e. row-major output
// order) while the source position is updated incrementally: stepping
// dimension d moves by strides[d], and when a dimension wraps its full
// extent is subtracted before the next dimension steps. Bounds are
// validated on the Python side before this is called.
TF_EXPORT void tf_storage_materialize(
    const void* handle, double* dst,
    const int64_t* shape, const int64_t* strides,
    int64_t offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    const double* src = as_storage(handle)->data;
    if (ndim == 0) {  // scalar view: one element at the offset
        dst[0] = src[offset];
        return;
    }
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        total *= shape[d];
    }
    std::vector<int64_t> counter = tf::make_counter(ndim);
    int64_t src_pos = offset;
    for (int64_t out = 0; out < total; ++out) {
        dst[out] = src[src_pos];
        for (int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[d];
            src_pos += strides[d];
            if (counter[d] < shape[d]) {
                break;  // this dimension advanced; done
            }
            counter[d] = 0;  // wrap and carry into the next dimension
            src_pos -= shape[d] * strides[d];
        }
    }
    TF_GUARD_END_VOID()
}
