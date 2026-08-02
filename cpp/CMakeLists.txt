# Canonical build for the experimental TensorForge C++ backend.
#
# Produces one shared library, ``_tensorforge_cpp``, in the location the
# Python ctypes loader expects (src/tensorforge/backends/). Plain C-ABI
# kernels — no Python headers, no pybind11 — so this is fully independent
# of the uv_build package build. cpp/build.py wraps this file; developers
# with cmake on PATH get it automatically.
#
#   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build cpp/build
#
# Sanitizer builds (Linux/macOS with Clang or GCC; not MSVC):
#   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug -DTF_SANITIZE=address
#   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug -DTF_SANITIZE=undefined
#   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug -DTF_SANITIZE=address,undefined

cmake_minimum_required(VERSION 3.15)
project(tensorforge_cpp LANGUAGES CXX)

# C++17, enforced, no compiler extensions — the ABI contract, RAII
# storage, and guard macros assume a conforming C++17 toolchain.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release when a single-config generator leaves the build type
# empty, so a bare `cmake -S cpp -B build` still optimizes.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Where the shared library lands. cpp/build.py passes the absolute path to
# src/tensorforge/backends/; a standalone cmake invocation defaults there
# too, computed relative to this file so it works from any build dir.
set(TF_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/tensorforge/backends"
    CACHE PATH "Directory to place the compiled backend library")

option(TF_SANITIZE "Comma-separated sanitizers (address, undefined); empty to disable" "")

file(GLOB TF_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(_tensorforge_cpp SHARED ${TF_SOURCES})
target_include_directories(_tensorforge_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Hidden default visibility: only the TF_EXPORT-marked functions are
# exported, keeping the C ABI narrow (matches -fvisibility=hidden in the
# direct build). Ignored by MSVC, which exports nothing without dllexport.
set_target_properties(_tensorforge_cpp PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  PREFIX ""                       # no leading "lib" — the loader expects _tensorforge_cpp.*
  LIBRARY_OUTPUT_DIRECTORY "${TF_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${TF_OUTPUT_DIR}"  # Windows places DLLs here
)

# Keep the output name stable across configs so multi-config generators
# (MSVC) never append a per-config suffix the loader would miss.
foreach(config "" DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  if(config STREQUAL "")
    set(suffix "")
  else()
    set(suffix "_${config}")
  endif()
  set_target_properties(_tensorforge_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY${suffix} "${TF_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY${suffix} "${TF_OUTPUT_DIR}")
endforeach()

if(NOT MSVC)
  target_compile_options(_tensorforge_cpp PRIVATE -Wall -Wextra)
endif()

# Optional sanitizers — Clang/GCC only, and never on MSVC (which supports
# only a different /fsanitize=address form and no UBSan). Applied to both
# compile and link so the runtime is pulled in.
if(TF_SANITIZE AND NOT MSVC)
  string(REPLACE "," ";" _tf_sanitizers "${TF_SANITIZE}")
  set(_tf_sanitize_flags "")
  foreach(san ${_tf_sanitizers})
    if(san STREQUAL "address")
      list(APPEND _tf_sanitize_flags -fsanitize=address -fno-omit-frame-pointer)
    elseif(san STREQUAL "undefined")
      list(APPEND _tf_sanitize_flags -fsanitize=undefined -fno-omit-frame-pointer)
    else()
      message(FATAL_ERROR "TF_SANITIZE: unknown sanitizer '${san}' (use address and/or undefined)")
    endif()
  endforeach()
  target_compile_options(_tensorforge_cpp PRIVATE ${_tf_sanitize_flags})
  target_link_options(_tensorforge_cpp PRIVATE ${_tf_sanitize_flags})
  message(STATUS "TensorForge backend: sanitizers enabled -> ${TF_SANITIZE}")
elseif(TF_SANITIZE AND MSVC)
  message(WARNING "TF_SANITIZE is not supported with MSVC; building without sanitizers")
endif()

# ---------------------------------------------------------------------------
# Optional native C++ tests (CTest). Off by default, so the ordinary package
# build and cpp/build.py (which never sets this option) are unaffected;
# enable with -DTF_BUILD_TESTS=ON. Each test executable compiles the
# internal kernel source(s) it exercises *directly* — the shared library
# exports only the TF_EXPORT symbols, and these internal compute functions
# are intentionally hidden, so they are not reachable through the library.
# ---------------------------------------------------------------------------
option(TF_BUILD_TESTS "Build the native C++ test executables (CTest)" OFF)

if(TF_BUILD_TESTS)
  enable_testing()

  # Conv2d forward (D2): links the internal kernel source directly. Since
  # D3 added the exported tf_core_conv2d_forward wrapper into conv2d.cpp,
  # that translation unit now references the thread-local error slot
  # (tf::set_error / tf::clear_error via TF_GUARD), so error.cpp is compiled
  # in too; the D2 test still exercises only the internal kernel.
  add_executable(test_conv2d_forward
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_conv2d_forward.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/conv2d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_conv2d_forward PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_conv2d_forward PRIVATE -Wall -Wextra)
  endif()
  # Honor the same optional sanitizers as the library (Clang/GCC only).
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_conv2d_forward PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_conv2d_forward PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME conv2d_forward COMMAND test_conv2d_forward)

  # Conv2d input-gradient (D4): links the same internal kernel source
  # (conv2d.cpp) directly. It compiles error.cpp too for the identical
  # reason as the forward target — conv2d.cpp carries the D3 guarded
  # wrapper, which references the thread-local error slot — while the test
  # itself exercises only the internal tf::conv2d_input_backward_contiguous
  # kernel (and the internal forward kernel for its finite-difference
  # objective).
  add_executable(test_conv2d_input_backward
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_conv2d_input_backward.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/conv2d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_conv2d_input_backward PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_conv2d_input_backward PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_conv2d_input_backward PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_conv2d_input_backward PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME conv2d_input_backward COMMAND test_conv2d_input_backward)

  # Conv2d weight-gradient (D5): links the same internal kernel source
  # (conv2d.cpp) directly, plus error.cpp for the identical reason as the
  # other backward target (conv2d.cpp carries the D3 guarded wrapper). The
  # test exercises only the internal tf::conv2d_weight_backward_contiguous
  # kernel (and the internal forward kernel for its finite-difference and
  # padded-reference oracles).
  add_executable(test_conv2d_weight_backward
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_conv2d_weight_backward.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/conv2d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_conv2d_weight_backward PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_conv2d_weight_backward PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_conv2d_weight_backward PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_conv2d_weight_backward PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME conv2d_weight_backward COMMAND test_conv2d_weight_backward)

  # MaxPool2d forward + winner buffer (D8): links the internal pooling
  # kernel source (pooling.cpp) directly, plus error.cpp because that
  # translation unit also carries the exported, guarded
  # tf_core_maxpool2d_forward wrapper (which references the thread-local
  # error slot). The test itself exercises only the internal
  # tf::maxpool2d_forward_contiguous kernel — output values and saved
  # winner indices together.
  add_executable(test_maxpool2d_forward
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_maxpool2d_forward.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/pooling.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_maxpool2d_forward PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_maxpool2d_forward PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_maxpool2d_forward PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_maxpool2d_forward PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME maxpool2d_forward COMMAND test_maxpool2d_forward)

  # MaxPool2d backward (D9): links the same internal pooling source
  # (pooling.cpp) plus error.cpp. Unlike the forward target this test
  # exercises *both* layers of that translation unit — the internal
  # tf::maxpool2d_backward_contiguous scatter kernel (with valid winners
  # only) and the exported guarded tf_core_maxpool2d_backward wrapper,
  # whose winner-value validation it drives through plain tf::Storage
  # handles and the thread-local error slot.
  add_executable(test_maxpool2d_backward
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_maxpool2d_backward.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/pooling.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_maxpool2d_backward PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_maxpool2d_backward PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_maxpool2d_backward PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_maxpool2d_backward PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME maxpool2d_backward COMMAND test_maxpool2d_backward)

  # Native exponential (Phase E, E1): links the elementwise source
  # (elementwise.cpp) plus error.cpp, since the exports under test are
  # guarded and reference the thread-local error slot. Unlike the conv2d
  # and pooling targets this test drives the *exported* C ABI pair
  # (tf_core_exp / tf_core_exp_contiguous) rather than a hidden tf::
  # kernel — the exponential's compute is file-local to that translation
  # unit, and the exports are where E1's own argument validation lives.
  add_executable(test_exp
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_exp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/elementwise.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_exp PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_exp PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_exp PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_exp PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME exp COMMAND test_exp)

  # Native logarithm (Phase E, E2): same target shape as the E1 exp test —
  # it links elementwise.cpp plus error.cpp and drives the *exported*
  # guarded pair (tf_core_log / tf_core_log_contiguous), since the
  # logarithm's compute is file-local to that translation unit. It also
  # regression-covers the validators E2 shares with exp.
  add_executable(test_log
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_log.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/elementwise.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_log PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_log PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_log PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_log PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME log COMMAND test_log)

  # Fused softmax (Phase E, E3): links the new classification source unit
  # plus error.cpp (the guarded export references the thread-local error
  # slot). Like the pooling tests it drives *both* layers — the internal
  # tf::softmax_forward_contiguous kernel for the numerics and the
  # exported tf_core_softmax_forward wrapper for the validation matrix.
  add_executable(test_softmax
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_softmax.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/classification.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_softmax PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_softmax PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_softmax PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_softmax PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME softmax COMMAND test_softmax)

  # Fused log-softmax (Phase E, E4): links the same classification source
  # unit plus error.cpp. Like the softmax target it drives *both* layers —
  # the internal tf::log_softmax_forward_contiguous kernel for the fused
  # maximum-shift / log-sum-exp numerics and the exported
  # tf_core_log_softmax_forward wrapper for the validation matrix. It also
  # re-drives softmax's rejection matrix as regression coverage for the
  # file-local validator E4 factored out of the two exports.
  add_executable(test_log_softmax
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_log_softmax.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/classification.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_log_softmax PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_log_softmax PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_log_softmax PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_log_softmax PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME log_softmax COMMAND test_log_softmax)

  # Fused cross-entropy (Phase E, E5): links the same classification
  # source unit plus error.cpp. Like the softmax and log-softmax targets
  # it drives *both* layers — the internal
  # tf::cross_entropy_forward_contiguous /
  # tf::cross_entropy_backward_contiguous kernels for the fused
  # maximum-shift / log-sum-exp numerics and the saved-probability
  # gradient, and the exported tf_core_cross_entropy_forward /
  # tf_core_cross_entropy_backward wrappers for the validation matrix,
  # including the C++-side revalidation of every target index.
  add_executable(test_cross_entropy
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cross_entropy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/classification.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_cross_entropy PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_cross_entropy PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_cross_entropy PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_cross_entropy PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME cross_entropy COMMAND test_cross_entropy)

  # Stateless random derivation + Dropout forward (Phase G, G2): links the
  # new random source unit plus error.cpp (the guarded export references
  # the thread-local error slot). Like the pooling and classification
  # targets it drives *both* layers — the internal tf::splitmix64_mix /
  # tf::dropout_stream_key / tf::dropout_element_bits / tf::dropout_uniform
  # derivation and the tf::dropout_forward_contiguous kernel against
  # committed known-answer vectors, and the exported
  # tf_core_dropout_forward wrapper for the validation matrix (including
  # the probability boundary and the destination-aliasing rule).
  add_executable(test_dropout_forward
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dropout_forward.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/random.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_dropout_forward PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_dropout_forward PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_dropout_forward PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_dropout_forward PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME dropout_forward COMMAND test_dropout_forward)

  # Storage creation contract (Phase H, H1): links the storage source
  # directly plus error.cpp, which owns the thread-local error slot the
  # guarded exports feed and the pre-existing allocation fault-injection
  # hook. It drives the *exported* layer only — tf_storage_create,
  # tf_storage_create_uninitialized, and their shared validation,
  # allocation-failure, error, ownership, destruction, and live-handle
  # behavior — and runs every shared check against **both** constructors
  # so the two cannot drift apart. Whether a given kernel writes every
  # destination element is a per-kernel property proved from Python,
  # where the poison wraps the private allocation helper around a real
  # Core operation; no poison-control export exists in this library.
  add_executable(test_storage_allocation
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_storage_allocation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/storage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_storage_allocation PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_storage_allocation PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_storage_allocation PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_storage_allocation PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME storage_allocation COMMAND test_storage_allocation)

  # The dtype model and dtype-tagged storage (Phase I, milestone I1).
  #
  # This target links **every** kernel translation unit, which none of the
  # others does, and it needs to: a third of what it proves is that a
  # float32 handle is rejected by an operation that has not been
  # dtype-generalized yet, and each compute unit has its own validation
  # front end. Checking one of them would say nothing about the rest, so
  # the test drives a representative export from each — elementwise,
  # matmul, reduction, classification, pooling, conv2d, random — beside the
  # storage primitives themselves.
  #
  # The other two thirds are the dtype model (frozen ABI codes, the one
  # item-size authority, canonical names, total conversion) and the two
  # typed creators, including every rejection path and the byte-level proof
  # that zero-initialized storage is +0.0 at both widths.
  add_executable(test_dtype_storage
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dtype_storage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/storage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/elementwise.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/matmul.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/reduction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/classification.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/pooling.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/conv2d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/random.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_dtype_storage PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_dtype_storage PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_dtype_storage PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_dtype_storage PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME dtype_storage COMMAND test_dtype_storage)

  # Matmul loop order and dispatch (Phase H, H2): links the matmul source
  # directly, plus storage.cpp (the exported-wrapper check drives real
  # tf::Storage handles) and error.cpp (matmul.cpp carries the guarded
  # tf_core_matmul export, which references the thread-local error slot).
  # Like the D8/D9 pooling targets it drives *both* layers — the internal
  # tf::matmul_generic_strided / tf::matmul_row_sweep kernels and the
  # tf::matmul_prefers_row_sweep predicate, which the shared library does
  # not export and which are reachable only by compiling the source in.
  # That is what proves which path a layout takes without adding any
  # "which kernel ran" control to the shipped ABI.
  add_executable(test_matmul
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_matmul.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/matmul.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/storage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_matmul PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_matmul PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_matmul PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_matmul PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME matmul COMMAND test_matmul)

  # Strided-to-contiguous gather and its H5 traversal dispatch: compiles
  # elementwise.cpp directly so it can reach the hidden
  # tf::copy_prefers_contiguous predicate alongside the exported
  # tf_core_contiguous_copy wrapper it lives inside. storage.cpp supplies
  # the Storage handles the ABI takes, and error.cpp the thread-local
  # error slot the guarded wrapper records into.
  add_executable(test_contiguous_copy
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_contiguous_copy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/elementwise.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/storage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_contiguous_copy PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_contiguous_copy PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_contiguous_copy PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_contiguous_copy PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME contiguous_copy COMMAND test_contiguous_copy)

  # Typed array transfer, materialization, and the identity copy (Phase I,
  # milestone I2). Compiles storage.cpp and elementwise.cpp directly so it
  # reaches the hidden tf::copy_prefers_contiguous predicate beside the four
  # exports the milestone generalizes, and error.cpp for the thread-local
  # slot the guarded wrappers record into.
  #
  # Its subject is deliberately narrow and complementary to test_dtype_storage:
  # that target proves float32 is *allocatable and otherwise rejected*, this
  # one proves the exact set of boundaries float32 may now cross, at both
  # dtypes, by raw IEEE-754 bit comparison rather than by tolerance.
  add_executable(test_typed_transfer
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_typed_transfer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/storage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/elementwise.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_typed_transfer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_typed_transfer PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_typed_transfer PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_typed_transfer PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME typed_transfer COMMAND test_typed_transfer)

  # Sum reduction and its H6 traversal dispatch: compiles reduction.cpp
  # directly so it can reach the hidden tf::reduce_prefers_contiguous_blocks
  # predicate and BOTH hidden traversals alongside the exported tf_core_sum
  # wrapper they live inside. storage.cpp supplies the Storage handles the
  # ABI takes, and error.cpp the thread-local error slot the guarded wrapper
  # records into.
  add_executable(test_sum_reduction
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sum_reduction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/reduction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/storage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_sum_reduction PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_sum_reduction PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_sum_reduction PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_sum_reduction PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME sum_reduction COMMAND test_sum_reduction)

  # Elementwise traversal and its H8 collapsed plan: compiles elementwise.cpp
  # directly so it can reach the hidden tf::build_unary_plan /
  # tf::build_binary_plan builders and the templated traversals alongside the
  # exported wrappers they live inside. storage.cpp supplies the Storage
  # handles the ABI takes, and error.cpp the thread-local error slot the
  # guarded wrappers record into.
  add_executable(test_elementwise_traversal
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_elementwise_traversal.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/elementwise.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/storage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_elementwise_traversal PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_elementwise_traversal PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_elementwise_traversal PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_elementwise_traversal PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME elementwise_traversal COMMAND test_elementwise_traversal)

  # Dtype-general elementwise, broadcast, and unary execution (Phase I,
  # milestone I3). Compiles elementwise.cpp directly so it reaches the hidden
  # tf::build_unary_plan / tf::build_binary_plan builders and the templated
  # traversals beside the exports they live inside; storage.cpp supplies the
  # typed Storage handles, and error.cpp the thread-local slot the guarded
  # wrappers record into.
  #
  # Deliberately complementary rather than a superset: test_elementwise_traversal
  # keeps H8's float64 NaN sweep, test_dtype_storage proves float32 is
  # allocatable and otherwise rejected, and test_typed_transfer proves the
  # boundaries I2 opened. This one proves the set I3 opened — the arithmetic
  # and unary family at both dtypes, through all three traversal tiers, with
  # mixed dtype rejected before any destination is touched.
  add_executable(test_dtype_elementwise
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dtype_elementwise.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/elementwise.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/storage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_dtype_elementwise PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_dtype_elementwise PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_dtype_elementwise PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_dtype_elementwise PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME dtype_elementwise COMMAND test_dtype_elementwise)

  # Dtype-general reductions, matmul, and the narrow-backward scatter (Phase
  # I, milestone I4). Compiles reduction.cpp and matmul.cpp directly so it
  # reaches the hidden tf::reduce_prefers_contiguous_blocks and
  # tf::matmul_prefers_row_sweep predicates and BOTH traversals of each
  # family beside the exports they live inside; storage.cpp supplies the
  # typed Storage handles and the two scalar primitives, and error.cpp the
  # thread-local slot the guarded wrappers record into.
  #
  # Deliberately complementary rather than a superset: test_sum_reduction
  # keeps H6's float64 sweep and test_matmul keeps H2's, while this one
  # proves the set I4 opened — both families at both dtypes, on both
  # traversals, with the float32 accumulation witness that I3's single
  # correctly-rounded operations could not support, and mixed dtype refused
  # in every participating handle position with the destination unmoved.
  add_executable(test_dtype_reduction_matmul
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dtype_reduction_matmul.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/reduction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/matmul.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/storage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_dtype_reduction_matmul PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_dtype_reduction_matmul PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_dtype_reduction_matmul PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_dtype_reduction_matmul PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME dtype_reduction_matmul COMMAND test_dtype_reduction_matmul)

  # Conv2d compute paths and their H9 dispatch: compiles conv2d.cpp directly
  # so it can reach the hidden tf::conv2d_*_prefers_* predicates and BOTH
  # compute paths per direction — the retained Phase-D generic loops and the
  # H9 optimized traversals — none of which the shared library exports. That
  # is what proves which path a geometry takes, and that the two agree bit
  # for bit, without adding any dispatch control to the shipped ABI.
  # error.cpp is compiled in for the same reason as the D2/D4/D5 targets:
  # conv2d.cpp carries the guarded exported wrappers, which reference the
  # thread-local error slot.
  add_executable(test_conv2d_execution
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_conv2d_execution.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/conv2d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_conv2d_execution PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_conv2d_execution PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_conv2d_execution PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_conv2d_execution PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME conv2d_execution COMMAND test_conv2d_execution)

  # Dtype-general Conv2d and MaxPool2d (Phase I, milestone I5): compiles
  # conv2d.cpp and pooling.cpp directly so it reaches the hidden
  # tf::conv2d_*_prefers_* predicates and BOTH compute paths per Conv2d
  # direction at BOTH dtypes, plus the winner-buffer dtype rule the shared
  # library enforces but does not expose. Deliberately complementary rather
  # than a superset: the D2-D9 targets keep their hand-computed float64
  # cases and test_conv2d_execution keeps H9's float64 sweeps, while this
  # one proves the set I5 opened — all three Conv2d directions and both
  # pooling directions at both dtypes, the float32 accumulation witnesses
  # on both traversals of each direction, the float64 winner buffer and
  # 2**53 plane bound at a float32 pool, and mixed dtype refused in every
  # participating handle position with the destination unmoved.
  # storage.cpp is compiled in because the exported wrappers are driven
  # through real typed storage handles; error.cpp for the guarded wrappers'
  # thread-local error slot.
  add_executable(test_dtype_cnn
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dtype_cnn.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/conv2d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/pooling.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/storage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/error.cpp")
  target_include_directories(test_dtype_cnn PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT MSVC)
    target_compile_options(test_dtype_cnn PRIVATE -Wall -Wextra)
  endif()
  if(TF_SANITIZE AND NOT MSVC)
    target_compile_options(test_dtype_cnn PRIVATE ${_tf_sanitize_flags})
    target_link_options(test_dtype_cnn PRIVATE ${_tf_sanitize_flags})
  endif()

  add_test(NAME dtype_cnn COMMAND test_dtype_cnn)
endif()
