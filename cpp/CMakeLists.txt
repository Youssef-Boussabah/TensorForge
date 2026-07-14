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
endif()
