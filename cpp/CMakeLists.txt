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
