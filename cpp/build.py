"""Build the experimental C++ backend.

Compiles the native sources in ``cpp/src/`` (headers in ``cpp/include/``)
into one shared library next to the Python wrapper
(``src/tensorforge/backends/``), loaded with ctypes — no Python extension
machinery, so the normal package build (uv_build) is unaffected.

This is a **thin wrapper around the canonical CMake build**
(``cpp/CMakeLists.txt``): when ``cmake`` is on PATH it configures and
builds through CMake, which owns the real compilation architecture
(standard, per-config flags, optional sanitizers — see the CMakeLists and
docs/backend_experiments.md). When CMake is not available — as on CI,
which builds with the runner's ``g++`` or the bundled ``ziglang`` package
— it falls back to a single direct compiler invocation over the same
source list. The fallback does not reproduce CMake's configuration logic;
it just compiles every ``cpp/src/*.cpp`` together.

    # with a C++ compiler on PATH:
    uv run python cpp/build.py
    # no system compiler? install the bundled one first:
    uv sync --group cpp
    uv run python cpp/build.py
    # a debug build (unoptimized, assertions on):
    uv run python cpp/build.py --debug

CMake developers can additionally choose sanitizers; see the CMakeLists.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CPP_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = CPP_ROOT / "src"
INCLUDE_DIR = CPP_ROOT / "include"
SOURCES = sorted(SOURCE_DIR.glob("*.cpp"))
SUFFIX = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")
LIBRARY_NAME = "_tensorforge_cpp"
OUTPUT_DIR = REPO_ROOT / "src" / "tensorforge" / "backends"
OUTPUT = OUTPUT_DIR / (LIBRARY_NAME + SUFFIX)


def _remove_side_artifacts():
    """Drop linker leftovers (import lib, debug info) some compilers emit
    next to the DLL — only the shared library itself is needed."""
    for pattern in (f"{LIBRARY_NAME}.pdb", f"{LIBRARY_NAME}.lib", "*.exp"):
        for leftover in OUTPUT_DIR.glob(pattern):
            if leftover != OUTPUT:
                leftover.unlink()


def _compiler_commands():
    """Yield candidate direct-compiler invocations, preferred first."""
    for name in ("g++", "clang++"):
        if shutil.which(name):
            yield [name]
    try:
        import ziglang  # noqa: F401  (pip package bundling a C++ compiler)

        yield [sys.executable, "-m", "ziglang", "c++"]
    except ImportError:
        pass


def _build_with_cmake(build_type):
    """Configure and build through CMake. Returns True on success, False
    if CMake is unavailable (so the caller can fall back)."""
    if shutil.which("cmake") is None:
        return False
    build_dir = CPP_ROOT / "build"
    build_dir.mkdir(exist_ok=True)
    configure = [
        "cmake", "-S", str(CPP_ROOT), "-B", str(build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DTF_OUTPUT_DIR={OUTPUT_DIR}",
    ]
    print("configuring:", " ".join(configure))
    if subprocess.run(configure).returncode != 0:
        print("cmake configure failed")
        return True  # cmake exists but failed — do not silently fall back
    build = ["cmake", "--build", str(build_dir), "--config", build_type]
    print("building:", " ".join(build))
    if subprocess.run(build).returncode == 0:
        _remove_side_artifacts()
        print(f"built {OUTPUT.relative_to(REPO_ROOT)} (cmake, {build_type})")
    else:
        print("cmake build failed")
    return True


def _build_direct(build_type):
    """Compile every source in one direct invocation. The CI/no-CMake
    path; kept minimal on purpose."""
    optimization = ["-O0", "-g"] if build_type == "Debug" else ["-O2"]
    for base in _compiler_commands():
        command = (
            base
            + ["-std=c++17", "-fPIC", "-fvisibility=hidden"]
            + optimization
            + ["-I", str(INCLUDE_DIR), "-shared", "-o", str(OUTPUT)]
            + [str(source) for source in SOURCES]
        )
        print("building:", " ".join(command))
        if subprocess.run(command).returncode == 0:
            _remove_side_artifacts()
            print(f"built {OUTPUT.relative_to(REPO_ROOT)} (direct, {build_type})")
            return 0
        print("compiler failed, trying the next one...")
    print("No working C++ compiler found. Either install g++/clang++, or use")
    print("the bundled-compiler fallback:")
    print("    uv sync --group cpp")
    print("    uv run python cpp/build.py")
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the C++ backend.")
    parser.add_argument(
        "--debug", action="store_true",
        help="unoptimized debug build (-O0 -g); default is Release (-O2)",
    )
    parser.add_argument(
        "--no-cmake", action="store_true",
        help="skip CMake and use the direct compiler fallback",
    )
    args = parser.parse_args(argv)
    build_type = "Debug" if args.debug else "Release"

    if not SOURCES:
        print(f"no sources found in {SOURCE_DIR}")
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.no_cmake and _build_with_cmake(build_type):
        return 0 if OUTPUT.exists() else 1
    return _build_direct(build_type)


if __name__ == "__main__":
    raise SystemExit(main())
