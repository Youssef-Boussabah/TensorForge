"""Build the experimental C++ backend.

Compiles cpp/kernels.cpp into a shared library next to the
Python wrapper (src/tensorforge/backends/). Tries system compilers
first; if none exist, falls back to the ziglang pip package, which
bundles a full clang-based C++ compiler and works anywhere uv works:

    uv sync --group cpp
    uv run python cpp/build.py

The built library is a plain C-ABI shared object loaded with ctypes —
no Python extension machinery, so the normal package build (uv_build)
is completely unaffected.
"""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(__file__).with_name("kernels.cpp")
SUFFIX = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")
OUTPUT = REPO_ROOT / "src" / "tensorforge" / "backends" / ("_tensorforge_cpp" + SUFFIX)


def _compiler_commands():
    """Yield candidate compiler invocations, preferred first."""
    for name in ("g++", "clang++"):
        if shutil.which(name):
            yield [name]
    try:
        import ziglang  # noqa: F401  (pip package bundling a C++ compiler)

        yield [sys.executable, "-m", "ziglang", "c++"]
    except ImportError:
        pass


def _remove_side_artifacts():
    """Drop linker leftovers (import lib, debug info) some compilers
    emit next to the DLL — only the shared library itself is needed."""
    for leftover in OUTPUT.parent.glob("_tensorforge_cpp.pdb"):
        leftover.unlink()
    for leftover in OUTPUT.parent.glob(f"{SOURCE.stem}.lib"):
        leftover.unlink()


def main():
    for base in _compiler_commands():
        command = base + ["-O2", "-shared", "-o", str(OUTPUT), str(SOURCE)]
        print("building:", " ".join(command))
        if subprocess.run(command).returncode == 0:
            _remove_side_artifacts()
            print(f"built {OUTPUT.relative_to(REPO_ROOT)}")
            return 0
        print("compiler failed, trying the next one...")
    print("No working C++ compiler found. Either install g++/clang++, or use")
    print("the bundled-compiler fallback:")
    print("    uv sync --group cpp")
    print("    uv run python cpp/build.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
