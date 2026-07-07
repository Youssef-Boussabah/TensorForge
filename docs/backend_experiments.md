# Backend experiments

This page tracks the advanced experimental line that started after the
v3.0 Python release. **Nothing here is part of the finished Python
framework** — `import tensorforge` never touches it, Tensor and
autograd are unchanged, and every existing API works exactly as
before.

## C++ backend — v0.1 (current)

The smallest possible proof that Python TensorForge can call compiled
C++ code:

- `cpp/elementwise_add.cpp` — one kernel, a plain C-ABI function that
  adds two float64 buffers. No Python C-API, no pybind11, no NumPy
  headers.
- `src/tensorforge/backends/cpp.py` — a ctypes wrapper that loads the
  compiled shared library and exposes `elementwise_add(a, b)`,
  handling array conversion and validation on the Python side.

Usage:

```python
from tensorforge.backends.cpp import elementwise_add

elementwise_add(np.array([1.0, 2.0]), np.array([3.0, 4.0]))  # [4. 6.]
```

### Building it

```
uv run python cpp/build.py
```

The build script uses `g++` or `clang++` if you have one. If not,
install the bundled-compiler fallback first (the `ziglang` package
ships a clang-based C++ compiler that works anywhere uv works):

```
uv sync --group cpp
uv run python cpp/build.py
```

The compiled library lands next to the wrapper and is gitignored.
If the backend is not built, importing `tensorforge.backends.cpp`
raises an ImportError with these instructions, and its tests skip.

### v0.1 limitations

- float64 only (other inputs are converted).
- Shapes must match exactly — no broadcasting.
- One kernel, elementwise add. Not connected to Tensor or autograd.
- A proof of mechanism, not a performance claim.

## What might come next

Future backend milestones may add more kernels (subtract, multiply,
divide, ReLU, eventually matmul), and only much later consider wiring
them into Tensor behind a flag. CUDA experiments remain a separate
future branch. The Python framework stays the reference
implementation throughout.
