"""The unavailable-backend contract, executed rather than skipped.

Post-Phase-G test maintenance. Five Phase-E operations each carried a
test asserting that, without the compiled library, the operation raises
ImportError with build instructions and never falls back to NumPy:

    tests/test_native_exp.py                test_native_exp_...
    tests/test_native_log.py                test_native_log_...
    tests/test_native_softmax.py            test_native_softmax_...
    tests/test_native_log_softmax.py        test_native_log_softmax_...
    tests/test_native_cross_entropy_core.py test_native_cross_entropy_...

Each began with ``if cpp.is_available(): pytest.skip(...)``, so on any
machine where the backend *is* built — every machine that can run the
native suite at all — the behavior was never actually executed. Five
tests only ran where they were least needed.

The condition cannot be forced in-process: by the time these modules
import, ``cpp._lib`` may already hold the real DLL, live NativeTensorCore
objects hold handles into it, and clearing the cache or repointing
``cpp._LIBRARY_PATH`` would corrupt the rest of the session. So the
simulation happens in a **fresh child process** instead, which starts
with ``cpp._lib is None`` (the wrapper loads lazily) and can repoint its
own module-private ``_LIBRARY_PATH`` at a guaranteed nonexistent file
inside pytest's ``tmp_path``.

Nothing here renames, moves, deletes, overwrites, or re-permissions the
real ``_tensorforge_cpp`` library, and nothing mutates the parent pytest
process's backend state. Every case fingerprints the real library on
both sides of its subprocess to prove it, and separate tests re-verify
that the parent's backend is still loaded, still computing, and still
reporting the same capability registries.

This is test quality only: no runtime file, capability, registry value,
or Phase-G artifact is involved.

Selector: python -m pytest -q -k "backend_unavailable"
"""

import json
import subprocess
import sys
from pathlib import Path
from string import Template

import numpy as np
import pytest

from tensorforge.backends import cpp

REPO_ROOT = Path(__file__).resolve().parents[1]

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)

# Generous next to the child's real cost (import tensorforge, fail one
# load), and the same order of magnitude as the repository's other
# subprocess tests (120 s in tests/test_multiclass_example.py).
CHILD_TIMEOUT = 120

# Read-only: the parent never assigns to cpp._LIBRARY_PATH. This is the
# file whose continued existence, size, and mtime every case asserts.
REAL_LIBRARY = Path(cpp._LIBRARY_PATH)


def _fingerprint(path):
    """(exists, size, mtime_ns) — enough to catch any rename, deletion,
    truncation, or rewrite of the real library."""
    if not path.exists():
        return (False, None, None)
    stat = path.stat()
    return (True, stat.st_size, stat.st_mtime_ns)


# --------------------------------------------------------------------------
# The five operations
#
# One entry per formerly-skipped test, so each behavior stays its own
# pytest case with its owning module named beside it. The expressions are
# the ones those tests used, reduced to the smallest valid input for each
# signature; test_the_five_expressions_succeed_against_the_real_backend
# below proves they are genuinely valid, so a child's ImportError can only
# have come from the missing library and not from a malformed call.
# --------------------------------------------------------------------------

OPERATIONS = {
    "exp": (
        "tests/test_native_exp.py",
        "cpp.NativeTensorCore.from_array([1.0]).exp()",
    ),
    "log": (
        "tests/test_native_log.py",
        "cpp.NativeTensorCore.from_array([1.0]).log()",
    ),
    "softmax": (
        "tests/test_native_softmax.py",
        "cpp.NativeTensorCore.from_array([1.0, 2.0]).softmax(-1)",
    ),
    "log_softmax": (
        "tests/test_native_log_softmax.py",
        "cpp.NativeTensorCore.from_array([1.0, 2.0]).log_softmax(-1)",
    ),
    "cross_entropy": (
        "tests/test_native_cross_entropy_core.py",
        "cpp.NativeTensorCore.from_array([[1.0, 2.0]]).cross_entropy_forward([1])",
    ),
}


# --------------------------------------------------------------------------
# The child program
#
# Written once and specialized per case, so the five subprocess bodies are
# one script rather than five copies. It reports as JSON on stdout and
# exits 0 only when every step held; distinct nonzero codes separate an
# unexpected exception (2) from a NumPy-style fallback that returned a
# value instead of raising (3).
# --------------------------------------------------------------------------

_CHILD_SCRIPT = Template('''"""Child: simulate an unavailable native backend, in this process only.

Written into a pytest tmp_path by tests/test_native_backend_unavailable.py
and run with the same interpreter. It never touches the real compiled
library file and never affects the parent pytest process.
"""

import json
import sys
import traceback
from pathlib import Path

from tensorforge.backends import cpp

report = {"operation": $OPERATION_NAME}

# The wrapper loads lazily, so a fresh process starts with no library
# cached. This is exactly what the parent process cannot promise, and
# the reason the simulation lives out here.
assert cpp._lib is None, "importing the backend wrapper loaded the library"
report["lib_none_on_import"] = True

# The real library, recorded before and after purely as evidence that
# nothing below goes anywhere near it.
real = Path(cpp._LIBRARY_PATH)
report["real_library_name"] = real.name
report["real_before"] = [real.exists(),
                         real.stat().st_size if real.exists() else None]

# Repoint THIS process's module-private path at the simulated library and
# clear the (still empty) cache, so the next use attempts a real load.
simulated = Path($LIBRARY_PATH)
if $EXPECT_EXISTS:
    assert simulated.exists(), "the simulated library file should exist"
else:
    assert not simulated.exists(), "the simulated library path must not exist"
cpp._LIBRARY_PATH = simulated
cpp._lib = None
report["simulated_library_path"] = str(simulated)

# 1. The backend reports itself unavailable — a real attempted load, not
#    a file-existence guess, and it must not raise.
available = cpp.is_available()
report["is_available"] = available
assert available is False, "is_available() did not return False"

# 2-4. The operation itself, on the ordinary public path, must raise
#      ImportError carrying the build instructions.
outcome = raised = None
try:
    outcome = $OPERATION
except ImportError as error:
    raised = error
except BaseException:
    report["unexpected_traceback"] = traceback.format_exc()
    print(json.dumps(report))
    sys.exit(2)

if raised is None:
    # 5. No NumPy fallback: returning anything at all is a failure.
    report["fallback_result"] = repr(outcome)
    print(json.dumps(report))
    sys.exit(3)

report["exception_type"] = type(raised).__name__
report["message"] = str(raised)
report["build_instructions"] = cpp.build_instructions()

# The failure cached nothing, so a retry re-attempts the load rather than
# handing back a half-initialized library.
assert cpp._lib is None, "a library was cached despite the failed load"
report["lib_none_after_failure"] = True

# 6. The real library file is exactly as it was.
report["real_after"] = [real.exists(),
                        real.stat().st_size if real.exists() else None]

print(json.dumps(report))
sys.exit(0)
''')


def _run_child(tmp_path, operation_name, expression, library_path,
               expect_exists=False):
    """Write and run the specialized child, returning its JSON report.

    ``library_path`` is embedded with ``repr()``, so Windows backslashes
    and spaces cross safely and no shell quoting is involved.
    """
    script = tmp_path / f"unavailable_{operation_name}.py"
    script.write_text(
        _CHILD_SCRIPT.substitute(
            OPERATION_NAME=repr(operation_name),
            LIBRARY_PATH=repr(str(library_path)),
            EXPECT_EXISTS=repr(bool(expect_exists)),
            OPERATION=expression,
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        check=False, timeout=CHILD_TIMEOUT,
    )
    context = (
        f"\n--- child exit {result.returncode} ---"
        f"\n--- stdout ---\n{result.stdout}"
        f"\n--- stderr ---\n{result.stderr}"
    )
    assert result.returncode == 0, context
    assert "Traceback" not in result.stderr, context
    assert "Traceback" not in result.stdout, context
    return json.loads(result.stdout), result


# ==========================================================================
# The five behaviors, executed
# ==========================================================================


@pytest.mark.parametrize("operation", list(OPERATIONS))
def test_operation_requires_the_built_backend(operation, tmp_path):
    """Without the compiled library the operation raises ImportError with
    build instructions — never a silent NumPy fallback.

    This is the assertion the five owning modules used to make and then
    skip; it now runs on every machine, built backend or not.
    """
    owner, expression = OPERATIONS[operation]
    before = _fingerprint(REAL_LIBRARY)

    # A path that cannot exist: a fresh, never-created subdirectory of
    # this test's own tmp_path, carrying the real library's basename so
    # the error message names the file a user would expect.
    missing = tmp_path / "no-such-build-directory" / REAL_LIBRARY.name
    assert not missing.exists()

    report, _ = _run_child(tmp_path, operation, expression, missing)

    # The child started clean and never had a library cached.
    assert report["operation"] == operation
    assert report["lib_none_on_import"] is True
    assert report["lib_none_after_failure"] is True

    # 1. Unavailable.
    assert report["is_available"] is False

    # 2. ImportError, from the ordinary public operation path.
    assert report["exception_type"] == "ImportError", owner

    # 3. The message identifies the missing experimental backend.
    message = report["message"]
    assert "experimental C++ backend" in message, message
    assert "not built" in message, message
    assert REAL_LIBRARY.name in message, message

    # 4. ...and carries the documented build instructions verbatim.
    assert "cpp/build.py" in message, message
    assert "uv run python cpp/build.py" in message, message
    assert "uv sync --group cpp" in message, message
    assert report["build_instructions"] in message, message

    # 5. No fallback value was produced (the child exits 3 if one is).
    assert "fallback_result" not in report
    assert "unexpected_traceback" not in report

    # 6. The real library was never touched — by the child, or by this.
    assert report["real_before"] == report["real_after"]
    assert report["real_library_name"] == REAL_LIBRARY.name
    assert _fingerprint(REAL_LIBRARY) == before

    # 7./8. Nothing escaped tmp_path: the script is the only artifact,
    #       and the nonexistent directory stayed nonexistent.
    assert not missing.exists()
    assert not missing.parent.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        f"unavailable_{operation}.py"
    ]


def test_a_library_that_exists_but_cannot_be_loaded_also_raises(tmp_path):
    """The contract's other half: a present-but-unloadable library gives
    the same ImportError with the same instructions, not an OSError.

    The decoy is a text file the parent writes inside tmp_path; the real
    library is still never involved.
    """
    before = _fingerprint(REAL_LIBRARY)
    decoy = tmp_path / REAL_LIBRARY.name
    decoy.write_text("not a shared library\n", encoding="utf-8")

    report, _ = _run_child(tmp_path, "exp", OPERATIONS["exp"][1], decoy,
                           expect_exists=True)

    assert report["is_available"] is False
    assert report["exception_type"] == "ImportError"
    message = report["message"]
    assert "experimental C++ backend" in message, message
    assert "failed to load" in message, message
    assert "cpp/build.py" in message, message
    assert report["build_instructions"] in message, message
    assert "fallback_result" not in report
    assert report["real_before"] == report["real_after"]
    assert _fingerprint(REAL_LIBRARY) == before
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [REAL_LIBRARY.name, "unavailable_exp.py"]
    )


# ==========================================================================
# The parent process is unaffected
# ==========================================================================


@needs_native
def test_the_five_expressions_succeed_against_the_real_backend():
    """Each simulated call is a *valid* operation, so the child's
    ImportError can only come from the missing library.

    Without this, a typo in one of the five expressions would raise some
    other error in the child and the case would prove nothing.
    """
    results = {
        "exp": cpp.NativeTensorCore.from_array([1.0]).exp(),
        "log": cpp.NativeTensorCore.from_array([1.0]).log(),
        "softmax": cpp.NativeTensorCore.from_array([1.0, 2.0]).softmax(-1),
        "log_softmax":
            cpp.NativeTensorCore.from_array([1.0, 2.0]).log_softmax(-1),
    }
    assert sorted(list(results) + ["cross_entropy"]) == sorted(OPERATIONS)
    assert np.allclose(results["exp"].to_numpy(), np.exp([1.0]), atol=1e-15)
    assert np.allclose(results["log"].to_numpy(), [0.0], atol=1e-15)
    assert np.allclose(results["softmax"].to_numpy().sum(), 1.0, atol=1e-15)
    assert np.allclose(np.exp(results["log_softmax"].to_numpy()).sum(), 1.0,
                       atol=1e-15)

    logits = cpp.NativeTensorCore.from_array([[1.0, 2.0]])
    outcome = logits.cross_entropy_forward([1])
    assert np.isfinite(float(outcome.loss.to_numpy()))
    outcome.close()
    logits.close()
    for core in results.values():
        core.close()


@needs_native
def test_the_parent_backend_is_untouched_by_the_simulation():
    """The parent's own backend state is exactly as the session left it:
    the real path, a loaded library, and working native math."""
    assert Path(cpp._LIBRARY_PATH) == REAL_LIBRARY
    assert REAL_LIBRARY.exists()
    assert cpp.is_available() is True
    assert cpp._lib is not None

    # A small native smoke operation still computes.
    core = cpp.NativeTensorCore.from_array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(core.relu().to_numpy(), [[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(core.sum().to_numpy(), 10.0)
    core.close()
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    assert cpp.backend_info()["available"] is True


def test_capability_registries_are_unchanged():
    """Post-Phase-G maintenance moves no capability boundary."""
    assert cpp.UNSUPPORTED == ("float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert "dropout" in cpp.AUTOGRAD_OPS
    assert "dropout" not in cpp.UNSUPPORTED
    assert "NativeDropout" in cpp.NATIVE_MODULES


def test_the_simulation_never_mutates_the_parent_backend_state():
    """A source-level guardrail: the simulation must stay in the child.

    Rebinding the backend's private library cache or its library path in
    *this* process would corrupt the rest of the pytest session, so those
    two assignments must appear only inside the child script template.
    The searched tokens are assembled from fragments so this guardrail
    cannot match its own source.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    head, rest = source.split("_CHILD_SCRIPT = Template('''", 1)
    child, tail = rest.split("''')", 1)
    outside_the_child = head + tail

    for name in ("_lib", "_LIBRARY_PATH"):
        assignment = "cpp." + name + " ="
        assert assignment not in outside_the_child, name
        assert assignment in child, name          # the child is where it lives
    assert "setattr(" + "cpp" not in outside_the_child
