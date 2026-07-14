"""Version consistency (repair milestone, Stage 11).

The distributable package version has one source of truth —
pyproject.toml's ``[project].version`` — surfaced as
``tensorforge.__version__`` via the installed metadata. This test pins
them together so they can never silently drift. The project's *milestone*
labels (v0.1 … v3.0, "Advanced C++ v3.x") are a separate concept and are
deliberately NOT tied to this number.
"""

import tomllib
from importlib.metadata import version as pkg_version
from pathlib import Path

import tensorforge

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject_version():
    with open(_PYPROJECT, "rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def test_package_exposes_version():
    assert isinstance(tensorforge.__version__, str)
    assert tensorforge.__version__


def test_version_matches_pyproject():
    assert tensorforge.__version__ == _pyproject_version()


def test_version_matches_installed_metadata():
    assert tensorforge.__version__ == pkg_version("tensorforge")
