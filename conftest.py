"""Pytest configuration.

On this machine, directories created by one process often cannot be
deleted by a later one (broken ACLs — see CLAUDE.md). That breaks both
pytest's default tmp_path location (%TEMP%/pytest-of-<user>, whose old
numbered dirs can't be cleaned up) and any fixed --basetemp (whose
startup wipe fails on the previous session's directory).

So: give every session a fresh, unique basetemp. It never exists
beforehand, so pytest never has to delete anything.
"""

import os
import tempfile
import time
from pathlib import Path


def pytest_configure(config):
    if config.option.basetemp is None:
        unique = f"tensorforge-pytest-{os.getpid()}-{time.time_ns()}"
        config.option.basetemp = Path(tempfile.gettempdir()) / unique
