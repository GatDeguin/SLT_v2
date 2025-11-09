"""Pytest configuration helpers for the SLT project.

This repository's ``pytest.ini`` passes ``--cov`` options by default so that
coverage is collected when the ``pytest-cov`` plugin is available.  The plugin
is not part of the execution environment for these kata style exercises,
which causes ``pytest`` to abort with ``unrecognized arguments`` errors before
any tests run.  To keep the configuration flexible while still allowing the
tests to execute in environments without ``pytest-cov`` we provide a minimal
shim that registers the relevant command line options when the plugin cannot be
imported.  When ``pytest-cov`` *is* installed this shim stays dormant and the
real plugin handles the options, so normal development workflows remain
unchanged.
"""

from __future__ import annotations

from typing import Any


def pytest_addoption(parser: Any) -> None:
    """Register dummy ``--cov`` options when ``pytest-cov`` is unavailable.

    ``pytest.ini`` supplies ``--cov`` and ``--cov-report`` arguments via
    ``addopts``.  In environments where the ``pytest-cov`` plugin is not
    installed Pytest raises an ``UsageError`` because those options are
    unknown.  We catch that situation early by trying to import the plugin and
    only registering lightweight placeholders if the import fails.  The
    placeholders accept and store the arguments but otherwise have no effect,
    which is sufficient to let the test suite proceed.
    """

    try:  # pragma: no cover - we simply need the import check for behaviour
        import pytest_cov  # type: ignore  # noqa: F401  (imported for side effect)
    except ModuleNotFoundError:
        parser.addoption(
            "--cov",
            action="append",
            default=[],
            help="Dummy option registered when pytest-cov is unavailable.",
        )
        parser.addoption(
            "--cov-report",
            action="append",
            default=[],
            help="Dummy option registered when pytest-cov is unavailable.",
        )

