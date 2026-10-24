"""
tests/conftest.py
-----------------
Session-level fixtures and custom marker plumbing shared by all test files.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# --live CLI flag
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests that call the real Gemini API (requires GEMINI_API_KEY in .env).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip @pytest.mark.live tests unless --live is passed."""
    if not config.getoption("--live"):
        skip = pytest.mark.skip(reason="skipped by default — use --live to run")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip)


# ---------------------------------------------------------------------------
# Real database — ensure it exists once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def ensure_real_db() -> None:
    """
    Guarantee db/sales.db exists before any test that queries it.
    Runs ETL once per pytest session if the file is absent; a no-op otherwise.
    """
    from app.etl import DB_PATH, load

    if not DB_PATH.exists():
        load()
