"""
tests/test_etl.py
-----------------
Tests for app.etl.load().

Every test runs against a *temporary* SQLite file so the production
db/sales.db is never touched and tests are fully isolated.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.etl import CSV_PATH, load


# ---------------------------------------------------------------------------
# Module-level fixture: run ETL once into a temp DB for the whole file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """ETL runs once; all tests in this module share the result."""
    tmp = tmp_path_factory.mktemp("etl_test")
    p   = tmp / "test_sales.db"
    load(csv_path=CSV_PATH, db_path=p)
    return p


@pytest.fixture(scope="module")
def con(db_path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(db_path)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# 1 — Row counts
# ---------------------------------------------------------------------------

class TestRowCounts:
    def test_loaded_49_rows(self, con: sqlite3.Connection) -> None:
        """51 CSV rows − 2 bad rows = 49 in the sales table."""
        n = con.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
        assert n == 49

    def test_rejected_2_rows(self, con: sqlite3.Connection) -> None:
        """Exactly 2 rows must land in rejected_rows."""
        n = con.execute("SELECT COUNT(*) FROM rejected_rows").fetchone()[0]
        assert n == 2

    def test_rejected_ids_are_txn0034_and_txn0051(self, con: sqlite3.Connection) -> None:
        """TXN0034 (qty=0) and TXN0051 (blank unit_price) are the only rejects."""
        ids = {
            row[0]
            for row in con.execute("SELECT transaction_id FROM rejected_rows")
        }
        assert ids == {"TXN0034", "TXN0051"}

    def test_rejected_reasons(self, con: sqlite3.Connection) -> None:
        """Each rejected row must carry a human-readable reason."""
        reasons = {
            row[0]: row[1]
            for row in con.execute(
                "SELECT transaction_id, reason FROM rejected_rows"
            )
        }
        assert "0" in reasons["TXN0034"] or "must be" in reasons["TXN0034"].lower()
        assert "blank" in reasons["TXN0051"].lower()


# ---------------------------------------------------------------------------
# 2 — v_revenue view contract
# ---------------------------------------------------------------------------

class TestVRevenueView:
    def test_v_revenue_row_count(self, con: sqlite3.Connection) -> None:
        """49 loaded − 1 Returned (TXN0033) = 48 rows in v_revenue."""
        n = con.execute("SELECT COUNT(*) FROM v_revenue").fetchone()[0]
        assert n == 48

    def test_v_revenue_excludes_returned_order(self, con: sqlite3.Connection) -> None:
        """TXN0033 (Status=Returned) must not appear in the view."""
        row = con.execute(
            "SELECT COUNT(*) FROM v_revenue WHERE transaction_id = 'TXN0033'"
        ).fetchone()[0]
        assert row == 0

    def test_v_revenue_has_no_status_column(self, con: sqlite3.Connection) -> None:
        """status is filtered by the view definition — it must not be a column."""
        cols = [
            d[0]
            for d in con.execute("SELECT * FROM v_revenue LIMIT 1").description
        ]
        assert "status" not in cols

    def test_v_revenue_has_net_revenue_column(self, con: sqlite3.Connection) -> None:
        """net_revenue must be a pre-computed column available in the view."""
        cols = [
            d[0]
            for d in con.execute("SELECT * FROM v_revenue LIMIT 1").description
        ]
        assert "net_revenue" in cols


# ---------------------------------------------------------------------------
# 3 — Net revenue arithmetic
# ---------------------------------------------------------------------------

class TestNetRevenue:
    def test_txn0004_with_discount(self, con: sqlite3.Connection) -> None:
        """TXN0004: 6 × 9500 − 500 = 56500."""
        val = con.execute(
            "SELECT net_revenue FROM sales WHERE transaction_id = 'TXN0004'"
        ).fetchone()[0]
        assert val == pytest.approx(56500.0)

    def test_txn0001_no_discount(self, con: sqlite3.Connection) -> None:
        """TXN0001: 10 × 1200 − 0 = 12000."""
        val = con.execute(
            "SELECT net_revenue FROM sales WHERE transaction_id = 'TXN0001'"
        ).fetchone()[0]
        assert val == pytest.approx(12000.0)


# ---------------------------------------------------------------------------
# 4 — Date parsing
# ---------------------------------------------------------------------------

class TestDateParsing:
    def test_order_date_iso_format(self, con: sqlite3.Connection) -> None:
        """DD-MM-YYYY in CSV must be stored as YYYY-MM-DD in SQLite."""
        date = con.execute(
            "SELECT order_date FROM sales WHERE transaction_id = 'TXN0001'"
        ).fetchone()[0]
        assert date == "2026-01-03"

    def test_order_month_format(self, con: sqlite3.Connection) -> None:
        """order_month must be YYYY-MM for easy GROUP BY."""
        month = con.execute(
            "SELECT order_month FROM sales WHERE transaction_id = 'TXN0001'"
        ).fetchone()[0]
        assert month == "2026-01"


# ---------------------------------------------------------------------------
# 5 — Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_load_twice_same_counts(self, tmp_path: Path) -> None:
        """Calling load() twice on the same DB must not double-insert rows."""
        db = tmp_path / "idem.db"
        loaded1, rejected1 = load(csv_path=CSV_PATH, db_path=db)
        loaded2, rejected2 = load(csv_path=CSV_PATH, db_path=db)
        assert (loaded1, rejected1) == (loaded2, rejected2) == (49, 2)
