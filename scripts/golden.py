"""
scripts/golden.py
-----------------
Runs deterministic SQL queries against v_revenue and prints the results.
These numbers are the GROUND TRUTH used as expected values in the test suite.
Verify them in a spreadsheet BEFORE writing any tests.

No LLM, no business logic — pure SQL + formatting.

Usage:
    python scripts/golden.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "db" / "sales.db"

# ---------------------------------------------------------------------------
# Helper — run a query and print a labelled table
# ---------------------------------------------------------------------------

def _print_section(title: str, rows: list[tuple], headers: list[str]) -> None:
    """Print a titled, column-aligned result table."""
    separator = "-" * 60
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(separator)

    # Calculate column widths from headers + data
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Header row
    header_line = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("  ".join("-" * w for w in widths))

    # Data rows
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))

    print(separator)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def q1_total_revenue(cur: sqlite3.Cursor) -> None:
    """Sum of all net_revenue across every completed order."""
    cur.execute("SELECT ROUND(SUM(net_revenue), 2) AS total_revenue FROM v_revenue")
    rows = cur.fetchall()
    _print_section("Q1 — Total Revenue", rows, ["total_revenue"])


def q2_revenue_by_product(cur: sqlite3.Cursor) -> None:
    """Net revenue per product, highest first."""
    cur.execute("""
        SELECT
            product,
            ROUND(SUM(net_revenue), 2) AS revenue
        FROM  v_revenue
        GROUP BY product
        ORDER BY revenue DESC
    """)
    rows = cur.fetchall()
    _print_section("Q2 — Revenue by Product (desc)", rows, ["product", "revenue"])


def q3_revenue_by_region(cur: sqlite3.Cursor) -> None:
    """Net revenue per region, highest first."""
    cur.execute("""
        SELECT
            region,
            ROUND(SUM(net_revenue), 2) AS revenue
        FROM  v_revenue
        GROUP BY region
        ORDER BY revenue DESC
    """)
    rows = cur.fetchall()
    _print_section("Q3 — Revenue by Region (desc)", rows, ["region", "revenue"])


def q4_monthly_trend(cur: sqlite3.Cursor) -> None:
    """Net revenue per calendar month, chronological order."""
    cur.execute("""
        SELECT
            order_month,
            ROUND(SUM(net_revenue), 2) AS revenue
        FROM  v_revenue
        GROUP BY order_month
        ORDER BY order_month
    """)
    rows = cur.fetchall()
    _print_section("Q4 — Monthly Revenue Trend", rows, ["order_month", "revenue"])


def q5_month_on_month_change(cur: sqlite3.Cursor) -> None:
    """
    Absolute and percentage revenue change between 2026-02 and 2026-03.
    Uses a CTE so each month's total is computed once.
    """
    cur.execute("""
        WITH monthly AS (
            SELECT
                order_month,
                ROUND(SUM(net_revenue), 2) AS revenue
            FROM  v_revenue
            WHERE order_month IN ('2026-02', '2026-03')
            GROUP BY order_month
        )
        SELECT
            feb.revenue                                   AS feb_revenue,
            mar.revenue                                   AS mar_revenue,
            ROUND(mar.revenue - feb.revenue, 2)           AS absolute_change,
            ROUND(
                (mar.revenue - feb.revenue) * 100.0
                / feb.revenue
            , 2)                                          AS pct_change
        FROM  monthly feb
        JOIN  monthly mar ON feb.order_month = '2026-02'
                         AND mar.order_month = '2026-03'
    """)
    rows = cur.fetchall()
    _print_section(
        "Q5 — Revenue Change: 2026-02 → 2026-03",
        rows,
        ["feb_revenue", "mar_revenue", "absolute_change", "pct_change (%)"],
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run `python -m app.etl` first."
        )

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        q1_total_revenue(cur)
        q2_revenue_by_product(cur)
        q3_revenue_by_region(cur)
        q4_monthly_trend(cur)
        q5_month_on_month_change(cur)
    finally:
        con.close()

    print("\n✓ All queries complete. Cross-check these numbers in your spreadsheet.\n")


if __name__ == "__main__":
    main()
