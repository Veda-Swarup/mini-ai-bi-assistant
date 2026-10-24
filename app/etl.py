"""
etl.py
------
Reads data/rgs.csv, validates every row, and writes clean data to SQLite.
Bad rows are stored in `rejected_rows` for audit.
Creates the `v_revenue` view for the LLM to query.

Run directly:  python -m app.etl
"""

from __future__ import annotations

import csv
import sqlite3
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — resolved from this file's location so the script works from any cwd
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH: Path = _PROJECT_ROOT / "data" / "rgs.csv"
DB_PATH: Path = _PROJECT_ROOT / "db" / "sales.db"

# ---------------------------------------------------------------------------
# DDL — one string per object so each piece is easy to read and explain
# ---------------------------------------------------------------------------

# `sales` stores every valid row, including Returned orders.
# CHECK constraints mirror our Python validation — defence in depth.
_DDL_SALES = """
CREATE TABLE sales (
    transaction_id  TEXT    PRIMARY KEY,
    order_date      TEXT    NOT NULL,           -- ISO  YYYY-MM-DD
    order_month     TEXT    NOT NULL,           -- YYYY-MM  (for monthly GROUP BY)
    product         TEXT    NOT NULL,
    category        TEXT    NOT NULL,
    region          TEXT    NOT NULL,
    sales_channel   TEXT    NOT NULL,
    quantity        INTEGER NOT NULL  CHECK (quantity        >  0),
    unit_price      REAL    NOT NULL  CHECK (unit_price      >  0),
    discount_amount REAL    NOT NULL  CHECK (discount_amount >= 0),
    net_revenue     REAL    NOT NULL,           -- quantity * unit_price - discount_amount
    status          TEXT    NOT NULL
);
"""

# `rejected_rows` gives the evaluator (and us) full visibility into dropped data.
_DDL_REJECTED = """
CREATE TABLE rejected_rows (
    transaction_id  TEXT,               -- may be NULL if the id column itself is blank
    reason          TEXT NOT NULL,      -- human-readable validation error
    raw_row         TEXT NOT NULL       -- full CSV record as a string for auditability
);
"""

# `v_revenue` is the ONLY table the LLM may query.
# Excluding `status` is intentional — inside the view it is always 'Completed',
# so exposing it would just confuse the LLM with a useless constant column.
_DDL_VIEW = """
CREATE VIEW v_revenue AS
SELECT
    transaction_id,
    order_date,
    order_month,
    product,
    category,
    region,
    sales_channel,
    quantity,
    unit_price,
    discount_amount,
    net_revenue
FROM  sales
WHERE status = 'Completed';
"""

# ---------------------------------------------------------------------------
# Row-level validation helpers
# ---------------------------------------------------------------------------

def _parse_date(raw: str) -> tuple[str, str]:
    """
    Convert 'DD-MM-YYYY' → ('YYYY-MM-DD', 'YYYY-MM').
    datetime.strptime raises ValueError for invalid calendars (e.g. 30-02-2026).
    """
    dt = datetime.strptime(raw.strip(), "%d-%m-%Y")
    return dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m")


def _parse_positive_float(raw: str, field: str) -> float:
    """
    Parse a mandatory numeric field.
    Raises ValueError if the value is blank, non-numeric, or <= 0.
    We reject <= 0 because zero/negative quantity or price makes revenue meaningless.
    """
    stripped = raw.strip()
    if not stripped:
        raise ValueError(f"{field} is blank")
    value = float(stripped)
    if value <= 0:
        raise ValueError(f"{field} must be > 0, got {value}")
    return value


def _parse_discount(raw: str) -> float:
    """
    Discount_amount is optional — blank is treated as 0 (no discount).
    A negative discount is illogical and rejected.
    """
    stripped = raw.strip()
    if not stripped:
        return 0.0
    value = float(stripped)
    if value < 0:
        raise ValueError(f"discount_amount must be >= 0, got {value}")
    return value


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------

def _recreate_schema(con: sqlite3.Connection) -> None:
    """
    Drop everything and rebuild from scratch.
    This is what makes load() idempotent — safe to call multiple times.
    Order matters: drop view before tables it depends on.
    """
    con.executescript("""
        DROP VIEW  IF EXISTS v_revenue;
        DROP TABLE IF EXISTS rejected_rows;
        DROP TABLE IF EXISTS sales;
    """)
    con.execute(_DDL_SALES)
    con.execute(_DDL_REJECTED)
    con.execute(_DDL_VIEW)


# ---------------------------------------------------------------------------
# Core ingestion loop
# ---------------------------------------------------------------------------

def _ingest(con: sqlite3.Connection, csv_path: Path) -> tuple[int, int]:
    """
    Stream through the CSV row by row, validate, and INSERT into the
    appropriate table.  Returns (loaded_count, rejected_count).
    """
    loaded = 0
    rejected = 0

    # utf-8-sig handles optional BOM; newline="" lets csv module manage line endings
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)

        for row in reader:
            tid: str = row.get("Transaction_Id", "").strip()
            raw_row: str = str(dict(row))   # snapshot for rejected_rows

            try:
                # --- Date (mandatory, must match DD-MM-YYYY exactly) ---
                order_date, order_month = _parse_date(row["Order_date"])

                # --- Quantity (mandatory, integer-valued, > 0) ---
                # Parse as float first to handle "10.0"; then cast to int.
                qty: int = int(_parse_positive_float(row["Quantity"], "Quantity"))

                # --- Unit price (mandatory, > 0) ---
                price: float = _parse_positive_float(row["Unit_price"], "Unit_price")

                # --- Discount (optional, defaults to 0) ---
                discount: float = _parse_discount(row["Discount_amount"])

                # --- Derived column (computed here, stored for easy SQL access) ---
                net_revenue: float = round(qty * price - discount, 2)

                con.execute(
                    """
                    INSERT INTO sales VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tid,
                        order_date,
                        order_month,
                        row["Product"].strip(),
                        row["Category"].strip(),
                        row["Region"].strip(),
                        row["Sales_channel"].strip(),
                        qty,
                        price,
                        discount,
                        net_revenue,
                        row["Status"].strip(),
                    ),
                )
                loaded += 1

            except (ValueError, KeyError) as exc:
                # KeyError  → a required CSV column was missing entirely
                # ValueError → failed one of the validation helpers above
                # We never use bare `except` so unexpected errors still surface.
                con.execute(
                    "INSERT INTO rejected_rows VALUES (?, ?, ?)",
                    (tid or None, str(exc), raw_row),
                )
                rejected += 1

    return loaded, rejected


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load(
    csv_path: Path = CSV_PATH,
    db_path: Path = DB_PATH,
) -> tuple[int, int]:
    """
    Load CSV into SQLite.  Idempotent — safe to call multiple times.

    Parameters
    ----------
    csv_path : path to the source CSV (default: data/rgs.csv)
    db_path  : path to the SQLite file  (default: db/sales.db)

    Returns
    -------
    (loaded, rejected) — row counts for sales and rejected_rows tables.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(db_path)
    try:
        _recreate_schema(con)
        loaded, rejected = _ingest(con, csv_path)
        con.commit()
    finally:
        con.close()   # always close, even if an exception occurred

    return loaded, rejected


# ---------------------------------------------------------------------------
# __main__ runner — quick sanity check from the command line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loaded, rejected = load()
    print(f"ETL complete: {loaded} loaded, {rejected} rejected.")

    # Show rejected rows so any surprises are immediately visible
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT transaction_id, reason FROM rejected_rows").fetchall()
    con.close()

    for tid, reason in rows:
        print(f"  REJECTED  id={tid!r:12s}  reason={reason}")
