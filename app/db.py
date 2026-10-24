"""
db.py
-----
Executes a validated SQL string against the SQLite database.

Three defence layers (full picture)
─────────────────────────────────────────────────────────────────────
1. sqlglot parse + statement type check  (guard.py)
   The AST parser rejects non-SELECT statements and syntax errors
   before a connection is opened.

2. Table allowlist                       (guard.py)
   Every FROM / JOIN reference is checked against {"v_revenue"}.
   The LLM can never name a table it has not been explicitly permitted
   to use.

3. Read-only URI  "file:<path>?mode=ro"  (this file)
   SQLite's URI virtual-filesystem flag opens the file with O_RDONLY.
   Even if layers 1 and 2 were bypassed, no INSERT / UPDATE / DROP /
   ATTACH can succeed — the OS will refuse the write.

run_query() is the only function exported. It never opens a writable
connection, never calls commit(), and never swallows exceptions.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH: Path = _PROJECT_ROOT / "db" / "sales.db"

# How long (seconds) to wait if the database is locked before giving up.
# Five seconds is generous for a small SQLite file.
_TIMEOUT_SECONDS: float = 5.0


def run_query(
    sql: str,
    db_path: Path = DB_PATH,
) -> tuple[list[str], list[tuple]]:
    """
    Execute a validated SELECT and return its results.

    Parameters
    ----------
    sql     : SQL string that has already passed guard.validate_sql().
              Passing unvalidated SQL here is a caller error.
    db_path : path to the SQLite file (default: db/sales.db).

    Returns
    -------
    columns : list[str]   — column names in result order
    rows    : list[tuple] — one tuple per result row

    Raises
    ------
    sqlite3.OperationalError
        Database file not found, connection locked beyond timeout, or
        (if somehow unvalidated SQL is passed) a write attempt on a
        read-only connection.
    """
    # Resolve to absolute path so the URI is unambiguous regardless of cwd.
    uri = f"file:{db_path.resolve()}?mode=ro"

    # uri=True tells sqlite3 to interpret the first arg as a URI, not a path.
    con = sqlite3.connect(uri, uri=True, timeout=_TIMEOUT_SECONDS)
    try:
        cur = con.execute(sql)
        # cur.description is a sequence of 7-item tuples; index 0 is column name.
        columns: list[str] = [desc[0] for desc in cur.description]
        rows: list[tuple] = cur.fetchall()
        return columns, rows
    finally:
        # Always close — even on exception — to release the file handle.
        con.close()
