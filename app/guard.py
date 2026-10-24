"""
guard.py
--------
Validates LLM-generated SQL before it touches the database.

Three defence layers (this is layer 1 and 2; layer 3 is in db.py):
─────────────────────────────────────────────────────────────────────
1. sqlglot parse (layer 1)
   sqlglot builds a real AST — it cannot be tricked by SQL in comments,
   obfuscated casing, or whitespace tricks the way regex can.

2. Allowlist + statement checks (layer 2)
   Only SELECT statements on v_revenue (or CTE aliases defined inside
   the same query) can pass.  Everything else raises UnsafeSQL.

3. Read-only URI connection (layer 3, in db.py)
   Even if layers 1-2 were somehow bypassed, SQLite's ?mode=ro VFS
   flag makes writes physically impossible at the OS level.

The single public function is validate_sql(sql) -> str.
It either returns a normalised, safe SQL string or raises UnsafeSQL.
"""

from __future__ import annotations

import sqlglot
import sqlglot.expressions as exp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The only base table the LLM may query.
ALLOWED_TABLES: frozenset[str] = frozenset({"v_revenue"})

# Rows returned per query when the LLM omits a LIMIT clause.
# Caps runaway full-table scans; 200 is enough for any BI answer.
MAX_ROWS: int = 200

# These keywords cannot appear in a safe SELECT.  Checked as a second pass
# after the AST walk in case sqlglot ever changes how it represents them.
_BLOCKED_KEYWORDS: frozenset[str] = frozenset({"PRAGMA", "ATTACH", "DETACH"})


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class UnsafeSQL(Exception):
    """Raised when generated SQL fails any guard check.
    The message is safe to surface to the user."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_cte_names(stmt: exp.Select) -> set[str]:
    """
    Return the lowercase alias of every CTE defined in this query's
    WITH clause.  These names are allowed in FROM / JOIN because they
    are defined entirely within the same query — they never reference
    an external table.
    """
    cte_names: set[str] = set()
    with_clause = stmt.args.get("with")
    if with_clause:
        for cte in with_clause.expressions:   # each item is an exp.CTE node
            cte_names.add(cte.alias.lower())
    return cte_names


def _check_table_allowlist(stmt: exp.Select, allowed: frozenset[str]) -> None:
    """
    Walk every Table node in the AST (including inside CTEs, subqueries,
    and UNION branches) and reject any that are not in `allowed`.

    find_all() performs a depth-first traversal so it catches nested
    references that a shallow check would miss.
    """
    for table_node in stmt.find_all(exp.Table):
        name = table_node.name.lower()
        if name and name not in allowed:
            raise UnsafeSQL(
                f"Table '{table_node.name}' is not permitted. "
                "Only v_revenue may be queried."
            )


def _inject_limit(stmt: exp.Select) -> exp.Select:
    """
    Add LIMIT MAX_ROWS to the outermost SELECT if it has no LIMIT clause.
    This prevents the LLM from accidentally producing a full-table scan.
    """
    if stmt.args.get("limit") is None:
        stmt = stmt.limit(MAX_ROWS)
    return stmt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_sql(sql: str) -> str:
    """
    Parse, validate, and normalise a SQL string produced by the LLM.

    Validation steps
    ----------------
    1. sqlglot.parse        — rejects syntax errors and non-SQL garbage
    2. Single statement     — blocks semicolon-separated injection
                              e.g. "SELECT 1; DROP TABLE sales"
    3. SELECT-only          — rejects DROP / DELETE / INSERT / CREATE / PRAGMA
                              (CTEs are fine: they parse as a Select node)
    4. Blocked keywords     — belt-and-suspenders scan for PRAGMA / ATTACH
    5. Allowlist            — every Table node must be v_revenue or a CTE alias
    6. LIMIT injection      — appends LIMIT 200 when absent
    7. Normalise            — returns sqlglot's canonical SQLite string

    Returns
    -------
    Normalised, validated SQL string ready for db.run_query().

    Raises
    ------
    UnsafeSQL — on any violation; message is safe to show the user.
    """
    stripped = sql.strip()

    # ── Step 1: Parse ────────────────────────────────────────────────────
    # RAISE level means parse errors throw immediately instead of being
    # collected silently, so garbage input always raises UnsafeSQL.
    try:
        statements = sqlglot.parse(
            stripped,
            dialect="sqlite",
            error_level=sqlglot.ErrorLevel.RAISE,
        )
    except sqlglot.errors.SqlglotError as exc:
        raise UnsafeSQL(f"Unparseable SQL: {exc}") from exc

    # ── Step 2: Single statement ─────────────────────────────────────────
    # A semicolon splits input into multiple statements.
    # "SELECT 1; DROP TABLE sales" → [Select, Drop] → len == 2 → rejected.
    if len(statements) != 1 or statements[0] is None:
        raise UnsafeSQL(
            f"Expected exactly 1 statement, got {len(statements)}. "
            "Semicolons and multiple statements are not allowed."
        )

    stmt = statements[0]

    # ── Step 3: SELECT-only ──────────────────────────────────────────────
    # sqlglot represents  WITH cte AS (...) SELECT ...  as an exp.Select
    # node whose .args['with'] holds the CTE definitions.
    # So isinstance(stmt, exp.Select) is true for both plain SELECTs and CTEs.
    # Note: UNION ALL produces a Union node (not Select), so it is also
    # rejected here — preventing cross-table UNION tricks entirely.
    if not isinstance(stmt, exp.Select):
        raise UnsafeSQL(
            f"Only SELECT statements are allowed; received {type(stmt).__name__}."
        )

    # ── Step 4: Blocked keywords ─────────────────────────────────────────
    # Checked on the raw string (uppercased) so they are caught even if
    # sqlglot ever changes how it represents PRAGMA / ATTACH in the AST.
    upper = stripped.upper()
    for kw in _BLOCKED_KEYWORDS:
        if kw in upper:
            raise UnsafeSQL(f"Forbidden keyword: {kw}")

    # ── Step 5: Allowlist ────────────────────────────────────────────────
    cte_names = _collect_cte_names(stmt)
    allowed = ALLOWED_TABLES | cte_names      # CTE aliases are safe to reference
    _check_table_allowlist(stmt, allowed)

    # ── Step 6: LIMIT ────────────────────────────────────────────────────
    stmt = _inject_limit(stmt)

    # ── Step 7: Normalise ────────────────────────────────────────────────
    return stmt.sql(dialect="sqlite")
