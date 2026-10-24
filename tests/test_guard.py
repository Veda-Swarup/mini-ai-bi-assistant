"""
tests/test_guard.py
--------------------
Unit tests for app.guard.validate_sql().

We test the guard in isolation — no database, no LLM.
Each test targets one specific rule so failures are easy to diagnose.
"""

import pytest
from app.guard import UnsafeSQL, validate_sql


# ── Happy-path ────────────────────────────────────────────────────────────────

class TestValidSelects:
    """Queries that must pass validation and return normalised SQL."""

    def test_simple_select_passes(self):
        """A plain SELECT on v_revenue is the baseline happy path."""
        sql = "SELECT product, SUM(net_revenue) FROM v_revenue GROUP BY product"
        result = validate_sql(sql)
        # Normalised output must still reference v_revenue
        assert "v_revenue" in result.lower()

    def test_limit_injected_when_absent(self):
        """A query with no LIMIT must have LIMIT 200 appended."""
        result = validate_sql("SELECT * FROM v_revenue")
        assert "200" in result

    def test_existing_limit_preserved(self):
        """A query that already has LIMIT 10 must not get a second LIMIT."""
        result = validate_sql("SELECT * FROM v_revenue LIMIT 10")
        # LIMIT 10 present, LIMIT 200 must NOT be added
        assert "10" in result
        assert result.upper().count("LIMIT") == 1

    def test_cte_query_passes(self):
        """
        CTEs parse as a Select node with a 'with' attribute — they must
        be allowed.  The CTE alias is collected and added to the allowlist
        so the outer SELECT can reference it.
        """
        sql = """
            WITH monthly AS (
                SELECT order_month, SUM(net_revenue) AS rev
                FROM   v_revenue
                GROUP  BY order_month
            )
            SELECT order_month, rev
            FROM   monthly
            ORDER  BY order_month
        """
        result = validate_sql(sql)
        assert result  # returns normalised SQL, not empty

    def test_select_with_where_and_aggregation(self):
        """Realistic LLM output: filtered aggregate on v_revenue."""
        sql = (
            "SELECT region, ROUND(SUM(net_revenue), 2) AS revenue "
            "FROM v_revenue "
            "WHERE order_month = '2026-05' "
            "GROUP BY region "
            "ORDER BY revenue DESC "
            "LIMIT 5"
        )
        result = validate_sql(sql)
        assert "v_revenue" in result.lower()


# ── Destructive statements ────────────────────────────────────────────────────

class TestDestructiveStatements:
    """DROP, DELETE, INSERT must all be rejected."""

    def test_drop_table_rejected(self):
        """DROP TABLE is not a SELECT — must raise UnsafeSQL."""
        with pytest.raises(UnsafeSQL, match="SELECT"):
            validate_sql("DROP TABLE sales")

    def test_delete_rejected(self):
        """DELETE is not a SELECT — must raise UnsafeSQL."""
        with pytest.raises(UnsafeSQL, match="SELECT"):
            validate_sql("DELETE FROM v_revenue WHERE 1=1")

    def test_insert_rejected(self):
        """INSERT is not a SELECT — must raise UnsafeSQL."""
        with pytest.raises(UnsafeSQL, match="SELECT"):
            validate_sql("INSERT INTO v_revenue (transaction_id) VALUES ('X')")

    def test_update_rejected(self):
        """UPDATE is not a SELECT — must raise UnsafeSQL."""
        with pytest.raises(UnsafeSQL, match="SELECT"):
            validate_sql("UPDATE sales SET status = 'Completed' WHERE 1=1")


# ── Injection via multiple statements ────────────────────────────────────────

class TestSemicolonInjection:
    """Semicolons split input into multiple statements — all must be rejected."""

    def test_classic_sqli_rejected(self):
        """
        'SELECT 1; DROP TABLE sales' — the canonical SQL injection pattern.
        sqlglot.parse() returns two statements; the length check catches it.
        """
        with pytest.raises(UnsafeSQL, match="statement"):
            validate_sql("SELECT 1; DROP TABLE sales")

    def test_select_then_insert_rejected(self):
        """Two otherwise-valid statements still count as injection."""
        with pytest.raises(UnsafeSQL, match="statement"):
            validate_sql(
                "SELECT * FROM v_revenue; "
                "INSERT INTO v_revenue (transaction_id) VALUES ('FAKE')"
            )


# ── Table allowlist ───────────────────────────────────────────────────────────

class TestTableAllowlist:
    """Only v_revenue (and CTE aliases) are permitted in FROM / JOIN."""

    def test_direct_sales_table_rejected(self):
        """
        The raw `sales` table holds all statuses and internal columns.
        The LLM must never reach it directly — only through v_revenue.
        """
        with pytest.raises(UnsafeSQL, match="sales"):
            validate_sql("SELECT * FROM sales")

    def test_rejected_rows_table_rejected(self):
        """Internal audit table — must never be queryable by the LLM."""
        with pytest.raises(UnsafeSQL, match="rejected_rows"):
            validate_sql("SELECT * FROM rejected_rows")

    def test_union_with_forbidden_table_rejected(self):
        """
        UNION trick: the first branch looks safe, the second hits `sales`.
        sqlglot parses a UNION as a Union node, not a Select — caught by the
        SELECT-only type check before we even reach the allowlist walk.
        """
        with pytest.raises(UnsafeSQL):
            validate_sql(
                "SELECT transaction_id FROM v_revenue "
                "UNION ALL "
                "SELECT transaction_id FROM sales"
            )


# ── Comment-based tricks ──────────────────────────────────────────────────────

class TestCommentTricks:
    """
    sqlglot parses the AST — SQL inside comments is stripped before
    the tree is built.  These tests confirm the parser cannot be fooled.
    """

    def test_comment_disguises_forbidden_table(self):
        """
        Attacker puts 'sales' in the real FROM clause and uses a comment
        to make it look like 'v_revenue' is the target.
        sqlglot strips the comment and sees FROM sales → rejected.
        """
        with pytest.raises(UnsafeSQL, match="sales"):
            validate_sql("SELECT * FROM sales -- v_revenue")

    def test_block_comment_disguise(self):
        """Same trick with a block comment."""
        with pytest.raises(UnsafeSQL, match="sales"):
            validate_sql("SELECT * FROM sales /* this is fine, just querying v_revenue */")

    def test_drop_hidden_after_comment_newline(self):
        """
        'SELECT * FROM v_revenue -- \\nDROP TABLE sales'
        The -- only comments to end-of-line; DROP TABLE sales is on a new
        line and is a second statement → length check rejects it.
        """
        with pytest.raises(UnsafeSQL):
            validate_sql("SELECT * FROM v_revenue --\nDROP TABLE sales")


# ── Forbidden keywords ────────────────────────────────────────────────────────

class TestForbiddenKeywords:
    """PRAGMA and ATTACH are blocked by explicit keyword scan."""

    def test_pragma_rejected(self):
        """
        PRAGMA is not a SELECT — caught by the type check.
        sqlglot parses it as a Pragma node, which is not exp.Select.
        """
        with pytest.raises(UnsafeSQL):
            validate_sql("PRAGMA table_info(v_revenue)")

    def test_attach_rejected(self):
        with pytest.raises(UnsafeSQL, match="ATTACH"):
            validate_sql("ATTACH DATABASE ':memory:' AS mem")
