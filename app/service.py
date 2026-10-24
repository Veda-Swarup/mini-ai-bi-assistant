"""
service.py
----------
Orchestrates the full ask() pipeline.  No FastAPI imports here.

Flow
────
  ask(question)
    ├─ _validate_input          → raises InvalidQuestion on bad input
    ├─ generate_sql             → LLM: question → {supported, sql, reason}
    ├─ if not supported         → return status="unsupported"
    ├─ _execute_with_retry      → validate_sql → run_query
    │    └─ on UnsafeSQL / sqlite3.Error: retry once with error attached
    ├─ if no rows               → return status="no_data"
    ├─ _build_answer            → LLM one-sentence summary + number verification
    │    └─ if numbers invented → _deterministic_answer (template fallback)
    └─ return status="ok"

Custom exceptions
─────────────────
  InvalidQuestion   – bad input (empty / too long); API layer returns 422
  UnsupportedQuestion – used internally when LLM refuses mid-retry
"""

from __future__ import annotations

import re
import sqlite3

from app.db import run_query
from app.guard import UnsafeSQL, validate_sql
from app.llm import LLMError, generate_answer, generate_sql

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_QUESTION_LEN: int = 300   # characters; questions longer than this are rejected

# Matches integers and decimals, with optional leading minus and comma formatting.
# Examples matched: 1439200.0  -4.56  1,439,200  200
_NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class InvalidQuestion(Exception):
    """Raised when the question fails basic input validation.
    HTTP layer should surface this as 422 Unprocessable Entity."""


class UnsupportedQuestion(Exception):
    """Raised internally when the LLM refuses mid-retry.
    Signals the caller that the question cannot be answered."""


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_input(question: str) -> None:
    """
    Raise InvalidQuestion if the question is empty or too long.
    Max length prevents prompt-injection attempts that rely on huge inputs.
    """
    stripped = question.strip()
    if not stripped:
        raise InvalidQuestion("Question must not be empty.")
    if len(stripped) > _MAX_QUESTION_LEN:
        raise InvalidQuestion(
            f"Question is too long ({len(stripped)} chars). "
            f"Maximum is {_MAX_QUESTION_LEN} characters."
        )


# ---------------------------------------------------------------------------
# SQL execution with one retry on failure
# ---------------------------------------------------------------------------

def _execute_with_retry(
    question: str,
    first_result: dict,
) -> tuple[list[str], list[tuple], str]:
    """
    Validate and run the SQL from first_result.

    If validate_sql or run_query raises on the first attempt, the error is
    passed back to generate_sql so the LLM can self-repair its SQL.
    The second attempt's exceptions propagate to the caller.

    Returns
    -------
    (columns, rows, validated_sql)

    Raises
    ------
    UnsupportedQuestion : LLM refused to answer after seeing the error
    UnsafeSQL           : second attempt also produced invalid SQL
    sqlite3.Error       : second attempt also failed at the DB layer
    LLMError            : retry call to the LLM failed
    """
    error_msg: str | None = None

    # ── First attempt ────────────────────────────────────────────────────────
    try:
        safe_sql = validate_sql(first_result["sql"])
        columns, rows = run_query(safe_sql)
        return columns, rows, safe_sql
    except (UnsafeSQL, sqlite3.Error) as exc:
        # Capture the error; we will ask the LLM to repair the SQL.
        error_msg = str(exc)

    # ── Retry: feed error back to LLM ────────────────────────────────────────
    retry_result = generate_sql(question, error=error_msg)

    if not retry_result["supported"]:
        # LLM decided it cannot answer after seeing the error — propagate up.
        raise UnsupportedQuestion(retry_result["reason"])

    # ── Second attempt — let any exception propagate ──────────────────────────
    safe_sql = validate_sql(retry_result["sql"])
    columns, rows = run_query(safe_sql)
    return columns, rows, safe_sql


# ---------------------------------------------------------------------------
# Answer generation with number verification
# ---------------------------------------------------------------------------

def _verify_numbers(sentence: str, rows: list[tuple]) -> bool:
    """
    Check that every number in `sentence` can be traced to a value in `rows`.

    Approach
    ────────
    1. Stringify all row values into one blob (removing commas).
    2. Extract candidate numbers from the sentence (removing commas).
    3. Each sentence number must appear as a substring of the blob OR
       as its absolute value (LLM may rephrase '-10700' as 'decreased by 10700').

    This prevents the LLM from hallucinating figures that were never in
    the database result.
    """
    # Build a searchable string from all row values (no commas, lowercased)
    row_blob: str = " ".join(
        str(val).replace(",", "")
        for row in rows
        for val in row
        if val is not None
    )

    raw_nums = _NUM_RE.findall(sentence)
    if not raw_nums:
        return True     # sentence has no numbers — nothing to verify

    for raw in raw_nums:
        normalized = raw.replace(",", "")           # e.g. "1,439,200" → "1439200"
        abs_normalized = normalized.lstrip("-")     # e.g. "-10700" → "10700"

        # Accept if either the signed or unsigned form appears in the row data.
        if normalized not in row_blob and abs_normalized not in row_blob:
            return False

    return True


def _deterministic_answer(columns: list[str], rows: list[tuple]) -> str:
    """
    Build a plain, template-based answer from columns and rows.
    Used as a fallback when the LLM's answer fails number verification
    or when the LLM call itself fails.

    Format
    ──────
    • 1 row  → "Result: col1=val1, col2=val2, ..."
    • N rows → "N results found. Top result: col1=val1, col2=val2, ..."
    """
    def _fmt_row(row: tuple) -> str:
        return ", ".join(f"{col}: {val}" for col, val in zip(columns, row))

    if len(rows) == 1:
        return f"Result — {_fmt_row(rows[0])}."
    return f"{len(rows)} results found. Top result — {_fmt_row(rows[0])}."


def _build_answer(
    question: str,
    columns: list[str],
    rows: list[tuple],
) -> str:
    """
    Ask the LLM for a one-sentence answer, then verify its numbers.
    Falls back to the deterministic template if:
      - the LLM answer contains a number not in the rows (hallucination), or
      - the LLM call itself fails (LLMError).
    """
    try:
        llm_sentence = generate_answer(question, columns, rows)
        if _verify_numbers(llm_sentence, rows):
            return llm_sentence
        # LLM invented a number — use the safe fallback.
    except LLMError:
        # Answer-generation call failed — use the safe fallback.
        pass

    return _deterministic_answer(columns, rows)


# ---------------------------------------------------------------------------
# Helper: build a consistent error-status response dict
# ---------------------------------------------------------------------------

def _error_response(message: str, error_type: str = "server") -> dict:
    """
    Build a status='error' response dict.

    error_type
    ----------
    'llm'    → main.py maps this to HTTP 502 Bad Gateway (upstream AI failure)
    'server' → main.py maps this to HTTP 500 Internal Server Error
    """
    return {
        "answer":    f"An error occurred: {message}",
        "sql":       None,
        "columns":   [],
        "rows":      [],
        "status":    "error",
        "error_type": error_type,   # consumed by main.py; not surfaced in API docs
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(question: str) -> dict:
    """
    Translate a natural-language question into a data-backed answer.

    Parameters
    ----------
    question : the user's business question (natural language)

    Returns
    -------
    {
      "answer"  : str              — one-sentence natural-language answer
      "sql"     : str | None       — the validated SQL that was executed
      "columns" : list[str]        — column names of the result
      "rows"    : list[list]       — result rows (JSON-serialisable)
      "status"  : str              — "ok" | "unsupported" | "no_data" | "error"
    }

    Raises
    ------
    InvalidQuestion : if the question is empty or exceeds _MAX_QUESTION_LEN.
                      The HTTP layer should handle this as a 422.
    """
    # ── 1. Input validation ───────────────────────────────────────────────────
    _validate_input(question)     # raises InvalidQuestion — let it propagate

    # ── 2. LLM: question → SQL ───────────────────────────────────────────────
    try:
        llm_result = generate_sql(question)
    except LLMError as exc:
        return _error_response(str(exc), error_type="llm")

    # ── 3. Unsupported question ───────────────────────────────────────────────
    if not llm_result["supported"]:
        return {
            "answer":  llm_result["reason"],
            "sql":     None,
            "columns": [],
            "rows":    [],
            "status":  "unsupported",
        }

    # ── 4. Validate + run (with one retry on guard/DB error) ──────────────────
    try:
        columns, rows, safe_sql = _execute_with_retry(question, llm_result)
    except UnsupportedQuestion as exc:
        return {
            "answer":  str(exc),
            "sql":     None,
            "columns": [],
            "rows":    [],
            "status":  "unsupported",
        }
    except (UnsafeSQL, sqlite3.Error, LLMError) as exc:
        # Differentiate LLM errors (502) from guard/DB errors (500)
        error_type = "llm" if isinstance(exc, LLMError) else "server"
        return _error_response(str(exc), error_type=error_type)

    # ── 5. No rows returned ───────────────────────────────────────────────────
    if not rows:
        return {
            "answer":  "No records in the dataset match your question.",
            "sql":     safe_sql,
            "columns": columns,
            "rows":    [],
            "status":  "no_data",
        }

    # ── 6. Build answer (LLM summary + number verification + fallback) ────────
    answer = _build_answer(question, columns, rows)

    # ── 7. Return ─────────────────────────────────────────────────────────────
    return {
        "answer":  answer,
        "sql":     safe_sql,
        "columns": columns,
        "rows":    [list(r) for r in rows],   # tuples → lists for JSON
        "status":  "ok",
    }
