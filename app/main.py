"""
main.py
-------
FastAPI application entry point.

Routes
──────
  POST /api/ask  – main endpoint: NL question → answer + SQL + rows
  GET  /health   – liveness probe (no DB or LLM check)
  GET  /         – serves static/index.html

HTTP status code mapping
────────────────────────
  200 – ok | no_data   (answer field explains what was found)
  400 – unsupported    (question can't be answered from this dataset)
  422 – invalid input  (empty / > 300 chars; Pydantic or InvalidQuestion)
  502 – LLM failure    (Gemini API unreachable or misconfigured)
  500 – unexpected     (logged in full; only a generic message is returned)

Logging
───────
  Logs: question text, guard verdict (pass/fail implied by SQL presence),
        validated SQL, result status, latency.
  Never logs: API keys, environment variables, raw stack traces to the user.

Run
───
  uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app import service
from app.llm import LLMError
from app.service import InvalidQuestion

# ---------------------------------------------------------------------------
# Logging — configured before anything else runs
# ---------------------------------------------------------------------------
# Structured one-line format makes it easy to grep in CI logs.
# IMPORTANT: never pass os.environ or api_key into log calls.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("bi_assistant")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_STATIC_DIR = Path(__file__).parent.parent / "static"
_INDEX_HTML  = _STATIC_DIR / "index.html"


# ---------------------------------------------------------------------------
# Lifespan — auto-run ETL on first start if the DB is missing
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(application: FastAPI):           # noqa: ARG001
    """
    Startup: run ETL if the database file does not yet exist.
    This means `uvicorn app.main:app` is the only command needed after
    setting up .env — no separate `python -m app.etl` step required.
    """
    from app.etl import DB_PATH, load  # local import avoids circular deps

    if not DB_PATH.exists():
        log.info("Database not found at %s — running ETL now.", DB_PATH)
        loaded, rejected = load()
        log.info("ETL complete: loaded=%d  rejected=%d", loaded, rejected)
    else:
        log.info("Database found at %s — skipping ETL.", DB_PATH)

    yield                                            # app runs here
    # Shutdown: nothing to clean up for SQLite


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Mini AI BI Assistant",
    description=(
        "Ask natural-language questions about sales data. "
        "The LLM translates each question to a SQLite SELECT on v_revenue; "
        "the guard validates the SQL before execution."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Incoming request body for POST /api/ask."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Natural-language question about sales data.",
        examples=["Which product generated the highest revenue?"],
    )


class AskResponse(BaseModel):
    """
    Successful response from POST /api/ask.
    `rows` is a list of lists so it is always JSON-serialisable.
    """
    answer:  str
    sql:     str | None
    columns: list[str]
    rows:    list[list]
    status:  str


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def pydantic_validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Override FastAPI's default 422 handler so the response shape matches
    our other error responses ({"detail": "..."}).
    Triggered when Pydantic rejects the request body (e.g. question too long).
    """
    # Log the error but not the full body (could contain large payloads)
    log.warning("VALIDATION_ERROR | path=%s errors=%s", request.url.path, exc.errors())
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(Exception)
async def unexpected_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Catch-all for any exception that escapes the endpoint function.
    Logs the full traceback; returns only a generic message to the client
    so internal details are never leaked.
    """
    log.exception(
        "UNEXPECTED_ERROR | method=%s path=%s",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"], summary="Liveness probe")
async def health() -> dict:
    """
    Returns 200 as long as the process is running.
    Does NOT check the database or LLM — use this only for process-level
    health checks (e.g. a load-balancer ping).
    """
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the single-page HTML UI."""
    return FileResponse(_INDEX_HTML)


@app.post(
    "/api/ask",
    response_model=AskResponse,
    tags=["ask"],
    summary="Ask a natural-language question about sales data",
    responses={
        400: {"description": "Question not answerable from this dataset"},
        422: {"description": "Invalid input (empty or > 300 chars)"},
        502: {"description": "LLM service failure"},
        500: {"description": "Unexpected server error"},
    },
)
def ask_endpoint(body: AskRequest) -> JSONResponse:
    """
    Pipeline
    ────────
    1. Validate input (Pydantic → 422 before we're even called)
    2. Call LLM to generate SQL
    3. Guard validates SQL (allowlist + sqlglot parse)
    4. Execute on read-only SQLite connection
    5. LLM produces one-sentence answer; numbers verified against rows
    6. Return answer + SQL + supporting rows

    Note: Defined as synchronous `def` (not `async def`) so FastAPI automatically
    runs it in a worker threadpool, preventing blocking of the asyncio event loop.
    """
    t_start = time.perf_counter()
    question = body.question.strip()

    # Log the question.  Never log os.environ or any key-like string.
    log.info("REQUEST | question=%r", question)

    # ── Call service layer ─────────────────────────────────────────────────
    try:
        result = service.ask(question)
    except InvalidQuestion as exc:
        # Whitespace-only question passes Pydantic (min_length=1) but is
        # caught by service._validate_input.
        log.warning("INVALID_QUESTION | reason=%s", exc)
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    # ── Log outcome ────────────────────────────────────────────────────────
    latency_ms = round((time.perf_counter() - t_start) * 1000, 1)
    sql_snippet = (result.get("sql") or "")[:120]   # truncate long CTEs
    log.info(
        "RESPONSE | status=%s  guard=%s  sql=%r  latency_ms=%s",
        result["status"],
        "pass" if result.get("sql") else "n/a",   # guard passed iff sql is present
        sql_snippet,
        latency_ms,
    )

    # ── Map internal status → HTTP code ────────────────────────────────────
    status = result["status"]

    if status == "unsupported":
        # The question is valid but cannot be answered from v_revenue.
        return JSONResponse(
            status_code=400,
            content={
                "answer":  result["answer"],
                "sql":     None,
                "columns": [],
                "rows":    [],
                "status":  "unsupported",
            },
        )

    if status == "error":
        # error_type="llm"  → upstream AI failure → 502 Bad Gateway
        # error_type="server" → guard/DB failure  → 500 Internal Server Error
        http_code = 502 if result.get("error_type") == "llm" else 500
        log.error(
            "SERVICE_ERROR | error_type=%s  answer=%r",
            result.get("error_type"),
            result["answer"],
        )
        return JSONResponse(
            status_code=http_code,
            content={
                "answer":  result["answer"],
                "sql":     None,
                "columns": [],
                "rows":    [],
                "status":  "error",
            },
        )

    # status == "ok" or "no_data" → 200
    # Strip the internal error_type key before sending to the client.
    clean = {k: v for k, v in result.items() if k != "error_type"}
    return JSONResponse(status_code=200, content=clean)
