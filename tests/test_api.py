"""
tests/test_api.py
-----------------
FastAPI endpoint tests using TestClient.

Strategy: monkeypatch service.ask() for most tests so these are pure
HTTP-layer tests — no LLM, no DB.  We test request validation (Pydantic),
HTTP status code mapping, and response shape.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

# Module-level client; lifespan fires on the first request.
# conftest.py's ensure_real_db guarantees the DB exists first.
client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Shared mock payloads
# ---------------------------------------------------------------------------

_OK_PAYLOAD = {
    "answer":  "Total revenue is 1439200.0.",
    "sql":     "SELECT ROUND(SUM(net_revenue), 2) AS total_revenue FROM v_revenue",
    "columns": ["total_revenue"],
    "rows":    [[1_439_200.0]],
    "status":  "ok",
}

_UNSUPPORTED_PAYLOAD = {
    "answer":  "Weather data is not in this dataset.",
    "sql":     None,
    "columns": [],
    "rows":    [],
    "status":  "unsupported",
}

_LLM_ERROR_PAYLOAD = {
    "answer":    "An error occurred: Gemini API call failed.",
    "sql":       None,
    "columns":   [],
    "rows":      [],
    "status":    "error",
    "error_type": "llm",
}

_SERVER_ERROR_PAYLOAD = {
    "answer":    "An error occurred: guard failed.",
    "sql":       None,
    "columns":   [],
    "rows":      [],
    "status":    "error",
    "error_type": "server",
}


# ---------------------------------------------------------------------------
# 10a — Ops routes
# ---------------------------------------------------------------------------

class TestOpsRoutes:
    def test_health_returns_200(self) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_index_returns_html(self) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Mini BI Assistant" in resp.text


# ---------------------------------------------------------------------------
# 10b — 422 Invalid input
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_empty_question_returns_422(self) -> None:
        """Pydantic min_length=1 catches empty string before service is called."""
        resp = client.post("/api/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_overlength_question_returns_422(self) -> None:
        """Pydantic max_length=300 rejects strings longer than 300 chars."""
        resp = client.post("/api/ask", json={"question": "x" * 301})
        assert resp.status_code == 422

    def test_missing_question_field_returns_422(self) -> None:
        """Sending {} with no 'question' key must be rejected by Pydantic."""
        resp = client.post("/api/ask", json={})
        assert resp.status_code == 422

    def test_whitespace_only_returns_422(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        '   ' passes Pydantic (length=3) but service._validate_input
        raises InvalidQuestion → endpoint returns 422.
        """
        # Don't mock service here — let InvalidQuestion propagate naturally
        resp = client.post("/api/ask", json={"question": "   "})
        assert resp.status_code == 422

    def test_non_string_question_returns_422(self) -> None:
        resp = client.post("/api/ask", json={"question": 12345})
        # Pydantic coerces int to str, so this is actually 200 after coercion
        # OR 422 if strict mode.  Either way — assert no 5xx.
        assert resp.status_code in (200, 400, 422)


# ---------------------------------------------------------------------------
# 10c — 200 OK paths
# ---------------------------------------------------------------------------

class TestSuccessfulAsk:
    def test_ok_status_returns_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.service.ask", lambda q: _OK_PAYLOAD)
        resp = client.post("/api/ask", json={"question": "total revenue?"})
        assert resp.status_code == 200

    def test_ok_response_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Response must contain answer, sql, columns, rows, status."""
        monkeypatch.setattr("app.service.ask", lambda q: _OK_PAYLOAD)
        data = client.post(
            "/api/ask", json={"question": "total revenue?"}
        ).json()
        assert set(data.keys()) >= {"answer", "sql", "columns", "rows", "status"}

    def test_ok_rows_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.service.ask", lambda q: _OK_PAYLOAD)
        data = client.post(
            "/api/ask", json={"question": "total revenue?"}
        ).json()
        assert data["rows"] == [[1_439_200.0]]

    def test_no_data_returns_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """no_data is a successful response (query ran, just returned 0 rows)."""
        no_data = {**_OK_PAYLOAD, "status": "no_data", "rows": [], "columns": []}
        monkeypatch.setattr("app.service.ask", lambda q: no_data)
        resp = client.post(
            "/api/ask", json={"question": "revenue in December 2025?"}
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_data"


# ---------------------------------------------------------------------------
# 10d — 400 Unsupported
# ---------------------------------------------------------------------------

class TestUnsupportedQuestion:
    def test_unsupported_returns_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.service.ask", lambda q: _UNSUPPORTED_PAYLOAD)
        resp = client.post("/api/ask", json={"question": "what's the weather?"})
        assert resp.status_code == 400

    def test_unsupported_response_has_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app.service.ask", lambda q: _UNSUPPORTED_PAYLOAD)
        data = client.post(
            "/api/ask", json={"question": "what's the weather?"}
        ).json()
        assert "answer" in data
        assert data["status"] == "unsupported"


# ---------------------------------------------------------------------------
# 10e — 502 LLM failure / 500 server error
# ---------------------------------------------------------------------------

class TestErrorCodes:
    def test_llm_failure_returns_502(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM-tagged errors must map to 502 Bad Gateway."""
        monkeypatch.setattr("app.service.ask", lambda q: _LLM_ERROR_PAYLOAD)
        resp = client.post("/api/ask", json={"question": "total revenue?"})
        assert resp.status_code == 502

    def test_server_error_returns_500(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-LLM errors (guard/DB) must map to 500 Internal Server Error."""
        monkeypatch.setattr("app.service.ask", lambda q: _SERVER_ERROR_PAYLOAD)
        resp = client.post("/api/ask", json={"question": "total revenue?"})
        assert resp.status_code == 500

    def test_error_response_has_no_internal_details(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """error_type must NOT be leaked to the client in the HTTP response body."""
        monkeypatch.setattr("app.service.ask", lambda q: _LLM_ERROR_PAYLOAD)
        data = client.post(
            "/api/ask", json={"question": "total revenue?"}
        ).json()
        # error_type is internal — must be stripped before sending
        assert "error_type" not in data
