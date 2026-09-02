"""Cross-cutting HTTP behaviour: headers, correlation ids, rate limiting, metrics."""

from __future__ import annotations

import json
import logging

import pytest

from app import config, middleware, observability, security


# ------------------------------------------------------------------- security headers
def test_every_response_carries_the_hardening_headers(client):
    headers = client.get("/api/v1/health").headers
    assert "default-src 'self'" in headers["content-security-policy"]
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["referrer-policy"] == "strict-origin-when-cross-origin"


def test_error_responses_are_hardened_too(client):
    """A 404 rendered without CSP is still a page an attacker can work with."""
    assert "content-security-policy" in client.get("/api/v1/meetings/nope").headers


def test_static_assets_are_hardened_too(client):
    assert "content-security-policy" in client.get("/static/app.js").headers


def test_hsts_is_absent_on_a_development_server(client):
    assert "strict-transport-security" not in client.get("/api/v1/health").headers


# ----------------------------------------------------------------------- request ids
def test_every_response_gets_a_request_id(client):
    rid = client.get("/api/v1/health").headers["x-request-id"]
    assert rid and rid != "-"


def test_request_ids_are_unique_per_request(client):
    first = client.get("/api/v1/health").headers["x-request-id"]
    second = client.get("/api/v1/health").headers["x-request-id"]
    assert first != second


def test_a_caller_supplied_request_id_is_honoured_for_tracing(client):
    response = client.get("/api/v1/health", headers={"X-Request-ID": "trace-abc-123"})
    assert response.headers["x-request-id"] == "trace-abc-123"


@pytest.mark.parametrize(
    "hostile",
    [
        "bad id with spaces",
        "injected\nlevel=ERROR message=fake",  # log-forging attempt
        "x" * 200,  # unbounded length
        "../../etc/passwd",
    ],
)
def test_a_hostile_request_id_is_replaced_not_echoed(client, hostile):
    """The id lands in log files, so anything outside a safe charset is discarded."""
    response = client.get("/api/v1/health", headers={"X-Request-ID": hostile})
    assert response.headers["x-request-id"] != hostile


def test_the_error_envelope_carries_the_same_request_id_as_the_header(client):
    response = client.get("/api/v1/meetings/nope")
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


def test_responses_report_their_own_latency(client):
    assert float(client.get("/api/v1/health").headers["x-response-time-ms"]) >= 0


# ------------------------------------------------------------------------- rate limiting
@pytest.fixture
def rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(config, "RATE_LIMIT_DEFAULT", 3)
    monkeypatch.setattr(config, "RATE_LIMIT_WINDOW_SECONDS", 60)
    security.limiter.reset()


def test_requests_beyond_the_budget_get_429(client, rate_limited):
    for _ in range(3):
        assert client.get("/api/v1/meetings").status_code == 200

    response = client.get("/api/v1/meetings")
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limited"


def test_a_429_tells_the_client_when_to_come_back(client, rate_limited):
    for _ in range(4):
        response = client.get("/api/v1/meetings")
    assert int(response.headers["retry-after"]) >= 1
    assert response.headers["x-ratelimit-remaining"] == "0"


def test_remaining_budget_is_advertised_on_every_response(client, rate_limited):
    assert client.get("/api/v1/meetings").headers["x-ratelimit-remaining"] == "2"
    assert client.get("/api/v1/meetings").headers["x-ratelimit-remaining"] == "1"


def test_budgets_are_tracked_per_endpoint(client, rate_limited):
    """Exhausting one endpoint must not lock a client out of the whole API."""
    for _ in range(4):
        client.get("/api/v1/meetings")
    assert client.get("/api/v1/rag/status").status_code == 200


def test_health_and_metrics_are_never_rate_limited(client, rate_limited):
    """A load balancer polls health constantly; throttling it takes the service down."""
    for _ in range(10):
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/metrics").status_code == 200


def test_a_429_still_carries_security_headers_and_a_request_id(client, rate_limited):
    for _ in range(4):
        response = client.get("/api/v1/meetings")
    assert "content-security-policy" in response.headers
    assert response.json()["error"]["request_id"]


def test_the_limiter_is_off_when_configured_off(client):
    for _ in range(30):
        assert client.get("/api/v1/meetings").status_code == 200


# ----------------------------------------------------------------------- client identity
def test_forwarded_headers_are_ignored_unless_a_proxy_is_trusted(monkeypatch):
    """X-Forwarded-For is spoofable, so trusting it by default would let one client
    mint unlimited identities and bypass the limiter entirely."""
    request = _fake_request(client_host="10.0.0.1", forwarded="1.2.3.4")

    monkeypatch.setattr(config, "TRUST_PROXY_HEADERS", False)
    assert middleware.client_key(request) == "10.0.0.1"

    monkeypatch.setattr(config, "TRUST_PROXY_HEADERS", True)
    assert middleware.client_key(request) == "1.2.3.4"


def test_the_first_hop_is_used_from_a_forwarded_chain(monkeypatch):
    monkeypatch.setattr(config, "TRUST_PROXY_HEADERS", True)
    request = _fake_request(client_host="10.0.0.1", forwarded="1.2.3.4, 5.6.7.8")
    assert middleware.client_key(request) == "1.2.3.4"


def _fake_request(*, client_host: str, forwarded: str):
    from starlette.requests import Request

    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"x-forwarded-for", forwarded.encode())],
            "client": (client_host, 12345),
            "query_string": b"",
        }
    )


# ------------------------------------------------------------------------------ metrics
def test_metrics_expose_request_counts_and_latency(client):
    client.get("/api/v1/health")
    body = client.get("/metrics").text

    assert "meetsaransh_http_requests_total" in body
    assert "meetsaransh_http_request_duration_seconds" in body
    assert 'route="/api/v1/health"' in body


def test_metrics_label_on_the_route_template_not_the_raw_path(client):
    """Labelling on the path would mint one time series per meeting id."""
    mid = client.post("/api/v1/meetings/sample").json()["id"]
    client.get(f"/api/v1/meetings/{mid}")
    body = client.get("/metrics").text

    assert 'route="/api/v1/meetings/{meeting_id}"' in body
    assert mid not in body


def test_stored_counts_are_gauged(client):
    client.post("/api/v1/meetings/sample")
    body = client.get("/metrics").text
    assert "meetsaransh_meetings_stored 1.0" in body


def test_rate_limit_rejections_are_counted(client, rate_limited):
    for _ in range(5):
        client.get("/api/v1/meetings")
    assert "meetsaransh_rate_limited_total" in client.get("/metrics").text


def test_metrics_can_be_switched_off(client, monkeypatch):
    monkeypatch.setattr(config, "METRICS_ENABLED", False)
    assert client.get("/metrics").status_code == 404


# ------------------------------------------------------------------------------ logging
def test_json_logs_are_one_parseable_object_per_line():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="a message",
        args=(),
        exc_info=None,
    )
    record.request_id = "abc123"
    record.status = 200

    payload = json.loads(observability.JsonFormatter().format(record))
    assert payload["message"] == "a message"
    assert payload["request_id"] == "abc123"
    assert payload["status"] == 200
    assert payload["level"] == "INFO"


def test_json_logs_include_the_traceback_on_an_exception():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )
    payload = json.loads(observability.JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exception"]


def test_console_logs_stay_on_one_line_with_their_context():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request",
        args=(),
        exc_info=None,
    )
    record.request_id = "abc123"
    record.status = 404

    line = observability.ConsoleFormatter().format(record)
    assert "[abc123]" in line
    assert "status=404" in line
    assert "\n" not in line


def test_a_rate_limited_request_is_attributed_to_its_real_route(client, rate_limited):
    """A 429 short-circuits before routing, but the metric must still name the endpoint
    being hammered -- otherwise the counter cannot answer the only question it exists
    to answer."""
    for _ in range(5):
        client.get("/api/v1/meetings")
    body = client.get("/metrics").text

    assert 'meetsaransh_rate_limited_total{route="/api/v1/meetings"}' in body
    assert 'meetsaransh_rate_limited_total{route="unmatched"}' not in body


def test_static_assets_share_one_time_series(client):
    client.get("/static/app.js")
    client.get("/static/style.css")
    body = client.get("/metrics").text

    assert 'route="/static"' in body
    assert "app.js" not in body  # not one series per file


def test_a_405_is_attributed_to_the_endpoint_that_was_aimed_at(client):
    client.put("/api/v1/health")
    body = client.get("/metrics").text
    assert 'route="/api/v1/health",status="405"' in body


def test_a_genuinely_unknown_path_is_labelled_unmatched(client):
    client.get("/no/such/thing")
    assert 'route="unmatched"' in client.get("/metrics").text
