"""No identifier reaches the log through httpx's own request line (platform-wide)."""

import logging

import httpx
import pytest

from core import http_log_redaction


@pytest.mark.asyncio
async def test_a_request_logs_its_path_and_status_but_not_its_query(caplog):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"resourceType": "Bundle"})
    )
    async with httpx.AsyncClient(transport=transport, base_url="https://fhir.example.test/r4/") as http:
        with caplog.at_level(logging.INFO, logger="httpx"):
            await http.get("Observation", params={"patient": "pt-0042", "_count": "100"})

    lines = [r.getMessage() for r in caplog.records if r.name == "httpx"]
    assert lines, "httpx emitted no request line"
    assert all("pt-0042" not in line and "patient=" not in line for line in lines)
    assert any("https://fhir.example.test/r4/Observation?<redacted>" in line for line in lines)
    assert any("200" in line for line in lines)


def test_installing_twice_adds_one_filter():
    http_log_redaction.install()
    http_log_redaction.install()
    filters = [
        f for f in logging.getLogger("httpx").filters
        if isinstance(f, http_log_redaction.RedactHttpxQuery)
    ]
    assert len(filters) == 1


def test_a_url_without_a_query_is_left_alone():
    record = logging.LogRecord(
        "httpx", logging.INFO, __file__, 1, 'HTTP Request: %s %s "%s %d %s"',
        ("GET", "https://fhir.example.test/r4/metadata", "HTTP/1.1", 200, "OK"), None,
    )
    http_log_redaction.RedactHttpxQuery().filter(record)
    assert record.getMessage() == 'HTTP Request: GET https://fhir.example.test/r4/metadata "HTTP/1.1 200 OK"'


def test_the_app_installs_it():
    import core.app  # noqa: F401
    assert any(isinstance(f, http_log_redaction.RedactHttpxQuery)
               for f in logging.getLogger("httpx").filters)
