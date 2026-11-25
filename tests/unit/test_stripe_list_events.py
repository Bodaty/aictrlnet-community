"""Stripe adapter `list_events`: cursor pagination over /v1/events, no network.

/v1/events is newest first. With ending_before=<id> Stripe returns events newer
than <id>; the adapter pages by the first (newest) id of each page and returns the
events oldest-first with the newest id as the next cursor.
"""
import httpx
import pytest

from adapters.implementations.payment.stripe_adapter import StripeAdapter
from adapters.models import AdapterCategory, AdapterConfig, AdapterRequest


def _evt(n):
    return {"id": f"evt_{n:04d}", "type": "invoice.paid", "created": 1_700_000_000 + n,
            "data": {"object": {"n": n}}}


def _adapter(pages):
    """Adapter whose client replays `pages` (list of (events_newest_first, has_more))."""
    sent = []
    queue = list(pages)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        events, has_more = queue.pop(0)
        return httpx.Response(200, json={"object": "list", "data": events, "has_more": has_more})

    adapter = StripeAdapter(AdapterConfig(name="stripe", category=AdapterCategory.PAYMENT, api_key="rk_test_x"))
    adapter.client = httpx.AsyncClient(base_url="https://api.stripe.com/v1",
                                       transport=httpx.MockTransport(handler))
    return adapter, sent


def _req(**params):
    return AdapterRequest(capability="list_events", parameters=params)


@pytest.mark.asyncio
async def test_pages_newer_than_cursor_and_returns_oldest_first():
    # cursor evt_0002; page 1 = the 3 nearest newer (newest-first), page 2 = the rest
    adapter, sent = _adapter([
        ([_evt(5), _evt(4), _evt(3)], True),
        ([_evt(7), _evt(6)], False),
    ])
    resp = await adapter.execute(_req(ending_before="evt_0002", limit=3))
    assert resp.status == "success"
    assert [e["id"] for e in resp.data["events"]] == [f"evt_{n:04d}" for n in (3, 4, 5, 6, 7)]
    assert resp.data["newest_id"] == "evt_0007"
    assert sent[0].url.params["ending_before"] == "evt_0002"
    assert sent[1].url.params["ending_before"] == "evt_0005"  # first (newest) id of page 1
    assert "starting_after" not in sent[1].url.params
    assert set(resp.data["events"][0]) == {"id", "type", "created", "data"}


@pytest.mark.asyncio
async def test_types_are_repeated_params_and_created_gte_is_forwarded():
    adapter, sent = _adapter([([], False)])
    await adapter.execute(_req(types=["invoice.paid", "checkout.session.completed"], created_gte=1_700_000_000))
    url = sent[0].url
    assert url.path == "/v1/events"
    assert url.params.get_list("types[]") == ["invoice.paid", "checkout.session.completed"]
    assert url.params["created[gte]"] == "1700000000"
    assert url.params["limit"] == "100"
    assert "ending_before" not in url.params


@pytest.mark.asyncio
async def test_no_events_returns_the_input_cursor():
    adapter, _ = _adapter([([], False)])
    resp = await adapter.execute(_req(ending_before="evt_0009"))
    assert resp.data == {"events": [], "newest_id": "evt_0009"}
    adapter, _ = _adapter([([], False)])
    resp = await adapter.execute(_req())
    assert resp.data == {"events": [], "newest_id": None}


@pytest.mark.asyncio
async def test_stops_at_one_thousand_events_even_if_has_more():
    pages = [([_evt(n) for n in range(i * 100 + 100, i * 100, -1)], True) for i in range(11)]
    adapter, sent = _adapter(pages)
    resp = await adapter.execute(_req(ending_before="evt_0000"))
    assert len(sent) == 10
    assert len(resp.data["events"]) == 1000
    assert resp.data["newest_id"] == "evt_1000"
    assert resp.data["events"][0]["id"] == "evt_0001" and resp.data["events"][-1]["id"] == "evt_1000"


@pytest.mark.asyncio
async def test_without_cursor_walks_older_pages_and_returns_oldest_first():
    adapter, sent = _adapter([
        ([_evt(9), _evt(8)], True),
        ([_evt(7)], False),
    ])
    resp = await adapter.execute(_req(limit=2))
    assert [e["id"] for e in resp.data["events"]] == ["evt_0007", "evt_0008", "evt_0009"]
    assert resp.data["newest_id"] == "evt_0009"
    assert sent[1].url.params["starting_after"] == "evt_0008"


def _failing(status, body):
    adapter = StripeAdapter(AdapterConfig(name="stripe", category=AdapterCategory.PAYMENT, api_key="rk_test_x"))
    adapter.client = httpx.AsyncClient(
        base_url="https://api.stripe.com/v1",
        transport=httpx.MockTransport(lambda r: httpx.Response(status, json=body)))
    return adapter


@pytest.mark.asyncio
async def test_http_429_is_an_error_response_carrying_the_status():
    # HTTP_<status>, the form an adapter step's retry_on_error_codes matches.
    adapter = _failing(429, {"error": {"message": "rate", "code": "rate_limit"}})
    resp = await adapter.execute(_req(ending_before="evt_0001"))
    assert resp.status == "error" and resp.error_code == "HTTP_429"


@pytest.mark.asyncio
async def test_a_server_error_carries_its_http_status():
    resp = await _failing(503, {"error": {"message": "down"}}).execute(_req(ending_before="evt_0001"))
    assert resp.status == "error" and resp.error_code == "HTTP_503"


@pytest.mark.asyncio
async def test_an_expired_cursor_carries_stripes_resource_missing_code():
    body = {"error": {"type": "invalid_request_error", "code": "resource_missing",
                      "message": "No such event: 'evt_0001'", "param": "ending_before"}}
    resp = await _failing(400, body).execute(_req(ending_before="evt_0001"))
    assert resp.status == "error" and resp.error_code == "resource_missing"


@pytest.mark.asyncio
async def test_a_refusal_without_a_stripe_code_carries_its_http_status():
    resp = await _failing(401, {"error": {"message": "bad key"}}).execute(_req())
    assert resp.error_code == "HTTP_401"


@pytest.mark.asyncio
async def test_cold_start_hitting_the_cap_is_flagged_truncated():
    pages = [([_evt(n) for n in range(1100 - i * 100, 1000 - i * 100, -1)], True) for i in range(11)]
    adapter, sent = _adapter(pages)
    resp = await adapter.execute(_req())
    assert len(sent) == 10 and len(resp.data["events"]) == 1000
    assert resp.data["truncated"] is True
    assert resp.data["newest_id"] == "evt_1100"


@pytest.mark.asyncio
async def test_cursor_mode_and_complete_pulls_are_not_flagged():
    adapter, _ = _adapter([([_evt(2)], False)])
    assert "truncated" not in (await adapter.execute(_req())).data
