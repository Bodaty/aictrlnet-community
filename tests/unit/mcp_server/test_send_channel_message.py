"""send_channel_message delivers to the caller's own linked account.

It used to run a conversation turn on the caller's web session and deliver
nothing to the channel; it now sends through the channel adapter and reports
a failed delivery as an error instead of success.
"""
import pytest

from mcp_server import tool_executor
from mcp_server.tool_executor import ToolExecutionError, _handle_send_channel_message
from services import channel_outbound
from services.channel_outbound import Delivery


@pytest.fixture
def linked(monkeypatch):
    async def channels(args, db, user_id):
        return {"channels": [{"channel_type": "telegram", "channel_user_id": "111"},
                             {"channel_type": "telegram", "channel_user_id": "222"}]}

    monkeypatch.setattr(tool_executor, "_handle_list_linked_channels", channels)
    sent = []

    async def record(channel_type, recipient_id, text, platform_metadata=None):
        sent.append((channel_type, recipient_id, text))
        return Delivery(True)

    monkeypatch.setattr(channel_outbound, "send_text", record)
    return sent


@pytest.mark.asyncio
async def test_sends_to_the_first_linked_account_by_default(linked):
    result = await _handle_send_channel_message({"channel_type": "Telegram", "message": "hi"}, None, "u1")
    assert linked == [("telegram", "111", "hi")]
    assert result == {"delivered": True, "channel_type": "telegram", "channel_user_id": "111"}


@pytest.mark.asyncio
async def test_a_named_recipient_must_be_one_of_the_callers_links(linked):
    await _handle_send_channel_message(
        {"channel_type": "telegram", "message": "hi", "channel_user_id": "222"}, None, "u1")
    with pytest.raises(ToolExecutionError, match="not one of your linked"):
        await _handle_send_channel_message(
            {"channel_type": "telegram", "message": "hi", "channel_user_id": "999"}, None, "u1")
    assert linked == [("telegram", "222", "hi")]


@pytest.mark.asyncio
async def test_a_failed_delivery_is_an_error(linked, monkeypatch):
    async def fail(*a, **k):
        return Delivery(False, "the telegram adapter is not registered")

    monkeypatch.setattr(channel_outbound, "send_text", fail)
    with pytest.raises(ToolExecutionError, match="not registered"):
        await _handle_send_channel_message({"channel_type": "telegram", "message": "hi"}, None, "u1")


@pytest.mark.asyncio
async def test_unsupported_channel_is_not_delivered():
    delivery = await channel_outbound.send_text("fax", "1", "hi")
    assert delivery == Delivery(False, "unsupported channel 'fax'")
