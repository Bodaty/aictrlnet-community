"""Tier 0 — the deterministic router (CONVERSATION_ORCHESTRATION_SPEC.md §7.6).

A labelled corpus of messages real users send, including the exact gate-3
strings. `act` admits write tools; `discover` only discovery/read; `chat`
none. A misroute either withholds the tool a request needs or offers a write
tool to a question — both are regressions.
"""

import time

import pytest

from services.conversation_router import ACT, CHAT, CONFIRM, DISCOVER, ONBOARDING_TOOLS, route

ACT_MESSAGES = [
    "Create a workflow that emails me invoices",
    "email the team the weekly report", "approve request 123", "connect Slack",
    "connect my Slack account", "integrate with Salesforce", "make me an agent that triages tickets",
    "I want to automate invoice processing", "automate my accounting", "upgrade my plan",
    "turn on MFA", "change my password", "reset my password", "pause the invoice workflow",
    "stop the workflow", "activate the onboarding workflow", "remember that I prefer short answers",
    "forget my phone number", "enroll me in the governance module", "upload this file",
    "link my telegram", "unlink my telegram", "test my quickbooks connection", "mark task 42 as done",
    "complete task 42", "assign this ticket to Maria", "send a message to the team",
    "post this on linkedin", "draft an email to my client", "rerun the failed workflow", "retry that",
    "publish the template", "clone this workflow", "share the workflow with Bob", "export my data",
    "book an appointment for tomorrow", "I need a workflow for onboarding new patients",
    "can you set up a reminder every Monday", "disable dry run mode", "delete it", "run it",
    "yes, go ahead and create it", "yes please do that", "sure, go ahead", "do it",
]

DISCOVER_MESSAGES = [
    "Tell me more about your capabilities.", "Test sidebar conversation",
    "how do I create a workflow?", "what happens when I delete an agent?", "can you explain approvals?",
    "who created this workflow?", "list running workflows", "show my scheduled tasks",
    "what changed in my workflow", "any updates on my tasks?",
    "is it possible to send emails from a workflow?",
    "what does the approve button do on an approval request", "why did my workflow fail to run",
    "what is my plan", "how do I connect Slack?", "show me my workflows", "list my agents",
    "list my workflows", "show my schedule", "is my workflow running?", "is my workflow automated?",
    "what are the steps to add a user", "don't delete my workflow",
]

CHAT_MESSAGES = [
    "What can you help me with?", "Hello, who are you?", "hi", "thanks!", "What's new?",
    "good morning, how are you doing?", "yes", "about 12 people", "", "   ", "👍",
]


@pytest.mark.parametrize("message", ACT_MESSAGES)
def test_requests_act(message):
    assert route(message).mode == ACT, route(message).reasons


@pytest.mark.parametrize("message", DISCOVER_MESSAGES)
def test_questions_discover(message):
    assert route(message).mode == DISCOVER, route(message).reasons


@pytest.mark.parametrize("message", CHAT_MESSAGES)
def test_small_talk_chats(message):
    assert route(message).mode == CHAT, route(message).reasons


@pytest.mark.parametrize("marker", ["gate3-reload-marker-1790805644", "gate3-route-marker-1790805644"])
def test_gate3_markers_never_act(marker):
    assert route(marker).mode != ACT


def test_admitted_classes_follow_the_mode():
    assert route("hi").allowed_classes == frozenset()
    assert route("list my workflows").allowed_classes == {"discovery", "read"}
    assert route("delete the api key").allowed_classes == {"discovery", "read", "write", "long_running"}


def test_bare_consent_acts_only_after_the_assistant_asked():
    assert route("yes").mode == CHAT
    assert route("yes", {"assistant_asked": True}).mode == ACT


def test_pending_proposal_confirms_that_tool():
    decision = route("yes", {"pending_proposal": {"tool": "execute_workflow"}})
    assert decision.mode == CONFIRM and decision.proposed_tool == "execute_workflow"
    assert route("go ahead", {"pending_proposal": {"tool": "execute_workflow"}}).mode == CONFIRM


def test_onboarding_admits_its_save_tool_in_every_mode():
    for message in ("about 12 people", "I'm a business owner", "mostly scheduling and billing"):
        assert route(message, {"onboarding_active": True}).extra_tools == ONBOARDING_TOOLS
        assert route(message).extra_tools == frozenset()


def test_non_english_never_acts():
    assert route("Créer un workflow").mode != ACT
    assert route("¿Puedes crear un flujo de trabajo?").mode != ACT


def test_long_input_routes_in_linear_time():
    # route_ms (5) is a measured budget; this guards the clause split against
    # quadratic backtracking (100 ms at 10k spaces before the fix). The API
    # caps content at 10k characters; channels may not.
    for message in (" " * 10_000, "word " * 2_000, "a - " * 2_500):
        started = time.perf_counter()
        route(message)
        assert (time.perf_counter() - started) * 1000 < 25
