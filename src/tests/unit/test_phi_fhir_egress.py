"""R-04 companion: under PHI mode the care-gap FHIR sync may only call an EHR
host the deployment has allowlisted, over https.

covers: r04-fhir-host-allowlist

Settings are constructed per test and passed in explicitly — no env mutation,
so nothing leaks into the other unit tests in this directory. The test egress
guard is default-on here; nothing in this module opens a socket.
"""

from types import SimpleNamespace

import pytest

from core.phi_egress import (
    FHIR_HOSTS_SETTING,
    PHI_REFUSAL_MARKER,
    PHIEgressRefused,
    assert_phi_fhir_host_allowed,
    parse_host_list,
    phi_boot_problems,
    phi_fhir_hosts,
)


def _settings(**overrides):
    base = {
        "AICTRLNET_PHI_MODE": True,
        "AICTRLNET_PHI_FHIR_HOSTS": "api.practicefusion.com",
        "AICTRLNET_PHI_LLM_PROVIDERS": "",
        "AICTRLNET_PHI_BAA_PROVIDERS": "",
        "CARE_GAPS_ENABLED": False,
        "DEFAULT_LLM_MODEL": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# --- parsing -----------------------------------------------------------------


def test_parse_host_list_strips_lowercases_and_drops_empties():
    assert parse_host_list(" API.PracticeFusion.com , ,auth.example.org ") == frozenset(
        {"api.practicefusion.com", "auth.example.org"}
    )
    assert parse_host_list("") == frozenset()
    assert parse_host_list(None) == frozenset()


def test_phi_fhir_hosts_reads_the_setting():
    settings = _settings(AICTRLNET_PHI_FHIR_HOSTS="a.example.com,b.example.com")
    assert phi_fhir_hosts(settings=settings) == frozenset(
        {"a.example.com", "b.example.com"}
    )


# --- runtime guard -----------------------------------------------------------


def test_no_op_when_phi_mode_off():
    settings = _settings(AICTRLNET_PHI_MODE=False, AICTRLNET_PHI_FHIR_HOSTS="")
    assert_phi_fhir_host_allowed(
        "https://anything.example.com/Patient", settings=settings
    )


def test_unlisted_host_is_refused_naming_host_and_setting():
    settings = _settings()
    with pytest.raises(PHIEgressRefused) as excinfo:
        assert_phi_fhir_host_allowed(
            "https://ehr.example.com/fhir/Patient?identifier=secret",
            settings=settings,
        )
    message = str(excinfo.value)
    assert message.startswith(PHI_REFUSAL_MARKER)
    assert "ehr.example.com" in message
    assert FHIR_HOSTS_SETTING in message


def test_refusal_message_never_carries_the_query_string():
    settings = _settings()
    with pytest.raises(PHIEgressRefused) as excinfo:
        assert_phi_fhir_host_allowed(
            "https://ehr.example.com/fhir/Patient?identifier=mrn-12345",
            settings=settings,
        )
    message = str(excinfo.value)
    assert "mrn-12345" not in message
    assert "/fhir/Patient" not in message


def test_plain_http_is_refused_even_when_the_host_is_listed():
    settings = _settings()
    with pytest.raises(PHIEgressRefused) as excinfo:
        assert_phi_fhir_host_allowed(
            "http://api.practicefusion.com/fhir", settings=settings
        )
    message = str(excinfo.value)
    assert message.startswith(PHI_REFUSAL_MARKER)
    assert "api.practicefusion.com" in message


def test_listed_https_host_is_allowed():
    settings = _settings()
    assert_phi_fhir_host_allowed(
        "https://api.practicefusion.com/fhir/Patient", settings=settings
    )


def test_host_match_is_case_insensitive():
    settings = _settings(AICTRLNET_PHI_FHIR_HOSTS="API.PracticeFusion.com")
    assert_phi_fhir_host_allowed(
        "https://api.PRACTICEFUSION.com/fhir/Patient", settings=settings
    )


def test_empty_allowlist_refuses_every_host():
    settings = _settings(AICTRLNET_PHI_FHIR_HOSTS="")
    with pytest.raises(PHIEgressRefused):
        assert_phi_fhir_host_allowed(
            "https://api.practicefusion.com/fhir", settings=settings
        )


# --- boot problems -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "https://api.practicefusion.com",
        "api.practicefusion.com/fhir",
        "api.practicefusion.com:443",
        "*.practicefusion.com",
        "api practicefusion.com",
    ],
)
def test_malformed_entries_are_boot_problems(raw):
    problems = phi_boot_problems(_settings(AICTRLNET_PHI_FHIR_HOSTS=raw))
    assert any(FHIR_HOSTS_SETTING in p and "bare hostnames" in p for p in problems)


def test_bare_hostnames_are_not_boot_problems():
    problems = phi_boot_problems(
        _settings(AICTRLNET_PHI_FHIR_HOSTS="api.practicefusion.com,auth.example.org")
    )
    assert not [p for p in problems if FHIR_HOSTS_SETTING in p]


def test_empty_allowlist_with_care_gaps_on_warns_but_is_not_a_boot_problem(caplog):
    """A practice that keeps its roster by hand runs the engine with no EHR
    egress at all, so this must not stop the boot — only say so in the log."""
    with caplog.at_level("WARNING", logger="core.phi_egress"):
        problems = phi_boot_problems(
            _settings(AICTRLNET_PHI_FHIR_HOSTS="", CARE_GAPS_ENABLED=True)
        )
    assert not [p for p in problems if FHIR_HOSTS_SETTING in p]
    assert any(FHIR_HOSTS_SETTING in r.getMessage() for r in caplog.records)


def test_care_gaps_off_with_empty_allowlist_does_not_warn(caplog):
    with caplog.at_level("WARNING", logger="core.phi_egress"):
        phi_boot_problems(_settings(AICTRLNET_PHI_FHIR_HOSTS=""))
    assert not [r for r in caplog.records if FHIR_HOSTS_SETTING in r.getMessage()]
