"""get_deployment_type(): hosted vs self-hosted.

The Studio and Beast run the self-hosted compose with neither DEPLOYMENT_TYPE
nor K_SERVICE set and must stay "self-hosted". Cloud Run always sets K_SERVICE.
The raw Settings field stays None when unset, so core/enforcement.py, which
reads it directly, keeps its current mode until it is moved deliberately.
"""

from core.config import get_deployment_type, get_settings


def test_studio_and_beast_env_is_self_hosted(monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_TYPE", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)
    assert get_deployment_type() == "self-hosted"


def test_cloud_run_is_cloud(monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_TYPE", raising=False)
    monkeypatch.setenv("K_SERVICE", "enterprise")
    assert get_deployment_type() == "cloud"


def test_explicit_setting_wins(monkeypatch):
    monkeypatch.setenv("K_SERVICE", "enterprise")
    monkeypatch.setenv("DEPLOYMENT_TYPE", "self-hosted")
    assert get_deployment_type() == "self-hosted"


def test_raw_setting_stays_unset_for_enforcement(monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_TYPE", raising=False)
    monkeypatch.setenv("K_SERVICE", "enterprise")
    assert get_settings().DEPLOYMENT_TYPE is None
