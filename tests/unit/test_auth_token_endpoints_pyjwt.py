"""Auth endpoints and control-plane tokens accept only HS256 tokens signed with
SECRET_KEY (PyJWT, algorithms pinned).

The host-side spec tests/integration/regressions/test_jwt_algorithm_pinning.py
covers core.security and the middlewares; these paths need the full app
import graph, so they run here in the edition container. Each path must reject
"alg": "none", HS512 with the same secret, an RS256 header, HS256 forged with
a PEM public key as the secret, a wrong secret and a garbage string.
"""
import asyncio
import base64
import hashlib
import hmac
import json
import time
import warnings
from datetime import timedelta
from types import SimpleNamespace

import jwt
import pytest
from fastapi import HTTPException

from core import security
from core.config import get_settings

SETTINGS = get_settings()

_PUBLIC_PEM = (
    "-----BEGIN PUBLIC KEY-----\n"
    "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAu1SU1LfVLPHCozMxH2Mo\n"
    "4lgOEePzNm0tRgeLezV6ffAt0gunVTLw7onLRnrq0/IzW7yWR7QkrmBL7jTKEn5u\n"
    "+qKhbwKfBstIs+bMY2Zkp18gnTxKLxoS2tFczGkPLPgizskuemMghRniWaoLcyeh\n"
    "kd3qqGElvW/VDL5AaWTg0nLVkjRo9z+40RQzuVaE8AkAFmxZzow3x+VJYKdjykkJ\n"
    "0iT9wCS0DRTXu269V264Vf/3jvredZiKRkgwlL9xNAwxXFg0x/XFw005UWVRIkdg\n"
    "cKWTjpBP2dPwVZ4WWC+9aGVd+Gyn1o0CLelf4rEjGoXbAAEgAqeGUxrcIlbjXfbc\n"
    "mwIDAQAB\n"
    "-----END PUBLIC KEY-----\n"
)


def _b64(obj) -> str:
    raw = obj if isinstance(obj, bytes) else json.dumps(obj, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _forged_tokens(**extra):
    claims = {"sub": "user-123", "tenant_id": "attacker-tenant", "type": "access",
              "exp": int(time.time()) + 600}
    claims.update(extra)
    body = _b64(claims)
    hs_input = f"{_b64({'alg': 'HS256', 'typ': 'JWT'})}.{body}".encode()
    rs_sig = _b64(bytes([1]) * 256)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", jwt.warnings.InsecureKeyLengthWarning)
        hs512 = jwt.encode(claims, SETTINGS.SECRET_KEY, algorithm="HS512")
    return {
        "alg_none": f"{_b64({'alg': 'none', 'typ': 'JWT'})}.{body}.",
        "alg_none_upper": f"{_b64({'alg': 'None', 'typ': 'JWT'})}.{body}.",
        "hs512_same_secret": hs512,
        "rs256_header": f"{_b64({'alg': 'RS256', 'typ': 'JWT'})}.{body}.{rs_sig}",
        "hs256_public_key_secret": (
            f"{hs_input.decode()}."
            f"{_b64(hmac.new(_PUBLIC_PEM.encode(), hs_input, hashlib.sha256).digest())}"
        ),
        "wrong_secret": jwt.encode(claims, "x" * 48, algorithm="HS256"),
        "garbage": "not-a-jwt",
    }


FORGED_KINDS = list(_forged_tokens().keys())


def _run(coro):
    return asyncio.run(coro)


# --- auth endpoints: reset / verify-email / refresh / logout -----------------

@pytest.fixture
def auth_ep(monkeypatch):
    from api.v1.endpoints import auth as ep

    async def _allow(*args, **kwargs):
        return None

    monkeypatch.setattr(ep, "enforce_rate_limit", _allow)
    return ep


@pytest.mark.parametrize("kind", FORGED_KINDS)
def test_password_reset_confirm_rejects_forged(kind, auth_ep):
    data = SimpleNamespace(token=_forged_tokens(type="password_reset")[kind], new_password="x")
    with pytest.raises(HTTPException) as exc:
        _run(auth_ep.confirm_password_reset(data, request=None, db=None))
    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid or expired reset token"


def test_password_reset_confirm_rejects_expired(auth_ep):
    tok = security.create_access_token(
        {"sub": "user-123", "type": "password_reset"}, expires_delta=timedelta(seconds=-5)
    )
    with pytest.raises(HTTPException) as exc:
        _run(auth_ep.confirm_password_reset(SimpleNamespace(token=tok, new_password="x"), request=None, db=None))
    assert exc.value.detail == "Invalid or expired reset token"


def test_password_reset_confirm_decodes_valid_token_then_checks_type(auth_ep):
    tok = security.create_access_token({"sub": "user-123"})  # type=access, not password_reset
    with pytest.raises(HTTPException) as exc:
        _run(auth_ep.confirm_password_reset(SimpleNamespace(token=tok, new_password="x"), request=None, db=None))
    assert exc.value.detail == "Invalid reset token"


@pytest.mark.parametrize("kind", FORGED_KINDS)
def test_verify_email_rejects_forged(kind, auth_ep):
    tok = _forged_tokens(type="email_verification", email="a@example.com")[kind]
    with pytest.raises(HTTPException) as exc:
        _run(auth_ep.verify_email(tok, db=None))
    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid or expired verification token"


def test_verify_email_decodes_valid_token_then_checks_type(auth_ep):
    tok = security.create_access_token({"sub": "user-123", "email": "a@example.com"})
    with pytest.raises(HTTPException) as exc:
        _run(auth_ep.verify_email(tok, db=None))
    assert exc.value.detail == "Invalid verification token"


@pytest.mark.parametrize("kind", FORGED_KINDS)
def test_refresh_rejects_forged(kind, auth_ep):
    tok = _forged_tokens(type="refresh", jti="j-1")[kind]
    with pytest.raises(HTTPException) as exc:
        _run(auth_ep.refresh_access_token(SimpleNamespace(refresh_token=tok), request=None, db=None))
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid or expired refresh token"


def test_refresh_rejects_access_token(auth_ep):
    tok = security.create_access_token({"sub": "user-123"})
    with pytest.raises(HTTPException) as exc:
        _run(auth_ep.refresh_access_token(SimpleNamespace(refresh_token=tok), request=None, db=None))
    assert exc.value.detail == "Invalid refresh token"


class _FakeDB:
    async def commit(self):
        return None

    async def rollback(self):
        return None


def _logout(auth_ep, monkeypatch, refresh_token):
    revoked = []

    async def _revoke(jti):
        revoked.append(jti)

    import core.token_revocation as tr
    monkeypatch.setattr(tr, "revoke_refresh_jti", _revoke)
    user = SimpleNamespace(token_version=0)
    _run(auth_ep.logout(body=SimpleNamespace(refresh_token=refresh_token), current_user=user, db=_FakeDB()))
    return revoked


def test_logout_revokes_valid_refresh_jti(auth_ep, monkeypatch):
    tok = security.create_refresh_token({"sub": "user-123", "jti": "jti-ok"})
    assert _logout(auth_ep, monkeypatch, tok) == ["jti-ok"]


@pytest.mark.parametrize("kind", FORGED_KINDS)
def test_logout_ignores_forged_refresh(kind, auth_ep, monkeypatch):
    tok = _forged_tokens(type="refresh", jti="jti-forged")[kind]
    assert _logout(auth_ep, monkeypatch, tok) == []


# --- control-plane component tokens --------------------------------------------

def _component():
    return SimpleNamespace(
        id="comp-1", name="worker", type=SimpleNamespace(value="adapter"),
        edition="community", capabilities=[SimpleNamespace(name="run")],
    )


def test_component_token_round_trip_and_refresh():
    from control_plane.auth import ComponentAuth
    auth = ComponentAuth()
    tok, _ = auth.create_component_token(_component())
    assert jwt.get_unverified_header(tok)["alg"] == "HS256"
    payload = auth.verify_component_token(tok)
    assert payload["sub"] == "comp-1" and payload["type"] == "component"
    assert payload["capabilities"] == ["run"]
    new_tok, _ = auth.refresh_component_token(tok)
    assert auth.verify_component_token(new_tok)["sub"] == "comp-1"


@pytest.mark.parametrize("kind", FORGED_KINDS)
def test_component_token_rejects_forged(kind):
    from control_plane.auth import ComponentAuth
    with pytest.raises(HTTPException) as exc:
        ComponentAuth().verify_component_token(_forged_tokens(type="component")[kind])
    assert exc.value.status_code == 403
    assert exc.value.detail.startswith("Invalid token")


def test_component_token_rejects_user_access_token():
    from control_plane.auth import ComponentAuth
    with pytest.raises(HTTPException) as exc:
        ComponentAuth().verify_component_token(security.create_access_token({"sub": "user-123"}))
    assert exc.value.detail == "Invalid token type"
