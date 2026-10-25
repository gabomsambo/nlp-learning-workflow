"""Cloudflare Access JWT check: valid, invalid, missing, and the desktop.

Hermetic: keys are generated here and the JWKS fetch is injected, so nothing
dials Cloudflare.
"""

import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from webui.access import (
    JWT_HEADER,
    AccessMiddleware,
    AccessVerifier,
    is_remote,
    normalise_team_domain,
)

TEAM = "captain.cloudflareaccess.com"
AUD = "a" * 64
KID = "test-key-1"


def _keypair():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


KEY = _keypair()
OTHER_KEY = _keypair()


def _jwk(private_key, kid=KID) -> dict:
    data = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    data.update({"kid": kid, "alg": "RS256", "use": "sig"})
    return data


def _token(private_key=KEY, kid=KID, **overrides) -> str:
    now = int(time.time())
    claims = {
        "aud": [AUD],
        "iss": f"https://{TEAM}",
        "iat": now,
        "exp": now + 600,
        "email": "gabriel.m.sambo@gmail.com",
    }
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": kid})


def _verifier(**kwargs) -> AccessVerifier:
    return AccessVerifier(TEAM, AUD, fetch_jwks=lambda: [_jwk(KEY)], **kwargs)


def _client(verifier=None) -> TestClient:
    app = FastAPI()

    @app.get("/ping")
    async def ping(request: Request):
        return {"remote": request.state.remote, "email": getattr(request.state, "access_email", None)}

    app.add_middleware(AccessMiddleware, verifier=verifier)
    return TestClient(app)


REMOTE = {"Cf-Ray": "8a1b2c3d4e5f-AMS", "Cf-Connecting-Ip": "203.0.113.9"}


def test_local_request_needs_no_token():
    response = _client(_verifier()).get("/ping")
    assert response.status_code == 200
    assert response.json() == {"remote": False, "email": None}


def test_local_request_works_even_when_remote_access_is_unconfigured():
    assert _client(verifier=None).get("/ping").status_code == 200


def test_remote_request_without_token_is_refused():
    response = _client(_verifier()).get("/ping", headers=REMOTE)
    assert response.status_code == 401


def test_remote_request_with_valid_token_passes_and_exposes_the_email():
    response = _client(_verifier()).get("/ping", headers={**REMOTE, JWT_HEADER: _token()})
    assert response.status_code == 200
    assert response.json() == {"remote": True, "email": "gabriel.m.sambo@gmail.com"}


@pytest.mark.parametrize(
    "token",
    [
        pytest.param("not-a-jwt", id="garbage"),
        pytest.param(_token(OTHER_KEY), id="signed-by-another-key"),
        pytest.param(_token(kid="unknown-kid"), id="unknown-kid"),
        pytest.param(_token(aud=["b" * 64]), id="wrong-audience"),
        pytest.param(_token(iss="https://evil.cloudflareaccess.com"), id="wrong-issuer"),
        pytest.param(_token(exp=int(time.time()) - 3600), id="expired"),
        pytest.param(_token(exp=None), id="no-expiry"),
    ],
)
def test_remote_request_with_bad_token_is_refused(token):
    response = _client(_verifier()).get("/ping", headers={**REMOTE, JWT_HEADER: token})
    assert response.status_code == 403
    assert "email" not in response.text


def test_unsigned_token_is_refused():
    now = int(time.time())
    forged = jwt.encode(
        {"aud": [AUD], "iss": f"https://{TEAM}", "iat": now, "exp": now + 600},
        key="",
        algorithm="none",
    )
    response = _client(_verifier()).get("/ping", headers={**REMOTE, JWT_HEADER: forged})
    assert response.status_code == 403


def test_remote_request_is_refused_when_access_is_not_configured():
    response = _client(verifier=None).get("/ping", headers={**REMOTE, JWT_HEADER: _token()})
    assert response.status_code == 403


def test_email_allow_list_is_enforced_when_set():
    verifier = _verifier(allowed_emails=frozenset({"someone.else@example.com"}))
    response = _client(verifier).get("/ping", headers={**REMOTE, JWT_HEADER: _token()})
    assert response.status_code == 403


def test_forwarded_for_alone_marks_a_request_remote():
    response = _client(_verifier()).get("/ping", headers={"X-Forwarded-For": "203.0.113.9"})
    assert response.status_code == 401


def test_forging_a_token_header_without_a_valid_token_does_not_make_a_request_local():
    response = _client(_verifier()).get("/ping", headers={JWT_HEADER: "x"})
    assert response.status_code == 403


def test_rotated_keys_are_picked_up_after_the_refetch_floor():
    calls = []

    def fetch():
        calls.append(1)
        return [_jwk(KEY)] if len(calls) == 1 else [_jwk(KEY), _jwk(OTHER_KEY, "rotated")]

    verifier = AccessVerifier(TEAM, AUD, fetch_jwks=fetch)
    verifier.verify(_token())
    verifier._fetched_at -= 120  # past the 60 s floor
    assert verifier.verify(_token(OTHER_KEY, kid="rotated"))["email"]


def test_unknown_kid_does_not_refetch_inside_the_floor():
    calls = []

    def fetch():
        calls.append(1)
        return [_jwk(KEY)]

    verifier = AccessVerifier(TEAM, AUD, fetch_jwks=fetch)
    verifier.verify(_token())
    for _ in range(5):
        with pytest.raises(Exception):
            verifier.verify(_token(kid="nope"))
    assert len(calls) == 1


def test_certs_endpoint_down_with_no_cached_keys_answers_503():
    def fetch():
        raise RuntimeError("certs endpoint down")

    verifier = AccessVerifier(TEAM, AUD, fetch_jwks=fetch)
    response = _client(verifier).get("/ping", headers={**REMOTE, JWT_HEADER: _token()})
    assert response.status_code == 503


def test_is_remote_markers():
    assert not is_remote({"host": "localhost:8000", "user-agent": "x"})
    assert is_remote({"cf-ray": "1"})
    assert is_remote({"CF-Connecting-IP": "1.2.3.4"})
    assert is_remote({"x-forwarded-proto": "https"})
    assert is_remote({"forwarded": "for=1.2.3.4"})


def test_team_domain_is_normalised():
    assert normalise_team_domain("https://Captain.cloudflareaccess.com/") == TEAM


def test_create_app_is_wired_to_the_environment(monkeypatch):
    from webui.app import create_app

    monkeypatch.setenv("CF_ACCESS_TEAM_DOMAIN", TEAM)
    monkeypatch.setenv("CF_ACCESS_AUD", AUD)
    monkeypatch.setattr(AccessVerifier, "_fetch_jwks_over_http", lambda self: [_jwk(KEY)])
    client = TestClient(create_app())

    assert client.get("/api/spend", headers=REMOTE).status_code == 401
    forged = client.get("/api/spend", headers={**REMOTE, JWT_HEADER: _token(OTHER_KEY)})
    assert forged.status_code == 403
