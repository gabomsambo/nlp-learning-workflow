"""Add-to-home-screen: manifest, icons and the head tags that point at them."""

import json
import struct
from pathlib import Path

from fastapi.testclient import TestClient

from webui.app import create_app

STATIC = Path(__file__).resolve().parents[2] / "webui" / "static"


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_manifest_is_installable():
    manifest = json.loads((STATIC / "manifest.webmanifest").read_text())
    assert manifest["display"] == "standalone"
    assert manifest["start_url"] == "/"
    assert manifest["name"] and manifest["short_name"]
    sizes = {icon["sizes"] for icon in manifest["icons"]}
    assert {"192x192", "512x512"} <= sizes
    assert any(icon.get("purpose") == "maskable" for icon in manifest["icons"])
    for icon in manifest["icons"]:
        path = STATIC / icon["src"].removeprefix("/static/")
        width, height = _png_size(path)
        assert f"{width}x{height}" == icon["sizes"], icon["src"]


def test_apple_touch_icon_is_180_square():
    assert _png_size(STATIC / "icons" / "apple-touch-icon.png") == (180, 180)


def test_pages_link_the_manifest_with_credentials():
    # Behind Cloudflare Access an anonymous manifest fetch is redirected to the
    # login page, so the link must send the session cookie.
    html = TestClient(create_app()).get("/discovery/").text
    assert '<link rel="manifest" href="/static/manifest.webmanifest' in html
    assert 'crossorigin="use-credentials"' in html
    assert 'rel="apple-touch-icon" href="/static/icons/apple-touch-icon.png"' in html
    assert 'name="apple-mobile-web-app-capable"' in html
    assert 'name="theme-color"' in html


def test_manifest_is_served_with_a_manifest_type():
    response = TestClient(create_app()).get("/static/manifest.webmanifest")
    assert response.status_code == 200
    assert "manifest" in response.headers["content-type"]


def test_spend_guard_loads_before_any_page_script():
    html = TestClient(create_app()).get("/discovery/").text
    assert html.index("/static/spend-guard.js") < html.index("/static/shell.js")
