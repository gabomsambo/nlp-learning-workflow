"""Workstation status: classifiers, a mocked collection, and a route that stays up.

Nothing here dials PostgREST, Qdrant, SearXNG or IndexTTS. The collection test
drives httpx.MockTransport and asserts a secret placed on the settings object
never appears in the payload.
"""

import asyncio
import json

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from webui.services import system_status as status


class _Settings:
    qdrant_url = "http://qdrant.test"
    qdrant_api_key = "super-secret-qdrant"
    searxng_url = "http://searxng.test"
    schedule_enabled = False
    openai_api_key = "sk-openai-secret"
    anthropic_api_key = ""
    deepseek_api_key = "sk-deepseek-secret"
    podcast_audio_dir = "/tmp/nlp-status-audio-missing"


def test_public_summary_matches_the_today_chip_contract():
    assert status.public_summary([{"state": "ok", "label": "Web app"}]) == "All services up"
    assert status.public_summary([
        {"state": "down", "label": "IndexTTS"},
    ]) == "IndexTTS needs attention"
    assert status.public_summary([
        {"state": "down", "label": "Qdrant Cloud"},
        {"state": "ok", "label": "Database"},
        {"state": "down", "label": "SearXNG"},
    ]) == "2 services need attention"


def test_local_qdrant_plain_text_404_is_not_the_cloud_cluster():
    check = status.classify_qdrant(404, "404 page not found\n", None)
    assert check["id"] == "qdrant"
    assert check["label"] == "Qdrant (local)"
    assert check["state"] == "down"
    assert "plain text 404" in check["summary"]
    assert "./workstation up" in check["fix"]
    assert "fatal" not in check


def test_cloud_suspension_is_shown_and_does_not_fail_the_page():
    cloud = status.classify_qdrant(404, "404 page not found\n", None, target="cloud")
    local = status.classify_qdrant(200, '{"title":"qdrant","version":"1.19.2"}', None)
    assert cloud["id"] == "qdrant-cloud"
    assert cloud["label"] == "Qdrant Cloud"
    assert cloud["state"] == "down"
    assert cloud["fatal"] is False
    assert "Suspended free-tier cluster" in cloud["summary"]
    assert "Qdrant Cloud console" in cloud["fix"]
    assert local["label"] == "Qdrant (local)"
    assert status.counts_as_down(cloud) is False
    assert status.public_summary([local, cloud]) == "All services up"


def test_qdrant_json_banner_is_up():
    check = status.classify_qdrant(200, '{"title":"qdrant","version":"1.15"}', None)
    assert check["label"] == "Qdrant (local)"
    assert check["state"] == "ok"
    assert check["fix"] is None


def test_searxng_names_a_suspended_arxiv_engine():
    payload = {"results": [], "unresponsive_engines": [["arxiv", "Suspended: too many requests"]]}
    check = status.classify_searxng(200, payload, None)
    assert check["state"] == "down"
    assert "suspended" in check["summary"].lower()


def test_other_engine_suspensions_are_info_not_a_failure():
    payload = {
        "results": [{"title": "Attention Is All You Need"}],
        "unresponsive_engines": [
            ["brave", "Suspended: too many requests"],
            ["duckduckgo", "CAPTCHA"],
            ["startpage", "Suspended: CAPTCHA"],
        ],
    }
    check = status.classify_searxng(200, payload, None)
    assert check["state"] == "ok"
    assert check["fix"] is None
    assert "arXiv is answering" in check["summary"]
    assert "brave" in check["summary"]
    assert "duckduckgo" in check["summary"]
    assert "startpage" in check["summary"]
    assert "suspended" not in check["summary"].lower()
    assert status.public_summary([check]) == "All services up"


def test_a_missing_column_is_pgrst204_and_a_present_one_is_applied():
    missing = {"code": "PGRST204", "message": "Could not find the 'source_material' column"}
    assert status.classify_column(400, missing) == "missing"
    assert status.classify_column(200, []) == "applied"
    assert status.classify_column(500, {"message": "boom"}) == "unknown"


def test_kind_probe_reads_the_check_constraint():
    assert status.classify_kind(201, "") == "applied"
    assert status.classify_kind(400, {"code": "23514", "message": "pipeline_runs_kind_check"}) == "missing"


def test_key_failure_is_a_short_phrase_and_the_raw_error_is_not_copied():
    raw = "HTTP 429 insufficient_quota: you exceeded your current quota. Traceback follows."
    found = status.mine_key_failure([{"error": raw, "created_at": "2026-10-01T00:00:00Z"}])
    assert found["provider"] == "openai"
    assert found["code"] == "insufficient_quota"
    assert "Traceback" not in found["summary"]
    assert "429" not in found["summary"]


def test_key_check_reports_presence_without_the_secret():
    check = status.classify_keys(
        {"openai": True, "anthropic": False, "deepseek": True},
        {"provider": "openai", "code": "insufficient_quota", "summary": "OpenAI is out of quota.", "fix": "Add credit."},
        False,
    )
    assert check["state"] == "down"
    assert "Anthropic key is missing" in check["summary"]
    assert "out of quota" in check["summary"]
    # A missing key is the thing to fix; the quota note stays in the summary.
    assert check["fix"] == "Add the missing key to .env."
    assert "sk-" not in json.dumps(check)


def test_disabled_scheduler_is_ok():
    check = status.classify_scheduler(False, None, False)
    assert check["state"] == "ok"
    assert "exited scheduler is normal" in check["summary"]


def test_tts_down_includes_the_start_command_and_workstation():
    check = status.classify_tts({
        "ready": False,
        "message": "IndexTTS is not running on port 7860.",
        "start_command": "cd /home/gabo/index-tts && uv run webui.py --host 0.0.0.0 --port 7860",
    })
    assert check["state"] == "down"
    assert "./workstation up --tts" in check["fix"]
    assert "uv run webui.py" in check["fix"]


def test_voices_count_only_usable_files():
    check = status.classify_voices({
        "voices": [
            {"usability": "usable"},
            {"usability": "usable"},
            {"usability": "unusable"},
        ]
    })
    assert check == status.make_check("voices", "Voices", "ok", "2 usable voices.", None)


def test_redact_strips_secrets_before_a_summary_is_shown():
    text = status.redact("failed with key super-secret-qdrant", ["super-secret-qdrant"])
    assert "super-secret-qdrant" not in text
    assert "••••" in text


def _handler(seen):
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "qdrant.test":
            seen["qdrant_key"] = request.headers.get("api-key")
            return httpx.Response(404, text="404 page not found")
        if host == "searxng.test":
            seen.setdefault("searxng_engines", []).append(request.url.params.get("engines"))
            return httpx.Response(
                200,
                json={"results": [], "unresponsive_engines": []},
            )
        if request.method == "DELETE":
            seen["deleted"] = True
            return httpx.Response(204)
        if request.method == "POST":
            body = json.loads(request.content)
            seen.setdefault("kinds", []).append(body["kind"])
            assert "tx=rollback" in request.headers.get("prefer", "")
            if body["kind"] == "podcast_script":
                return httpx.Response(
                    400,
                    json={"code": "23514", "message": "pipeline_runs_kind_check"},
                )
            return httpx.Response(201, headers={"Preference-Applied": "tx=rollback"})
        select = request.url.params.get("select", "")
        if request.url.path.endswith("/pillars"):
            return httpx.Response(200, json=[{"id": "formal-linguistics-nlp"}])
        if select == "source_material":
            return httpx.Response(
                400,
                json={"code": "PGRST204", "message": "Could not find the 'source_material' column"},
            )
        if select in ("options", "audio_metadata"):
            return httpx.Response(200, json=[])
        if select == "*" and request.url.params.get("limit") == "0":
            return httpx.Response(200, json=[])
        if request.url.params.get("error") == "not.is.null":
            return httpx.Response(
                200,
                json=[{
                    "error": "insufficient_quota from sk-openai-secret",
                    "created_at": "2026-10-01T00:00:00Z",
                    "kind": "upload",
                }],
            )
        if request.url.params.get("trigger_source") == "eq.scheduler":
            return httpx.Response(200, json=[])
        return httpx.Response(500, text="unexpected " + str(request.url))

    return handler


async def test_collection_survives_a_failing_check_and_hides_secrets():
    status.clear_status_caches()
    seen = {}

    def tts():
        raise RuntimeError("IndexTTS socket closed, key super-secret-qdrant")

    def voices():
        return {"voices": [{"usability": "usable"}]}

    def storage():
        return status.make_check("storage", "Storage", "ok", "Uploads 0 B. Audio 0 B.", None)

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler(seen))) as client:
        payload = await status.collect_status(
            client=client,
            settings=_Settings(),
            postgrest_url="http://postgrest.test",
            auth_token="jwt-secret-token",
            tts_status=tts,
            list_voices=voices,
            storage=storage,
        )

    encoded = json.dumps(payload)
    for secret in ("super-secret-qdrant", "sk-openai-secret", "sk-deepseek-secret", "jwt-secret-token"):
        assert secret not in encoded
    by_id = {check["id"]: check for check in payload["checks"]}
    assert by_id["web"]["state"] == "ok"
    assert by_id["postgrest"]["state"] == "ok"
    assert by_id["migration-011"]["state"] == "down"
    assert "011_podcast_source_material.sql" in by_id["migration-011"]["fix"]
    assert by_id["migration-012"]["state"] == "ok"
    assert by_id["migration-014"]["state"] == "ok"
    assert by_id["migration-015"]["state"] == "ok"
    assert by_id["migration-016"]["state"] == "down"
    for number in range(17, 23):
        row = by_id[f"migration-{number:03d}"]
        assert row["state"] == "ok", row
        assert row["summary"] == "Applied."
    assert by_id["semantic-scholar"]["state"] == "ok"
    assert by_id["semantic-scholar"]["fatal"] is False
    assert "No API key" in by_id["semantic-scholar"]["summary"]
    assert by_id["qdrant"]["label"] == "Qdrant (local)"
    assert by_id["qdrant"]["state"] == "down"
    assert "plain text 404" in by_id["qdrant"]["summary"]
    assert by_id["qdrant-cloud"]["label"] == "Qdrant Cloud"
    assert by_id["qdrant-cloud"]["state"] == "ok"
    assert by_id["qdrant-cloud"]["summary"] == "Not configured."
    assert by_id["qdrant-cloud"]["fatal"] is False
    assert by_id["searxng"]["state"] == "ok"
    assert by_id["indextts"]["state"] == "down"
    assert "super-secret" not in by_id["indextts"]["summary"]
    assert by_id["voices"]["state"] == "ok"
    assert by_id["keys"]["state"] == "down"
    assert "out of quota" in by_id["keys"]["summary"]
    assert by_id["scheduler"]["state"] == "ok"
    assert by_id["storage"]["state"] == "ok"
    assert payload["ok"] is False
    assert payload["summary"] == status.public_summary(payload["checks"])
    assert payload["summary"].endswith("services need attention")
    assert seen["qdrant_key"] == "super-secret-qdrant"
    assert seen["deleted"] is True
    assert seen["kinds"] == ["upload", "podcast_script"]
    assert seen["searxng_engines"] == ["arxiv"]


async def test_searxng_probes_arxiv_only_and_reuses_the_result():
    status.clear_status_caches()
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params.get("engines"))
        return httpx.Response(
            200,
            json={
                "results": [{"title": "Attention"}],
                "unresponsive_engines": [["brave", "Suspended: too many requests"]],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        first = await status._check_searxng(client, _Settings(), now=1_000.0)
        second = await status._check_searxng(client, _Settings(), now=1_000.0 + 60)
        third = await status._check_searxng(
            client, _Settings(), now=1_000.0 + status.SEARXNG_CACHE_SECONDS + 1
        )

    assert seen == ["arxiv", "arxiv"]
    assert first["state"] == "ok"
    assert "brave" in first["summary"]
    assert second == first
    assert third["state"] == "ok"


def test_voice_budget_outlasts_a_busy_scan():
    assert status.VOICES_TIMEOUT_SECONDS >= 8
    assert status.VOICES_TIMEOUT_SECONDS > status.OVERALL_TIMEOUT_SECONDS
    assert status.voice_check_age_phrase(30) == "less than a minute ago"
    assert status.voice_check_age_phrase(60) == "1 minute ago"
    assert status.voice_check_age_phrase(120) == "2 minutes ago"


async def test_voices_timeout_keeps_a_recent_good_result():
    status.clear_status_caches()
    calls = {"n": 0}

    def good():
        calls["n"] += 1
        return {"voices": [{"usability": "usable"}]}

    async def slow():
        calls["n"] += 1
        await asyncio.sleep(5)

    first = await status.check_voices(good, now=0.0, timeout=1.0)
    assert first == status.make_check("voices", "Voices", "ok", "1 usable voice.", None)
    cached = await status.check_voices(slow, now=10.0, timeout=1.0)
    assert cached["summary"] == first["summary"]
    assert calls["n"] == 1

    aged = await status.check_voices(
        slow, now=status.VOICES_CACHE_SECONDS + 5, timeout=0.05
    )
    assert aged["state"] == "ok"
    assert aged["summary"].startswith("1 usable voice.")
    assert "Last checked" in aged["summary"]
    assert calls["n"] == 2

    # The failed attempt starts its own cache window, so a still-busy CPU
    # does not time out again until that window ends.
    held = await status.check_voices(slow, now=status.VOICES_CACHE_SECONDS + 10, timeout=0.05)
    assert held["state"] == "ok"
    assert calls["n"] == 2

    expired = await status.check_voices(
        slow, now=status.VOICES_LAST_GOOD_SECONDS + status.VOICES_CACHE_SECONDS + 20, timeout=0.05
    )
    assert expired["state"] == "down"
    assert calls["n"] == 3


async def test_voices_scan_is_not_cut_by_the_overall_budget(monkeypatch):
    status.clear_status_caches()
    monkeypatch.setattr(status, "OVERALL_TIMEOUT_SECONDS", 0.01)

    async def slow():
        await asyncio.sleep(0.05)
        return {"voices": [{"usability": "usable"}]}

    seen = {}

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler(seen))) as client:
        payload = await status.collect_status(
            client=client,
            settings=_Settings(),
            postgrest_url="http://postgrest.test",
            auth_token="jwt-secret-token",
            tts_status=lambda: {"ready": True, "message": "ready"},
            list_voices=slow,
            storage=lambda: status.make_check("storage", "Storage", "ok", "Uploads 0 B.", None),
        )

    by_id = {check["id"]: check for check in payload["checks"]}
    assert by_id["voices"]["state"] == "ok"
    assert by_id["voices"]["summary"] == "1 usable voice."


def test_route_answers_200_when_collection_raises():
    from webui.routers.api import system_status as route

    def boom(**kwargs):
        raise RuntimeError("super-secret-qdrant exploded")

    app = FastAPI()
    app.include_router(route.router)
    client = TestClient(app, raise_server_exceptions=False)

    original = status.collect_status
    status.collect_status = boom
    try:
        response = client.get("/api/system/status")
    finally:
        status.collect_status = original

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["summary"] == "Status needs attention"
    assert body["checks"][0]["state"] == "down"
    assert "super-secret-qdrant" not in response.text


def test_migration_rows_follow_the_sql_file_convention(tmp_path):
    specs = status.migration_table_specs()
    ids = [spec["id"] for spec in specs]
    assert ids[:6] == [f"migration-{number:03d}" for number in range(17, 23)]
    by_id = {spec["id"]: spec for spec in specs}
    assert "paper_texts" in by_id["migration-017"]["tables"]
    assert "paper_chunks" in by_id["migration-017"]["tables"]
    assert "podcast_fact_checks" in by_id["migration-022"]["tables"]
    assert "podcast_claims" in by_id["migration-022"]["tables"]
    assert not any(spec["id"] == "migration-016" for spec in specs)

    (tmp_path / "016_skip.sql").write_text(
        "CREATE TABLE IF NOT EXISTS should_not_appear (id int);\n",
        encoding="utf-8",
    )
    (tmp_path / "023_future_widget.sql").write_text(
        "CREATE TABLE IF NOT EXISTS future_widget (id int);\n",
        encoding="utf-8",
    )
    (tmp_path / "024_no_table.sql").write_text(
        "-- a column migration with no new table\n",
        encoding="utf-8",
    )
    found = status.migration_table_specs(tmp_path)
    assert [spec["id"] for spec in found] == ["migration-023"]
    assert found[0]["tables"] == ["future_widget"]
    assert "024" not in found[0]["file"]


def test_a_missing_migration_table_is_down():
    assert status.classify_table(
        404,
        {"code": "PGRST205", "message": "Could not find the table paper_texts"},
    ) == "missing"
    assert status.classify_table(200, []) == "applied"


async def test_rejected_semantic_scholar_key_is_on_the_status_page(caplog):
    status.clear_status_caches()
    settings = _Settings()
    settings.semantic_scholar_api_key = "rejected-s2-key"
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert request.headers["x-api-key"] == "rejected-s2-key"
        assert "rejected-s2-key" not in str(request.url)
        return httpx.Response(403, json={"message": "Forbidden"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with caplog.at_level("ERROR"):
            check = await status._check_semantic_scholar(client, settings, now=10.0)
            again = await status._check_semantic_scholar(client, settings, now=20.0)

    assert calls["n"] == 1
    assert check["id"] == "semantic-scholar"
    assert check["state"] == "down"
    assert check["fatal"] is False
    assert "rejected" in check["summary"]
    assert "403" in check["summary"]
    assert "rejected-s2-key" not in json.dumps(check)
    assert again == check
    assert "SEMANTIC_SCHOLAR_API_KEY" in caplog.text
    assert "not changed" in caplog.text
    assert status.counts_as_down(check) is False
