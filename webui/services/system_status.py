"""Read-only workstation status. No model calls, no writes that stick.

Each check has a short timeout. A check that raises becomes one DOWN row; it
does not fail the collection. Kind probes insert a row only inside a PostgREST
``tx=rollback`` transaction and then delete ``pillar_id=status-probe`` either
way, so a server that ignores the rollback preference cannot leave a run behind.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import httpx

logger = logging.getLogger(__name__)

CHECK_TIMEOUT_SECONDS = 2.5
OVERALL_TIMEOUT_SECONDS = 3.0
# Status is fetched on every page. Cache the arXiv probe so that load does not
# search the web each time. A few minutes is fresh enough for this panel.
SEARXNG_CACHE_SECONDS = 180.0
# The voice folder is about 1.8 GB. A scan took 4.1s while the CPU was busy
# and 1.2s idle, and the 3s budget around every other check reported Voices
# DOWN. The library rarely changes, so a recent good scan stands in for one
# that does not finish, and the next attempt waits out the cache window
# instead of timing out on every page.
VOICES_TIMEOUT_SECONDS = 8.0
VOICES_CACHE_SECONDS = 300.0
VOICES_LAST_GOOD_SECONDS = 1800.0
PROBE_PILLAR = "status-probe"
SUSPENDED_QDRANT = "404 page not found"
MAX_STORAGE_FILES = 5000

MIGRATION_COLUMNS = (
    {
        "id": "migration-011",
        "label": "Podcast source material",
        "table": "podcast_scripts",
        "column": "source_material",
        "file": "docs/migrations/011_podcast_source_material.sql",
        "missing": "podcast_scripts.source_material is missing, so a script's provenance is not stored.",
    },
    {
        "id": "migration-012",
        "label": "Podcast options",
        "table": "podcast_scripts",
        "column": "options",
        "file": "docs/migrations/012_podcast_options.sql",
        "missing": "podcast_scripts.options is missing, so episode settings are not stored.",
    },
    {
        "id": "migration-014",
        "label": "Audio metadata",
        "table": "podcast_scripts",
        "column": "audio_metadata",
        "file": "docs/migrations/014_podcast_audio.sql",
        "missing": "podcast_scripts.audio_metadata is missing, so generated audio is not recorded.",
    },
)

KIND_PROBES = (
    {
        "id": "migration-015",
        "label": "Upload runs",
        "kind": "upload",
        "trigger": "ui_upload",
        "file": "docs/migrations/015_upload_runs.sql",
        "missing": "pipeline_runs does not accept an upload yet.",
    },
    {
        "id": "migration-016",
        "label": "Script runs",
        "kind": "podcast_script",
        "trigger": "ui_podcast_script",
        "file": "docs/migrations/016_podcast_script_runs.sql",
        "missing": "pipeline_runs does not accept a podcast script run yet.",
    },
    {
        "id": "migration-023",
        "label": "Fact-check runs",
        "kind": "podcast_factcheck",
        "trigger": "ui_podcast_factcheck",
        "file": "docs/migrations/023_podcast_factcheck_runs.sql",
        "missing": "pipeline_runs does not accept a fact-check run yet.",
    },
)

# Migrations 017 onward add tables. A new file under docs/migrations whose
# name starts with a number >= 017 and which contains CREATE TABLE is a
# status row. Column and kind checks above stay explicit: those files do
# not introduce a table the panel can GET.
_TABLE_SQL = re.compile(
    r"CREATE TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)
_MIGRATION_TABLES_FROM = 17
STATE_MIGRATION = "docs/migrations/026_cloud_publish_state.sql"
S2_PROBE_URL = "https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762"
S2_CACHE_SECONDS = 180.0

# Short, stable phrases mined from pipeline_runs.error. The raw exception stays
# on the Activity row, one click away; this surface only names the failure.
_KEY_FAILURES = (
    (
        "insufficient_quota",
        "openai",
        "OpenAI refused a request because the account is out of quota.",
        "Add OpenAI credit, then retry the run from Activity.",
    ),
    (
        "invalid_api_key",
        "openai",
        "OpenAI rejected the API key.",
        "Check OPENAI_API_KEY in .env.",
    ),
    (
        "credit balance is too low",
        "anthropic",
        "Anthropic refused a request because the credit balance is too low.",
        "Add Anthropic credit, then retry the run from Activity.",
    ),
    (
        "authentication fails",
        "deepseek",
        "DeepSeek rejected the API key.",
        "Check DEEPSEEK_API_KEY in .env.",
    ),
)


def resolve_postgrest_url() -> str:
    url = os.getenv("POSTGREST_URL") or os.getenv("SUPABASE_URL") or "http://localhost:3000"
    url = url.rstrip("/")
    if "supabase.co" in url and "/rest/v1" not in url:
        url = f"{url}/rest/v1"
    return url


def make_check(
    check_id: str,
    label: str,
    state: str,
    summary: str,
    fix: Optional[str] = None,
    *,
    fatal: bool = True,
) -> Dict[str, Any]:
    row = {
        "id": check_id,
        "label": label,
        "state": state,
        "summary": summary,
        "fix": fix,
    }
    # A non-fatal down is shown on Activity and does not flip the chip or `ok`.
    if not fatal:
        row["fatal"] = False
    return row


def counts_as_down(check: Dict[str, Any]) -> bool:
    """True when a check should fail the page summary.

    Qdrant Cloud is reported and is not one of these: a suspended or deleted
    cluster leaves the local index in use.
    """
    return check.get("state") != "ok" and check.get("fatal", True) is not False


def public_line(text: str, limit: int = 180) -> str:
    line = (text or "").strip().splitlines()[0] if text else ""
    return line[:limit]


_searxng_cache: Dict[str, tuple] = {}
_s2_cache: Dict[str, tuple] = {}
_voices_state: Dict[str, Any] = {
    "checked_at": None,
    "fallback": False,
    "latest": None,
    "good_at": None,
    "good": None,
}


def clear_status_caches() -> None:
    """Drop cached SearXNG, Semantic Scholar, and voice results."""
    _searxng_cache.clear()
    _s2_cache.clear()
    _voices_state.update(
        checked_at=None,
        fallback=False,
        latest=None,
        good_at=None,
        good=None,
    )


def _clock(now: Optional[float]) -> float:
    return time.monotonic() if now is None else now


def voice_check_age_phrase(seconds: float) -> str:
    """How old a cached voice scan is, for the status summary."""
    if seconds < 60:
        return "less than a minute ago"
    minutes = int(seconds // 60)
    if minutes < 60:
        unit = "minute" if minutes == 1 else "minutes"
        return f"{minutes} {unit} ago"
    hours = int(minutes // 60)
    unit = "hour" if hours == 1 else "hours"
    return f"{hours} {unit} ago"


def _with_checked_age(check: Dict[str, Any], age_seconds: float) -> Dict[str, Any]:
    aged = dict(check)
    summary = str(check.get("summary") or "").rstrip(".")
    aged["summary"] = f"{summary}. Last checked {voice_check_age_phrase(age_seconds)}."
    return aged


def redact(text: str, secrets: Sequence[Optional[str]]) -> str:
    cleaned = text or ""
    for secret in secrets:
        if secret and len(secret) >= 6:
            cleaned = cleaned.replace(secret, "••••")
    return cleaned


def human_bytes(size: int) -> str:
    value = float(max(size, 0))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{int(size)} B"


def classify_qdrant(
    status_code: Optional[int],
    body: str,
    error: Optional[str],
    *,
    target: str = "local",
) -> Dict[str, Any]:
    """Classify one Qdrant probe.

    ``target="local"`` is ``QDRANT_URL`` (the compose server search reads).
    ``target="cloud"`` is optional ``QDRANT_CLOUD_URL``. A cloud failure is
    ``fatal=False``: Activity still shows the row, and the page stays up.
    """
    cloud = target == "cloud"
    check_id = "qdrant-cloud" if cloud else "qdrant"
    label = "Qdrant Cloud" if cloud else "Qdrant (local)"
    fatal = not cloud
    if error:
        if cloud:
            return make_check(
                check_id,
                label,
                "down",
                "Could not reach Qdrant Cloud.",
                "Check QDRANT_CLOUD_URL in .env.",
                fatal=False,
            )
        return make_check(
            check_id,
            label,
            "down",
            "Could not reach the local Qdrant.",
            "./workstation up",
            fatal=fatal,
        )
    text = body or ""
    if status_code == 200 and text.lstrip().startswith("{"):
        return make_check(check_id, label, "ok", "Reachable.", None, fatal=fatal)
    if SUSPENDED_QDRANT in text:
        if cloud:
            return make_check(
                check_id,
                label,
                "down",
                "Cloud cluster suspended or deleted.",
                "Check the Qdrant Cloud console. A suspended or deleted cluster can leave ingress answering every path with plain text 404 page not found.",
                fatal=False,
            )
        return make_check(
            check_id,
            label,
            "down",
            "The local Qdrant answered plain text 404.",
            "./workstation up",
        )
    if cloud:
        return make_check(
            check_id,
            label,
            "down",
            "Could not reach Qdrant Cloud.",
            "Check QDRANT_CLOUD_URL in .env.",
            fatal=False,
        )
    return make_check(
        check_id,
        label,
        "down",
        "Could not reach the local Qdrant.",
        "./workstation up",
    )


def _split_unresponsive_engines(payload: Dict[str, Any]) -> tuple[bool, str, List[str]]:
    """Return whether arXiv is listed, its reason, and the other engine names."""
    engines = payload.get("unresponsive_engines") or []
    arxiv_named = False
    arxiv_reason = ""
    others: List[str] = []
    if not isinstance(engines, list):
        return False, "", []
    for item in engines:
        if isinstance(item, (list, tuple)) and item:
            name = str(item[0])
            reason = str(item[1]) if len(item) > 1 else ""
        elif isinstance(item, str) and item:
            name = item
            reason = ""
        else:
            continue
        if name.lower() == "arxiv":
            arxiv_named = True
            arxiv_reason = reason
        else:
            others.append(name)
    return arxiv_named, arxiv_reason, others


def classify_searxng(status_code: Optional[int], payload: Any, error: Optional[str]) -> Dict[str, Any]:
    if error or status_code is None:
        return make_check(
            "searxng",
            "SearXNG",
            "down",
            "No answer from SearXNG.",
            "./workstation logs searxng",
        )
    if status_code == 403:
        return make_check(
            "searxng",
            "SearXNG",
            "down",
            "JSON search is turned off.",
            "Set search.formats to include json in the SearXNG settings.",
        )
    if status_code != 200 or not isinstance(payload, dict):
        return make_check(
            "searxng",
            "SearXNG",
            "down",
            f"SearXNG answered HTTP {status_code}.",
            "./workstation logs searxng",
        )
    # Discovery uses the arXiv engine only. Another engine being benched is
    # information: it must not paint SearXNG DOWN, which is what a status
    # search of every engine was doing to itself.
    arxiv_named, arxiv_reason, others = _split_unresponsive_engines(payload)
    if arxiv_named:
        lowered = arxiv_reason.lower()
        if "suspend" in lowered or "too many" in lowered:
            summary = "The arXiv engine is suspended for about an hour."
        else:
            summary = "The arXiv engine is unavailable."
        return make_check(
            "searxng",
            "SearXNG",
            "down",
            summary,
            "Wait, then try discovery again. SearXNG still returns HTTP 200 "
            "with an empty result list while an engine is benched.",
        )
    if others:
        listed = ", ".join(others)
        return make_check(
            "searxng",
            "SearXNG",
            "ok",
            f"arXiv is answering. Other engines are unavailable: {listed}.",
            None,
        )
    return make_check("searxng", "SearXNG", "ok", "JSON search is answering.", None)


def classify_column(status_code: Optional[int], body: Any) -> str:
    """``applied``, ``missing`` (PGRST204), or ``unknown``."""
    if status_code == 200:
        return "applied"
    blob = body if isinstance(body, dict) else {}
    code = str(blob.get("code", ""))
    message = str(blob.get("message", ""))
    if code == "PGRST204" or "Could not find" in message:
        return "missing"
    return "unknown"


def column_check(spec: Dict[str, str], verdict: str) -> Dict[str, Any]:
    if verdict == "applied":
        return make_check(spec["id"], spec["label"], "ok", "Applied.", None)
    if verdict == "missing":
        return make_check(
            spec["id"],
            spec["label"],
            "down",
            spec["missing"],
            f"Apply {spec['file']} by hand, then reload the PostgREST schema.",
        )
    return make_check(
        spec["id"],
        spec["label"],
        "down",
        "Could not check this migration.",
        "Confirm PostgREST is up, then open Activity again.",
    )


def classify_kind(status_code: Optional[int], body: Any) -> str:
    if status_code in (200, 201):
        return "applied"
    text = body if isinstance(body, str) else ""
    if isinstance(body, dict):
        text = str(body.get("code", "")) + " " + str(body.get("message", ""))
    if "23514" in text or "kind_check" in text or "trigger_source_check" in text:
        return "missing"
    return "unknown"


def kind_check(spec: Dict[str, str], verdict: str) -> Dict[str, Any]:
    return column_check(
        {
            "id": spec["id"],
            "label": spec["label"],
            "missing": spec["missing"],
            "file": spec["file"],
        },
        verdict,
    )


def key_presence(settings: Any) -> Dict[str, bool]:
    def present(value: Optional[str]) -> bool:
        return bool(value and str(value).strip())

    return {
        "openai": present(getattr(settings, "openai_api_key", None)),
        "anthropic": present(getattr(settings, "anthropic_api_key", None)),
        "deepseek": present(getattr(settings, "deepseek_api_key", None)),
    }


def mine_key_failure(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    for row in rows:
        error = str(row.get("error") or "")
        lowered = error.lower()
        for needle, provider, summary, fix in _KEY_FAILURES:
            if needle in lowered:
                found = {
                    "provider": provider,
                    "code": needle,
                    "summary": summary,
                    "fix": fix,
                }
                created = row.get("created_at")
                if created:
                    found["at"] = str(created)
                return found
    return None


def classify_keys(
    presence: Dict[str, bool],
    failure: Optional[Dict[str, str]],
    history_unreadable: bool,
) -> Dict[str, Any]:
    labels = (("openai", "OpenAI"), ("anthropic", "Anthropic"), ("deepseek", "DeepSeek"))
    set_names = [label for key, label in labels if presence.get(key)]
    missing = [label for key, label in labels if not presence.get(key)]
    if not missing:
        summary = "OpenAI, Anthropic and DeepSeek keys are set."
    elif not set_names:
        summary = _join(missing) + (" key is" if len(missing) == 1 else " keys are") + " missing."
    else:
        summary = (
            _join(set_names)
            + (" key is" if len(set_names) == 1 else " keys are")
            + " set. "
            + _join(missing)
            + (" key is" if len(missing) == 1 else " keys are")
            + " missing."
        )
    if history_unreadable:
        summary += " Recent run errors could not be read."
    fix = "Add the missing key to .env." if missing else None
    state = "down" if missing else "ok"
    if failure:
        summary += " Last failure: " + failure["summary"]
        state = "down"
        if not missing:
            fix = failure["fix"]
    return make_check("keys", "Model keys", state, summary, fix)


def classify_scheduler(enabled: bool, last: Optional[Dict[str, Any]], read_error: bool) -> Dict[str, Any]:
    if read_error:
        return make_check(
            "scheduler",
            "Scheduler",
            "down",
            "Could not read scheduled runs.",
            "./workstation logs scheduler",
        )
    if not enabled:
        return make_check(
            "scheduler",
            "Scheduler",
            "ok",
            "Scheduling is off. An exited scheduler is normal.",
            None,
        )
    if not last:
        return make_check(
            "scheduler",
            "Scheduler",
            "ok",
            "Scheduling is on. No scheduled run has been recorded yet.",
            None,
        )
    status = str(last.get("status") or "unknown")
    if status == "failed":
        return make_check(
            "scheduler",
            "Scheduler",
            "down",
            "The last scheduled run failed.",
            "Open that run on this page for the reason.",
        )
    return make_check(
        "scheduler",
        "Scheduler",
        "ok",
        f"The last scheduled run {status}.",
        None,
    )


def classify_tts(payload: Dict[str, Any]) -> Dict[str, Any]:
    command = str(payload.get("start_command") or "").strip()
    workstation = "./workstation up --tts"
    if payload.get("ready"):
        return make_check("indextts", "IndexTTS", "ok", payload.get("message") or "IndexTTS is ready.", None)
    fix = command
    if workstation not in fix:
        fix = (fix + "\n" if fix else "") + workstation
    return make_check(
        "indextts",
        "IndexTTS",
        "down",
        public_line(str(payload.get("message") or "IndexTTS is not running.")),
        fix,
    )


def classify_voices(payload: Dict[str, Any]) -> Dict[str, Any]:
    voices = payload.get("voices") or []
    usable = 0
    if isinstance(voices, list):
        usable = sum(1 for voice in voices if isinstance(voice, dict) and voice.get("usability") == "usable")
    if usable:
        noun = "voice" if usable == 1 else "voices"
        return make_check("voices", "Voices", "ok", f"{usable} usable {noun}.", None)
    return make_check(
        "voices",
        "Voices",
        "down",
        "No usable voices.",
        "Add a WAV reference in the voices folder.",
    )


def classify_storage(uploads: Dict[str, Any], audio: Dict[str, Any]) -> Dict[str, Any]:
    def phrase(name: str, usage: Dict[str, Any]) -> str:
        if not usage.get("present"):
            return f"{name} folder is not there yet"
        extra = " or more" if usage.get("truncated") else ""
        return f"{name} {human_bytes(int(usage.get('bytes') or 0))} ({usage.get('files') or 0}{extra} files)"

    return make_check(
        "storage",
        "Storage",
        "ok",
        phrase("Uploads", uploads) + ". " + phrase("Audio", audio) + ".",
        None,
    )


def directory_usage(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"present": False, "bytes": 0, "files": 0, "truncated": False}
    total = 0
    files = 0
    truncated = False
    for entry in path.rglob("*"):
        if not entry.is_file():
            continue
        files += 1
        try:
            total += entry.stat().st_size
        except OSError:
            continue
        if files >= MAX_STORAGE_FILES:
            truncated = True
            break
    return {"present": True, "bytes": total, "files": files, "truncated": truncated}


def _join(names: Sequence[str]) -> str:
    items = list(names)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _auth_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if token:
        headers["apikey"] = token
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _json_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


async def _guard(
    check_id: str,
    label: str,
    factory: Callable[[], Any],
    secrets: Sequence[Optional[str]],
    *,
    fatal: bool = True,
) -> Dict[str, Any]:
    try:
        return await asyncio.wait_for(factory(), timeout=OVERALL_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.warning("status check %s failed: %s", check_id, type(exc).__name__)
        summary = public_line(redact(str(exc), secrets)) or "Check failed."
        return make_check(check_id, label, "down", summary, None, fatal=fatal)


def migration_table_specs(migrations_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Status rows for table migrations, from the SQL files themselves.

    ``011``–``016`` are columns and pipeline kinds, probed above. From
    ``017`` on, each migration that creates a table is one row. A future
    file is picked up without editing this module.
    """
    directory = migrations_dir
    if directory is None:
        directory = Path(__file__).resolve().parents[2] / "docs" / "migrations"
    specs: List[Dict[str, Any]] = []
    for path in sorted(directory.glob("[0-9][0-9][0-9]_*.sql")):
        number = int(path.name[:3])
        if number < _MIGRATION_TABLES_FROM:
            continue
        tables: List[str] = []
        for name in _TABLE_SQL.findall(path.read_text(encoding="utf-8")):
            if name not in tables:
                tables.append(name)
        if not tables:
            continue
        words = path.stem.split("_", 1)[1].replace("_", " ")
        rel = f"docs/migrations/{path.name}"
        specs.append(
            {
                "id": f"migration-{number:03d}",
                "label": f"{number:03d} {words}",
                "tables": tables,
                "file": rel,
                "missing": (
                    f"{', '.join(tables)} is missing. Apply {rel} by hand."
                ),
            }
        )
    return specs


def _unchecked(spec: Dict[str, Any]) -> Dict[str, Any]:
    return make_check(
        spec["id"],
        spec["label"],
        "down",
        "Not checked because the database is not answering.",
        "./workstation up",
    )


async def _check_postgrest(client: httpx.AsyncClient, base: str, headers: Dict[str, str]) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        response = await client.get(
            f"{base}/pillars",
            params={"select": "id", "limit": "1"},
            headers=headers,
        )
    except httpx.HTTPError:
        return make_check(
            "postgrest",
            "Database",
            "down",
            "No answer from PostgREST.",
            "./workstation up",
        )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if response.status_code == 200:
        return make_check("postgrest", "Database", "ok", f"Answered in {elapsed_ms} ms.", None)
    if response.status_code in (401, 403):
        return make_check(
            "postgrest",
            "Database",
            "down",
            "PostgREST rejected the request token.",
            "Check that SUPABASE_KEY matches PGRST_JWT_SECRET.",
        )
    return make_check(
        "postgrest",
        "Database",
        "down",
        f"PostgREST answered HTTP {response.status_code}.",
        "./workstation up",
    )


async def _probe_column(
    client: httpx.AsyncClient,
    base: str,
    headers: Dict[str, str],
    spec: Dict[str, str],
) -> Dict[str, Any]:
    try:
        response = await client.get(
            f"{base}/{spec['table']}",
            params={"select": spec["column"], "limit": "0"},
            headers=headers,
        )
    except httpx.HTTPError:
        return column_check(spec, "unknown")
    return column_check(spec, classify_column(response.status_code, _json_body(response)))


async def _delete_probes(client: httpx.AsyncClient, base: str, headers: Dict[str, str]) -> None:
    try:
        await client.delete(
            f"{base}/pipeline_runs",
            params={"pillar_id": f"eq.{PROBE_PILLAR}", "status": "eq.interrupted"},
            headers={**headers, "Prefer": "return=minimal"},
        )
    except httpx.HTTPError:
        logger.warning("could not delete status-probe rows")


async def _probe_kind(
    client: httpx.AsyncClient,
    base: str,
    headers: Dict[str, str],
    spec: Dict[str, str],
) -> Dict[str, Any]:
    try:
        response = await client.post(
            f"{base}/pipeline_runs",
            headers={**headers, "Prefer": "tx=rollback,return=minimal"},
            json={
                "pillar_id": PROBE_PILLAR,
                "trigger_source": spec["trigger"],
                "kind": spec["kind"],
                "status": "interrupted",
            },
        )
    except httpx.HTTPError:
        return kind_check(spec, "unknown")
    return kind_check(spec, classify_kind(response.status_code, _json_body(response) or response.text))


def classify_table(status_code: Optional[int], body: Any) -> str:
    """Classify a table probe as applied, missing, or unknown."""
    if status_code == 200:
        return "applied"
    blob = body if isinstance(body, dict) else {}
    code = str(blob.get("code", ""))
    message = str(blob.get("message", ""))
    if code == "PGRST205" or "Could not find the table" in message:
        return "missing"
    return "unknown"


async def _probe_migration_tables(
    client: httpx.AsyncClient,
    base: str,
    headers: Dict[str, str],
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    missing: List[str] = []
    unknown = False
    for table in spec["tables"]:
        try:
            response = await client.get(
                f"{base}/{table}",
                params={"select": "*", "limit": "0"},
                headers=headers,
            )
        except httpx.HTTPError:
            unknown = True
            continue
        verdict = classify_table(response.status_code, _json_body(response))
        if verdict == "missing":
            missing.append(table)
        elif verdict != "applied":
            unknown = True
    if missing:
        narrowed = dict(spec)
        narrowed["missing"] = (
            f"{', '.join(missing)} is missing. Apply {spec['file']} by hand."
        )
        return column_check(narrowed, "missing")
    if unknown:
        return column_check(spec, "unknown")
    return column_check(spec, "applied")


async def _migration_checks(
    client: httpx.AsyncClient,
    base: str,
    headers: Dict[str, str],
    database_up: bool,
) -> List[Dict[str, Any]]:
    tables = migration_table_specs()
    if not database_up:
        return [
            _unchecked(spec) for spec in (*MIGRATION_COLUMNS, *KIND_PROBES, *tables)
        ]
    columns = await asyncio.gather(
        *(_probe_column(client, base, headers, spec) for spec in MIGRATION_COLUMNS)
    )
    await _delete_probes(client, base, headers)
    kinds = []
    for spec in KIND_PROBES:
        kinds.append(await _probe_kind(client, base, headers, spec))
    await _delete_probes(client, base, headers)
    table_rows = await asyncio.gather(
        *(
            _probe_migration_tables(client, base, headers, spec)
            for spec in tables
        )
    )
    return [*columns, *kinds, *table_rows]


def _qdrant_headers(api_key: Any) -> Dict[str, str]:
    if api_key:
        return {"api-key": str(api_key)}
    return {}


async def _probe_qdrant(
    client: httpx.AsyncClient,
    url: str,
    api_key: Any,
    *,
    target: str,
) -> Dict[str, Any]:
    try:
        response = await client.get(str(url).rstrip("/") + "/", headers=_qdrant_headers(api_key))
    except httpx.HTTPError as exc:
        return classify_qdrant(
            None, "", public_line(str(exc)) or "request failed", target=target,
        )
    return classify_qdrant(response.status_code, response.text, None, target=target)


async def _check_qdrant(client: httpx.AsyncClient, settings: Any) -> Dict[str, Any]:
    url = getattr(settings, "qdrant_url", None)
    if not url or not str(url).startswith(("http://", "https://")):
        return make_check(
            "qdrant",
            "Qdrant (local)",
            "down",
            "QDRANT_URL is not set.",
            "Set QDRANT_URL in .env.",
        )
    return await _probe_qdrant(
        client,
        str(url),
        getattr(settings, "qdrant_api_key", None),
        target="local",
    )


async def _check_qdrant_cloud(client: httpx.AsyncClient, settings: Any) -> Dict[str, Any]:
    """Optional cloud copy. Missing URL is not a check. A failure is non-fatal."""
    url = getattr(settings, "qdrant_cloud_url", None)
    if not url or not str(url).startswith(("http://", "https://")):
        return make_check(
            "qdrant-cloud",
            "Qdrant Cloud",
            "ok",
            "Not configured.",
            None,
            fatal=False,
        )
    return await _probe_qdrant(
        client,
        str(url),
        getattr(settings, "qdrant_cloud_api_key", None),
        target="cloud",
    )


def classify_cloud_reader(
    status: Dict[str, Any],
    rows: Optional[List[Dict[str, Any]]],
    read_error: Optional[str] = None,
    reviews: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The cloud reader row on Activity. Never fatal: the app works without the cloud.

    ``status`` is ``cloud_publish_service.current_status()``; ``rows`` are the
    ``cloud_publish_state`` rows (None when they could not be read); ``reviews``
    is ``cloud_review_sync_service.current_status()``, the pull of the phone's
    reviews, which adds a sentence and turns a healthy row down when it is failing.
    """
    check = _classify_publishing(status, rows, read_error)
    if not reviews or not reviews.get("enabled"):
        return check
    if reviews.get("state") == "failed":
        retry = reviews.get("retry_in_seconds")
        again = f" Trying again in {int(retry // 60) + 1} min." if retry else ""
        note = f" Phone reviews could not be pulled: {public_line(str(reviews.get('last_error') or 'unknown error'))}.{again}"
        check["summary"] += note
        if check["state"] == "ok":
            check["state"] = "down"
            check["fix"] = "Reviews made on the phone are safe in the cloud. Run: docker compose exec webui python -m nlp_pillars.cli cloud pull-reviews"
        return check
    summary = reviews.get("last_summary") or {}
    if reviews.get("last_success_at"):
        got = int(summary.get("applied") or 0)
        check["summary"] += f" Phone reviews checked {reviews['last_success_at'].replace('T', ' ').replace('Z', ' UTC')}" + (
            f" ({got} new)." if got else "."
        )
    return check


def _classify_publishing(
    status: Dict[str, Any],
    rows: Optional[List[Dict[str, Any]]],
    read_error: Optional[str] = None,
) -> Dict[str, Any]:
    label = "Cloud reader"
    if not status.get("enabled"):
        return make_check(
            "cloud-reader", label, "ok",
            "Automatic publishing is off.",
            "Set CLOUD_PUBLISH_ENABLED=true in .env, then ./workstation up.",
            fatal=False,
        )
    if not status.get("configured"):
        missing = ", ".join(status.get("missing") or []) or "CLOUD_PUBLISH_URL, CF_ACCESS_CLIENT_ID, CF_ACCESS_CLIENT_SECRET"
        return make_check(
            "cloud-reader", label, "down",
            "Publishing is on but not configured.",
            f"Set {missing} in .env.",
            fatal=False,
        )
    held = ""
    if rows is None:
        held = f" Could not read what is published ({read_error or 'unknown error'})."
    else:
        audio = [r for r in rows if str(r.get("doc_key", "")).startswith("audio/")]
        docs = [r for r in rows if r.get("doc_key") != "manifest.json" and r not in audio]
        size = sum(int(r.get("byte_size") or 0) for r in rows)
        manifest = next((r for r in rows if r.get("doc_key") == "manifest.json"), None)
        when = f", manifest written {str(manifest['published_at'])[:16].replace('T', ' ')} UTC" if manifest and manifest.get("published_at") else ""
        held = f" The cloud holds {len(docs)} documents and {len(audio)} audio files ({human_bytes(size)}){when}."
    state = status.get("state")
    if state == "failed":
        retry = status.get("retry_in_seconds")
        again = f" Trying again in {int(retry // 60) + 1} min." if retry else ""
        return make_check(
            "cloud-reader", label, "down",
            f"The last publish failed: {public_line(str(status.get('last_error') or 'unknown error'))}.{again}{held}",
            "The app is unaffected. Fix the cause, or run: docker compose exec webui python -m nlp_pillars.cli cloud publish",
            fatal=False,
        )
    if state == "publishing":
        return make_check("cloud-reader", label, "ok", f"Publishing now ({status.get('message') or 'working'}).{held}", None, fatal=False)
    if state == "waiting":
        return make_check("cloud-reader", label, "ok", f"A publish is queued.{held}", None, fatal=False)
    ran = status.get("last_success_at")
    prefix = f"Last publish {ran.replace('T', ' ').replace('Z', ' UTC')}." if ran else "Nothing published since the app started."
    return make_check("cloud-reader", label, "ok", f"{prefix}{held}", None, fatal=False)


async def _check_cloud_reader(
    client: httpx.AsyncClient, base: str, headers: Dict[str, str], settings: Any,
    database_ok: bool,
) -> Dict[str, Any]:
    from webui.services import cloud_publish_service, cloud_review_sync_service

    status = cloud_publish_service.current_status(settings)
    reviews = cloud_review_sync_service.current_status()
    if not status.get("enabled") or not status.get("configured"):
        return classify_cloud_reader(status, None)
    if not database_ok:
        return classify_cloud_reader(status, None, "database is not answering", reviews)
    try:
        response = await client.get(
            f"{base}/cloud_publish_state",
            params={"select": "doc_key,byte_size,published_at", "limit": "5000"},
            headers=headers,
        )
        if response.status_code >= 400:
            return classify_cloud_reader(status, None, f"HTTP {response.status_code}; is {STATE_MIGRATION} applied?", reviews)
        rows = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return classify_cloud_reader(status, None, type(exc).__name__, reviews)
    return classify_cloud_reader(status, rows if isinstance(rows, list) else None, None, reviews)


def classify_playground_runner(
    url: Optional[str], status_code: Optional[int], payload: Any, error: Optional[str]
) -> Dict[str, Any]:
    """Activity row for the playground runner. Non-fatal: only Playground code
    cells need it, so a stopped runner must not turn the shell chip red."""
    if not url:
        return make_check(
            "playground-runner", "Playground runner", "ok", "Not configured.", None, fatal=False
        )
    if status_code == 200 and isinstance(payload, dict) and payload.get("status") == "ok":
        sessions = payload.get("sessions", 0)
        limit = payload.get("max_sessions", "?")
        return make_check(
            "playground-runner",
            "Playground runner",
            "ok",
            f"Ready. {sessions} of {limit} code sessions open.",
            None,
            fatal=False,
        )
    why = error or (f"HTTP {status_code}" if status_code is not None else "no answer")
    return make_check(
        "playground-runner",
        "Playground runner",
        "down",
        public_line(f"Not reachable at {url} ({why}). Playground code cells cannot run."),
        "./workstation up",
        fatal=False,
    )


async def _check_playground_runner(client: httpx.AsyncClient, settings: Any) -> Dict[str, Any]:
    url = getattr(settings, "playground_runner_url", None)
    if not url:
        return classify_playground_runner(None, None, None, None)
    base = str(url).rstrip("/")
    try:
        response = await client.get(base + "/health")
    except httpx.HTTPError as exc:
        return classify_playground_runner(base, None, None, type(exc).__name__)
    return classify_playground_runner(base, response.status_code, _json_body(response), None)


async def _check_searxng(
    client: httpx.AsyncClient,
    settings: Any,
    *,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    url = getattr(settings, "searxng_url", None)
    if not url or not str(url).startswith(("http://", "https://")):
        return make_check(
            "searxng",
            "SearXNG",
            "down",
            "SEARXNG_URL is not set.",
            "Set SEARXNG_URL in .env.",
        )
    key = str(url).rstrip("/")
    clock = _clock(now)
    cached = _searxng_cache.get(key)
    if cached is not None and clock - cached[0] < SEARXNG_CACHE_SECONDS:
        return dict(cached[1])
    try:
        response = await client.get(
            key + "/search",
            params={"q": "nlp", "format": "json", "engines": "arxiv"},
        )
    except httpx.HTTPError as exc:
        check = classify_searxng(None, None, public_line(str(exc)) or "request failed")
    else:
        check = classify_searxng(response.status_code, _json_body(response), None)
    _searxng_cache[key] = (clock, check)
    return dict(check)


def classify_semantic_scholar(
    status_code: Optional[int], *, had_key: bool
) -> Dict[str, Any]:
    """Activity row for the Semantic Scholar key.

    A rejected key is DOWN on the page and does not flip the shell chip:
    the client treats it as absent and runs keyless, so citations still run.
    """
    if not had_key:
        return make_check(
            "semantic-scholar",
            "Semantic Scholar",
            "ok",
            "No API key configured. Requests use the shared rate limit.",
            None,
            fatal=False,
        )
    if status_code == 403:
        return make_check(
            "semantic-scholar",
            "Semantic Scholar",
            "down",
            (
                "The configured SEMANTIC_SCHOLAR_API_KEY is rejected "
                "(HTTP 403). Requests run without it, keyless."
            ),
            (
                "Renew the key or remove SEMANTIC_SCHOLAR_API_KEY from .env. "
                "The file was not changed."
            ),
            fatal=False,
        )
    if status_code == 200:
        return make_check(
            "semantic-scholar",
            "Semantic Scholar",
            "ok",
            "API key accepted.",
            None,
            fatal=False,
        )
    if status_code is None:
        summary = "Semantic Scholar could not be reached."
    else:
        summary = f"Semantic Scholar answered HTTP {status_code}."
    return make_check(
        "semantic-scholar",
        "Semantic Scholar",
        "down",
        summary,
        "Open Activity again in a moment.",
        fatal=False,
    )


def _semantic_scholar_key(settings: Any) -> str:
    raw = getattr(settings, "semantic_scholar_api_key", None)
    return str(raw).strip() if raw else ""


async def _check_semantic_scholar(
    client: httpx.AsyncClient,
    settings: Any,
    *,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    key = _semantic_scholar_key(settings)
    if not key:
        return classify_semantic_scholar(None, had_key=False)
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    clock = _clock(now)
    cached = _s2_cache.get(digest)
    if cached is not None and clock - cached[0] < S2_CACHE_SECONDS:
        return dict(cached[1])
    try:
        response = await client.get(
            S2_PROBE_URL,
            params={"fields": "paperId"},
            headers={"x-api-key": key},
        )
    except httpx.HTTPError as exc:
        check = classify_semantic_scholar(None, had_key=True)
        detail = public_line(str(exc))
        if detail:
            check = dict(check)
            check["summary"] = public_line(
                f"Semantic Scholar could not be reached. {detail}"
            )
    else:
        check = classify_semantic_scholar(response.status_code, had_key=True)
        if response.status_code == 403:
            logger.error(
                "SEMANTIC_SCHOLAR_API_KEY was rejected with HTTP 403. "
                "Citation requests run keyless. "
                "The key in .env was not changed; renew or remove it."
            )
    _s2_cache[digest] = (clock, check)
    return dict(check)


async def _recent_errors(
    client: httpx.AsyncClient, base: str, headers: Dict[str, str]
) -> List[Dict[str, Any]]:
    response = await client.get(
        f"{base}/pipeline_runs",
        params={
            "select": "error,created_at,kind",
            "error": "not.is.null",
            "order": "created_at.desc",
            "limit": "20",
        },
        headers=headers,
    )
    response.raise_for_status()
    body = _json_body(response)
    return body if isinstance(body, list) else []


async def _last_scheduled(
    client: httpx.AsyncClient, base: str, headers: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    response = await client.get(
        f"{base}/pipeline_runs",
        params={
            "select": "id,status,created_at",
            "trigger_source": "eq.scheduler",
            "order": "created_at.desc",
            "limit": "1",
        },
        headers=headers,
    )
    response.raise_for_status()
    body = _json_body(response)
    if isinstance(body, list) and body:
        return body[0]
    return None


async def _call_sync(fn: Callable[[], Any]) -> Any:
    if asyncio.iscoroutinefunction(fn):
        return await fn()
    return await asyncio.to_thread(fn)


async def check_voices(
    list_voices: Callable[[], Any],
    *,
    now: Optional[float] = None,
    timeout: float = VOICES_TIMEOUT_SECONDS,
    secrets: Sequence[Optional[str]] = (),
) -> Dict[str, Any]:
    """Scan the voice library, or reuse a recent result when the scan cannot finish.

    A fresh result is reused for ``VOICES_CACHE_SECONDS`` without scanning again.
    If the scan then times out and the last good result is still inside
    ``VOICES_LAST_GOOD_SECONDS``, that result is returned with its age and the
    row stays up. The failed attempt also starts a cache window, so a busy CPU
    does not pay the timeout on every page.
    """
    clock = _clock(now)
    checked_at = _voices_state["checked_at"]
    if (
        checked_at is not None
        and clock - checked_at < VOICES_CACHE_SECONDS
        and _voices_state["latest"] is not None
    ):
        if _voices_state["fallback"] and _voices_state["good"] is not None:
            return _with_checked_age(_voices_state["good"], clock - _voices_state["good_at"])
        return dict(_voices_state["latest"])

    async def run() -> Dict[str, Any]:
        payload = await _call_sync(list_voices)
        if not isinstance(payload, dict):
            raise TypeError("voice list was not an object")
        return classify_voices(payload)

    try:
        check = await asyncio.wait_for(run(), timeout=timeout)
    except Exception as exc:
        logger.warning("status check voices failed: %s", type(exc).__name__)
        good = _voices_state["good"]
        good_at = _voices_state["good_at"]
        if (
            isinstance(good, dict)
            and good.get("state") == "ok"
            and good_at is not None
            and clock - good_at <= VOICES_LAST_GOOD_SECONDS
        ):
            _voices_state["checked_at"] = clock
            _voices_state["fallback"] = True
            _voices_state["latest"] = good
            return _with_checked_age(good, clock - good_at)
        summary = public_line(redact(str(exc), secrets)) or "Check failed."
        down = make_check("voices", "Voices", "down", summary, None)
        _voices_state["checked_at"] = clock
        _voices_state["fallback"] = False
        _voices_state["latest"] = down
        return dict(down)

    _voices_state["checked_at"] = clock
    _voices_state["fallback"] = False
    _voices_state["latest"] = check
    if check.get("state") == "ok":
        _voices_state["good"] = check
        _voices_state["good_at"] = clock
    return dict(check)


def _default_storage(settings: Any) -> Dict[str, Any]:
    upload_dir = Path(os.environ.get("UPLOAD_DIR", "data/uploads"))
    audio_dir = Path(getattr(settings, "podcast_audio_dir", "/app/data/podcast_audio"))
    return classify_storage(directory_usage(upload_dir), directory_usage(audio_dir))


async def collect_status(
    *,
    client: httpx.AsyncClient,
    settings: Any,
    postgrest_url: str,
    auth_token: Optional[str] = None,
    tts_status: Optional[Callable[[], Dict[str, Any]]] = None,
    list_voices: Optional[Callable[[], Dict[str, Any]]] = None,
    storage: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Collect every check. Always returns a payload; never raises for a check."""
    secrets = [
        auth_token,
        getattr(settings, "qdrant_api_key", None),
        getattr(settings, "qdrant_cloud_api_key", None),
        getattr(settings, "openai_api_key", None),
        getattr(settings, "anthropic_api_key", None),
        getattr(settings, "deepseek_api_key", None),
        getattr(settings, "semantic_scholar_api_key", None),
    ]
    headers = _auth_headers(auth_token)
    base = postgrest_url.rstrip("/")

    web = make_check("web", "Web app", "ok", "Responding.", None)
    database = await _guard(
        "postgrest",
        "Database",
        lambda: _check_postgrest(client, base, headers),
        secrets,
    )

    async def safe_migrations() -> List[Dict[str, Any]]:
        try:
            return await _migration_checks(client, base, headers, database["state"] == "ok")
        except Exception as exc:
            logger.warning("status check migrations failed: %s", type(exc).__name__)
            summary = public_line(redact(str(exc), secrets)) or "Could not check migrations."
            return [make_check("migrations", "Migrations", "down", summary, None)]

    async def keys() -> Dict[str, Any]:
        failure = None
        history_unreadable = False
        if database["state"] == "ok":
            try:
                failure = mine_key_failure(await _recent_errors(client, base, headers))
            except (httpx.HTTPError, ValueError):
                history_unreadable = True
        return classify_keys(key_presence(settings), failure, history_unreadable)

    async def scheduler() -> Dict[str, Any]:
        if database["state"] != "ok":
            return make_check(
                "scheduler",
                "Scheduler",
                "down",
                "Not checked because the database is not answering.",
                "./workstation up",
            )
        try:
            last = await _last_scheduled(client, base, headers)
        except (httpx.HTTPError, ValueError):
            return classify_scheduler(bool(getattr(settings, "schedule_enabled", False)), None, True)
        return classify_scheduler(bool(getattr(settings, "schedule_enabled", False)), last, False)

    async def tts() -> Dict[str, Any]:
        fn = tts_status or _default_tts
        payload = await _call_sync(fn)
        if not isinstance(payload, dict):
            raise TypeError("IndexTTS status was not an object")
        return classify_tts(payload)

    async def voices() -> Dict[str, Any]:
        fn = list_voices or _default_voices
        return await check_voices(fn, secrets=secrets)

    async def stored() -> Dict[str, Any]:
        fn = storage or (lambda: _default_storage(settings))
        payload = await _call_sync(fn)
        if isinstance(payload, dict) and payload.get("id") == "storage":
            return payload
        raise TypeError("storage check was not a status row")

    (
        migration_rows,
        qdrant,
        qdrant_cloud,
        searxng,
        scholar,
        indextts,
        voice_check,
        key_check,
        schedule,
        disk,
        cloud_reader,
        playground_runner,
    ) = await asyncio.gather(
        safe_migrations(),
        _guard("qdrant", "Qdrant (local)", lambda: _check_qdrant(client, settings), secrets),
        _guard(
            "qdrant-cloud",
            "Qdrant Cloud",
            lambda: _check_qdrant_cloud(client, settings),
            secrets,
            fatal=False,
        ),
        _guard("searxng", "SearXNG", lambda: _check_searxng(client, settings), secrets),
        _guard(
            "semantic-scholar",
            "Semantic Scholar",
            lambda: _check_semantic_scholar(client, settings),
            secrets,
            fatal=False,
        ),
        _guard("indextts", "IndexTTS", tts, secrets),
        voices(),
        _guard("keys", "Model keys", keys, secrets),
        _guard("scheduler", "Scheduler", scheduler, secrets),
        _guard("storage", "Storage", stored, secrets),
        _guard(
            "cloud-reader",
            "Cloud reader",
            lambda: _check_cloud_reader(client, base, headers, settings, database["state"] == "ok"),
            secrets,
            fatal=False,
        ),
        _guard(
            "playground-runner",
            "Playground runner",
            lambda: _check_playground_runner(client, settings),
            secrets,
            fatal=False,
        ),
    )
    checks = [
        web,
        database,
        *migration_rows,
        qdrant,
        qdrant_cloud,
        searxng,
        scholar,
        indextts,
        voice_check,
        key_check,
        schedule,
        disk,
        cloud_reader,
        playground_runner,
    ]
    down = sum(1 for check in checks if counts_as_down(check))
    return {
        "ok": down == 0,
        "down_count": down,
        "summary": public_summary(checks),
        "checks": checks,
    }


def _default_tts() -> Dict[str, Any]:
    from webui.services.podcast_audio_service import tts_status_payload

    return tts_status_payload()


def _default_voices() -> Dict[str, Any]:
    from webui.services.podcast_audio_service import list_voices_payload

    return list_voices_payload()


def public_summary(checks: List[Dict[str, Any]]) -> str:
    """One line for Today's date-row chip. That page prefers body.summary."""
    down = [check for check in checks if counts_as_down(check)]
    if not down:
        return "All services up"
    if len(down) == 1:
        label = str(down[0].get("label") or "A service").strip() or "A service"
        return f"{label} needs attention"
    return f"{len(down)} services need attention"


def failed_payload() -> Dict[str, Any]:
    """Used only when collection itself breaks. Still HTTP 200."""
    checks = [
        make_check(
            "status",
            "Status",
            "down",
            "Status could not be collected.",
            "Open Activity again in a moment.",
        )
    ]
    return {
        "ok": False,
        "down_count": 1,
        "summary": public_summary(checks),
        "checks": checks,
    }
