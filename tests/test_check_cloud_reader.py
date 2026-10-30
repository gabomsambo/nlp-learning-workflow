"""scripts/check_cloud_reader.py: the guard must pass the real reader and fail on each violation.

Each failing case copies the real reader into a temp dir, makes one change a
careless commit might make, and expects the guard to name it. Without these,
a guard that quietly stopped matching would stay green forever.
"""

import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "check_cloud_reader.py"
READER = pathlib.Path("cloud") / "reader"


def run_guard(root: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """A copy of the reader's guarded files (generated public/static/ left out)."""
    shutil.copytree(
        REPO / READER,
        tmp_path / READER,
        ignore=shutil.ignore_patterns("static", "node_modules", ".wrangler", "scripts"),
    )
    return tmp_path


def append(root: pathlib.Path, rel: str, text: str) -> None:
    path = root / READER / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(existing + text, encoding="utf-8")


def replace(root: pathlib.Path, rel: str, old: str, new: str) -> None:
    path = root / READER / rel
    text = path.read_text(encoding="utf-8")
    assert old in text, f"fixture is stale: {old!r} not in {rel}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_real_reader_passes():
    result = run_guard(REPO)
    assert result.returncode == 0, result.stdout + result.stderr


def test_clean_copy_passes(tree):
    assert run_guard(tree).returncode == 0


# --- bindings --------------------------------------------------------------

BINDING_VIOLATIONS = {
    "workers_ai": '\n[ai]\nbinding = "AI"\n',
    "vectorize": '\n[[vectorize]]\nbinding = "VEC"\nindex_name = "x"\n',
    "queue": '\n[[queues.producers]]\nbinding = "Q"\nqueue = "q"\n',
    "service": '\n[[services]]\nbinding = "S"\nservice = "other"\n',
    "kv": '\n[[kv_namespaces]]\nbinding = "KV"\nid = "abc"\n',
    "durable_object": '\n[durable_objects]\nbindings = [{ name = "D", class_name = "D" }]\n',
    "extra_environment": '\n[env.staging]\nname = "nlp-reader-staging"\n',
    "model_key_var": '\n[vars.extra]\nOPENAI_API_KEY = "sk-x"\n',
}


@pytest.mark.parametrize("name", sorted(BINDING_VIOLATIONS))
def test_extra_binding_fails(tree, name):
    append(tree, "wrangler.toml", BINDING_VIOLATIONS[name])
    result = run_guard(tree)
    assert result.returncode == 1
    assert "wrangler.toml" in result.stdout


def test_unlisted_var_fails(tree):
    replace(tree, "wrangler.toml", "[vars]\n", '[vars]\nOPENAI_API_KEY = "sk-not-real"\n')
    result = run_guard(tree)
    assert result.returncode == 1
    assert "OPENAI_API_KEY" in result.stdout


def test_second_r2_bucket_fails(tree):
    append(tree, "wrangler.toml", '\n[[r2_buckets]]\nbinding = "OTHER"\nbucket_name = "other"\n')
    assert run_guard(tree).returncode == 1


def test_workers_dev_fails(tree):
    replace(tree, "wrangler.toml", "workers_dev = false", "workers_dev = true")
    assert run_guard(tree).returncode == 1


def test_smoke_config_is_checked_too(tree):
    append(tree, "test/wrangler.smoke.toml", '\n[ai]\nbinding = "AI"\n')
    result = run_guard(tree)
    assert result.returncode == 1
    assert "wrangler.smoke.toml" in result.stdout


def test_missing_wrangler_fails(tree):
    (tree / READER / "wrangler.toml").unlink()
    assert run_guard(tree).returncode == 1


# --- Worker source ---------------------------------------------------------

SOURCE_VIOLATIONS = {
    "fetch_call": "export async function leak(body) { return fetch('https://example.com/x', { body }); }\n",
    "globalthis_fetch": "export const leak = (u) => globalThis.fetch(u);\n",
    "binding_fetch": "export const leak = (env, u) => env.AI.fetch(u);\n",
    "url_literal": "export const ENDPOINT = 'https://api.openai.com/v1/chat/completions';\n",
    "url_in_template": "export const url = (m) => `https://api.example.com/${m}`;\n",
    "package_import": "import OpenAI from 'openai';\n",
    "node_import": "import { connect } from 'cloudflare:sockets';\n",
    "dynamic_import": "export const load = (u) => import(u);\n",
    "websocket": "export const ws = (u) => new WebSocket(u);\n",
    "eval": "export const run = (s) => eval(s);\n",
}


@pytest.mark.parametrize("name", sorted(SOURCE_VIOLATIONS))
def test_worker_violation_fails(tree, name):
    append(tree, "src/extra.js", SOURCE_VIOLATIONS[name])
    result = run_guard(tree)
    assert result.returncode == 1, result.stdout
    assert "src/extra.js" in result.stdout


def test_fetch_in_existing_worker_file_fails(tree):
    append(tree, "src/worker.js", "\nawait fetch('https://example.com');\n")
    result = run_guard(tree)
    assert result.returncode == 1
    assert "src/worker.js" in result.stdout


def test_second_certs_fetch_fails(tree):
    append(tree, "src/certs.js", "\nexport const again = (u) => fetchImpl(u);\n")
    assert run_guard(tree).returncode == 1


def test_csp_widened_fails(tree):
    replace(tree, "src/worker.js", "\"connect-src 'self'\"", "\"connect-src 'self' *\"")
    assert run_guard(tree).returncode == 1


def test_csp_connect_src_removed_fails(tree):
    replace(tree, "src/worker.js", "    \"connect-src 'self'\",\n", "")
    assert run_guard(tree).returncode == 1


def test_words_in_comments_and_strings_are_not_flagged(tree):
    append(
        tree,
        "src/extra.js",
        "// fetch('https://example.com') and new WebSocket(x) are what we do not do\n"
        "/* import x from 'openai'; */\n"
        "export const note = \"fetch( and WebSocket are words here\";\n"
        "export const header = 'sec-fetch-site';\n"
        "export const ok = (env, r) => env.ASSETS.fetch(r);\n"
        "export const issuer = (teamDomain) => `https://${teamDomain}`;\n"
        "export const rx = /https?:\\/\\/[a-z]+/;\n",
    )
    result = run_guard(tree)
    assert result.returncode == 0, result.stdout


# --- the page ---------------------------------------------------------------

PAGE_VIOLATIONS = {
    "absolute_url": ("public/js/extra.js", "export const u = 'https://example.com/track';\n"),
    "xhr": ("public/js/extra.js", "export const x = () => new XMLHttpRequest();\n"),
    "beacon": ("public/js/extra.js", "export const b = () => navigator.sendBeacon('/x');\n"),
    "protocol_relative_fetch": ("public/js/extra.js", "export const f = () => fetch('//example.com/x');\n"),
    "import_scripts_absolute": ("public/sw-extra.js", "importScripts('https://cdn.example.com/a.js');\n"),
    "import_scripts_protocol_relative": ("public/sw-extra.js", "importScripts('//cdn.example.com/a.js');\n"),
    "import_scripts_mixed": ("public/sw-extra.js", "importScripts('/sw-core.js', 'https://cdn.example.com/a.js');\n"),
    "import_scripts_computed": ("public/sw-extra.js", "importScripts(self.location.search);\n"),
    "external_script": ("public/extra.html", '<script src="https://cdn.example.com/a.js"></script>\n'),
    "external_css": ("public/extra.css", "@import url('https://fonts.example.com/f.css');\n"),
}


@pytest.mark.parametrize("name", sorted(PAGE_VIOLATIONS))
def test_page_violation_fails(tree, name):
    rel, text = PAGE_VIOLATIONS[name]
    append(tree, rel, text)
    result = run_guard(tree)
    assert result.returncode == 1, result.stdout
    assert rel in result.stdout


def test_import_scripts_of_root_relative_files_passes(tree):
    append(tree, "public/sw-extra.js", "importScripts('/sw-core.js', '/shell.js');\n")
    result = run_guard(tree)
    assert result.returncode == 0, result.stdout


def test_page_same_origin_fetch_and_svg_namespace_pass(tree):
    append(
        tree,
        "public/js/extra.js",
        "export const f = () => fetch('/api/manifest', { credentials: 'same-origin' });\n"
        "export const ns = 'http://www.w3.org/2000/svg';\n",
    )
    assert run_guard(tree).returncode == 0
