"""Shell foundation: themes, navigation, and shared components.

These assertions lock the contract later page PRs build on. They render templates
only — nothing here talks to a database or the network.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "webui" / "templates"
STATIC = ROOT / "webui" / "static"


class _URL:
    def __init__(self, path: str) -> None:
        self.path = path


class _Request:
    def __init__(self, path: str) -> None:
        self.url = _URL(path)


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html", "xml"]),
    )


def _render(path: str, title: str | None = None) -> str:
    return _env().get_template("base.html").render(request=_Request(path), title=title)


def test_tokens_define_every_theme_mode_and_density():
    css = (STATIC / "tokens.css").read_text()
    for theme in ("paper", "cockpit", "journal"):
        assert f'data-theme="{theme}"' in css
    assert 'data-mode="dark"' in css
    assert 'prefers-color-scheme: dark' in css
    assert 'data-density="compact"' in css
    for slug in (
        "formal-linguistics-nlp",
        "neural-architectures-language",
        "llm-theory-practice",
        "computational-semantics",
        "model-interpretability",
        "ai-agents-tool-use",
        "ml-systems-production",
        "ai-safety-alignment",
    ):
        assert f".pillar-{slug}" in css
    assert "--pico-background-color: var(--bg)" in css
    assert "--primary: var(--accent)" in css
    assert "--muted-color: var(--muted)" in css


def test_component_stylesheet_has_no_raw_colours():
    import re

    css = (STATIC / "app.css").read_text()
    assert re.search(r"#[0-9a-fA-F]{3,8}", css) is None
    for line in css.splitlines():
        if "font-family:" in line:
            assert "var(--font" in line


def test_pico_is_vendored_and_fonts_are_local():
    base = (TEMPLATES / "base.html").read_text()
    assert "cdn.jsdelivr.net" not in base
    assert "/static/vendor/pico.min.css" in base
    assert (STATIC / "vendor" / "pico.min.css").is_file()
    assert "Pico CSS" in (STATIC / "vendor" / "pico.min.css").read_text(encoding="utf-8")[:200]
    for name in (
        "inter-latin.woff2",
        "source-serif-4-latin.woff2",
        "jetbrains-mono-latin.woff2",
        "ibm-plex-mono-400-latin.woff2",
        "ibm-plex-mono-500-latin.woff2",
        "dm-sans-latin.woff2",
        "fraunces-latin.woff2",
        "fraunces-italic-latin.woff2",
    ):
        blob = (STATIC / "fonts" / name).read_bytes()
        assert blob[:4] == b"wOF2"
    assert (STATIC / "fonts" / "licenses" / "inter-OFL.txt").is_file()


def test_shell_navigation_and_settings_persistence():
    home = _render("/")
    assert "<title>NLP Learning</title>" in home
    assert 'href="/"' in home
    assert 'href="/quiz/"' in home
    assert 'href="/papers/"' in home
    assert 'href="/lessons/"' in home
    assert 'href="/podcast/"' in home
    assert 'href="/discovery/"' in home
    assert 'href="/pillars/"' in home
    assert 'href="/pipeline/"' in home
    assert 'href="/analytics/"' in home
    assert "nlp:theme" in home
    assert "nlp:mode" in home
    assert "nlp:density" in home
    assert 'id="toasts"' in home
    assert 'href="#main"' in home
    assert ">0<" not in home

    papers = _render("/papers/", title="Library")
    assert "<title>Library</title>" in papers
    assert 'href="/papers/" aria-current="page"' in papers
    assert 'href="/quiz/" aria-current="page"' not in papers
    lessons = _render("/lessons/")
    assert 'href="/lessons/" aria-current="page"' in lessons


def test_page_titles_are_distinct():
    env = _env()
    titles = {
        "dashboard.html": "Today",
        "papers.html": "Library",
        "lessons.html": "Lessons",
        "quiz.html": "Review",
        "pipeline.html": "Activity",
        "analytics.html": "Analytics",
        "pillars.html": "Pillars",
    }
    for name, expected in titles.items():
        text = (TEMPLATES / name).read_text()
        assert expected in text.split("{% block content %}", 1)[0]


def test_shared_macros_render():
    env = _env()
    html = env.from_string(
        "{% from '_components.html' import button, stat_tile, chip, empty_state, pillar_tile %}"
        "{{ button('Save') }}"
        "{{ stat_tile('Papers', unavailable=True) }}"
        "{{ chip('Due', 'warn') }}"
        "{{ empty_state('Nothing here', 'Add a paper to begin.', '/discovery/', 'Add papers') }}"
        "{{ pillar_tile('Agents', 'ai-agents-tool-use') }}"
    ).render()
    assert 'class="btn btn-primary btn-md"' in html
    assert "Count unavailable" in html
    assert "chip-warn" in html
    assert "pillar-ai-agents-tool-use" in html
    assert 'href="/pillars/ai-agents-tool-use"' in html
    assert ">0<" not in html
