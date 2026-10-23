"""Add papers is one page: Find, upload, and auto-run.

/discovery/ opens on Find. /pipeline/ opens on auto-run and keeps ?pillar= and
?run= on that response (the script reads ?run=). The upload panel is the same
partial the pillar page includes, and the pillar id is passed in rather than
parsed from the URL.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from webui.routers import discovery as discovery_router
from webui.routers import pipeline as pipeline_router

ROOT = Path(__file__).resolve().parents[2]


class _Pillar:
    def __init__(self, pillar_id, name):
        self.id = pillar_id
        self.name = name


def _app():
    app = FastAPI()
    app.include_router(discovery_router.router)
    app.include_router(pipeline_router.router)
    app.state.templates = Jinja2Templates(directory=str(ROOT / "webui" / "templates"))
    return TestClient(app)


def _pillars(monkeypatch):
    monkeypatch.setattr(
        discovery_router,
        "get_pillars_or_empty",
        lambda limit=100: [_Pillar("ai-agents-tool-use", "Agents"), _Pillar("ai-safety-alignment", "Safety")],
    )


def test_discovery_opens_on_find_and_pipeline_opens_on_auto_run(monkeypatch):
    _pillars(monkeypatch)
    client = _app()

    find = client.get("/discovery/?pillar=ai-agents-tool-use")
    assert find.status_code == 200
    assert 'data-mode="find"' in find.text
    assert 'data-add-mode="find" hidden' not in find.text
    assert 'data-add-mode="upload" hidden' in find.text
    assert 'data-add-mode="auto" hidden' in find.text
    assert 'value="ai-agents-tool-use" selected' in find.text
    assert "Add papers" in find.text
    assert 'id="discover-form"' in find.text
    assert 'id="run-form"' in find.text
    assert 'id="pdf-upload-form"' in find.text
    assert 'id="queries-panel"' in find.text
    assert 'href="/activity/"' in find.text
    assert 'id="run-panel"' in find.text
    assert "is-floating" not in find.text

    auto = client.get("/pipeline/?pillar=ai-safety-alignment&run=run-123")
    assert auto.status_code == 200
    assert 'data-mode="auto"' in auto.text
    assert 'data-add-mode="auto" hidden' not in auto.text
    assert 'data-add-mode="find" hidden' in auto.text
    assert 'value="ai-safety-alignment" selected' in auto.text
    assert 'id="papers-input"' in auto.text
    assert 'value="1"' in auto.text

    upload = client.get("/discovery/?mode=upload")
    assert upload.status_code == 200
    assert 'data-mode="upload"' in upload.text
    assert 'data-add-mode="upload" hidden' not in upload.text

    overridden = client.get("/pipeline/?mode=find")
    assert overridden.status_code == 200
    assert 'data-mode="find"' in overridden.text


def test_upload_partial_is_shared_and_pillar_id_is_explicit():
    panel = (ROOT / "webui" / "templates" / "_upload_panel.html").read_text()
    pillar = (ROOT / "webui" / "templates" / "pillar_detail.html").read_text()
    add = (ROOT / "webui" / "templates" / "discovery.html").read_text()
    script = (ROOT / "webui" / "static" / "pillar-upload.js").read_text()

    assert '{% include "_upload_panel.html" %}' in pillar
    assert '{% include "_upload_panel.html" %}' in add
    assert 'name="run_summarizer"' in panel
    assert 'name="generate_quiz"' in panel
    assert 'id="file-generate-quiz"' in panel
    assert 'id="url-run-summarizer"' in panel
    assert 'id="upload-result"' in panel
    assert 'id="upload-status-container"' in panel
    assert "pathname.split" not in script
    assert "run.kind === 'upload'" in script
    assert "pillarId" in pillar
