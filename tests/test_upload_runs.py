"""Manual upload as a background run: dispatch, stages, outcome, and concurrency.

The concurrency tests are the point of this file. The captain's ruling is that an
upload must never be refused because a discovery run is in flight on the same pillar,
and that rule lives in ONE place — the partial unique index in
``docs/migrations/015_upload_runs.sql``. So these tests execute that file's own index
DDL rather than restating the predicate: sqlite supports partial unique indexes with
the same ``WHERE`` syntax, so the guard can be exercised for real without a database.
A rewrite of 015 that drops the exemption fails here.

Fixtures are local and the basename is unique across tests/ — there is no conftest.py
and no ``__init__.py``, so pytest's default import mode would collide two same-named
modules. Nothing here reaches a database, a scheduler or the network.
"""

import re
import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nlp_pillars.orchestrator import RunCancelledError
from nlp_pillars.schemas import UPLOAD_STAGES, RunStatus, StageName, StageStatus
from nlp_pillars.services import upload_service as upload_service_module
from nlp_pillars.services.upload_service import PipelineOutcome, UploadJobResult
from webui.services import run_service
from webui.services.run_service import (
    KIND_DISCOVER,
    KIND_UPLOAD,
    TRIGGER_UI_UPLOAD,
    UPLOAD_SOURCE_FILE,
    UPLOAD_SOURCE_URL,
    dispatch_run,
    execute_run,
)

PILLAR = "neural-architectures-language"

MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "docs" / "migrations" / "015_upload_runs.sql"
).read_text()


# ------------------------------------------------------------------- migration


def _statement(pattern: str) -> str:
    match = re.search(pattern, MIGRATION, re.S | re.I)
    assert match, f"migration 015 no longer contains a statement matching {pattern!r}"
    return match.group(0)


def test_the_migration_widens_both_check_constraints_without_dropping_anything():
    """Both constraints are DROP-and-re-ADD, so an omitted value is silently retired.

    Migration 014 did the same thing to add 'podcast_audio'; leaving one of its values
    out here would break podcast audio generation with a constraint violation that
    points at uploads.
    """
    kind = _statement(r"ADD CONSTRAINT pipeline_runs_kind_check.*?;")
    for value in (
        "run_daily", "process_selected", "discover", "podcast_audio", "upload"
    ):
        assert f"'{value}'" in kind

    trigger = _statement(r"ADD CONSTRAINT pipeline_runs_trigger_source_check.*?;")
    for value in (
        "ui_pipeline", "ui_select", "ui_discover", "scheduler",
        "ui_podcast_audio", "ui_upload",
    ):
        assert f"'{value}'" in trigger


def test_the_migration_ends_by_reloading_postgrests_schema_cache():
    """Without it PostgREST keeps answering with its cached table metadata."""
    assert MIGRATION.rstrip().endswith("NOTIFY pgrst, 'reload schema';")


# ----------------------------------------------------- the concurrency guard


def _guarded_table() -> sqlite3.Connection:
    """A pipeline_runs stand-in carrying migration 015's ACTUAL index definition."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE pipeline_runs ("
        " id INTEGER PRIMARY KEY, pillar_id TEXT NOT NULL,"
        " kind TEXT NOT NULL, status TEXT NOT NULL)"
    )
    conn.execute(_statement(
        r"CREATE UNIQUE INDEX[^;]*pipeline_runs_one_active_per_pillar[^;]*;"
    ))
    return conn


def _insert(conn, kind, status="running", pillar=PILLAR):
    conn.execute(
        "INSERT INTO pipeline_runs (pillar_id, kind, status) VALUES (?, ?, ?)",
        (pillar, kind, status),
    )


def test_an_upload_is_allowed_while_a_discovery_run_is_active():
    """The captain's ruling, executed against the migration's own index.

    Before 015 the guard was ``WHERE status IN ('pending','running')`` with no kind
    predicate, so this second insert raised and the upload route answered 409 for the
    ~30 seconds a discovery run takes.
    """
    conn = _guarded_table()
    _insert(conn, "discover", status="running")
    _insert(conn, "upload", status="pending")  # must not raise

    assert conn.execute("SELECT count(*) FROM pipeline_runs").fetchone()[0] == 2


def test_two_uploads_on_one_pillar_are_allowed():
    """A user pasting two URLs in a row is not a conflict. Concurrency is bounded by
    the scheduler's thread pool, not by this index."""
    conn = _guarded_table()
    _insert(conn, "upload", status="running")
    _insert(conn, "upload", status="running")

    assert conn.execute("SELECT count(*) FROM pipeline_runs").fetchone()[0] == 2


def test_the_guard_still_holds_for_the_kinds_it_was_written_for():
    """Regression on the exemption itself: widening it to every kind would let a
    second daily run compete with a discovery run over one pillar's queue."""
    conn = _guarded_table()
    _insert(conn, "discover", status="running")
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, "run_daily", status="pending")


def test_a_finished_run_stops_covering_its_pillar():
    conn = _guarded_table()
    _insert(conn, "discover", status="succeeded")
    _insert(conn, "run_daily", status="pending")  # must not raise


def test_dispatch_does_not_refuse_an_upload_behind_a_live_discovery_run():
    """The same rule, through the code path a route actually takes.

    ``db.create_pipeline_run`` is replaced with one backed by the sqlite table above,
    so the uniqueness decision is made by migration 015's index rather than by a mock
    that was told what to answer. It returns None on a conflict exactly as the real
    one does for Postgres 23505, which is what dispatch_run turns into
    RunAlreadyActiveError.
    """
    conn = _guarded_table()
    scheduler = MagicMock()

    def fake_create(pillar_id, trigger_source, kind, stage_names):
        try:
            _insert(conn, kind, status="pending", pillar=pillar_id)
        except sqlite3.IntegrityError:
            return None
        return MagicMock(id=f"run-{conn.total_changes}")

    with patch("webui.services.run_service.db") as mock_db:
        mock_db.create_pipeline_run.side_effect = fake_create

        discovery_id = dispatch_run(
            scheduler, {}, pillar_id=PILLAR,
            trigger_source="ui_discover", kind=KIND_DISCOVER, limit=10,
        )
        upload_id = dispatch_run(
            scheduler, {}, pillar_id=PILLAR,
            trigger_source=TRIGGER_UI_UPLOAD, kind=KIND_UPLOAD,
            source=UPLOAD_SOURCE_URL, url="https://arxiv.org/pdf/1706.03762",
        )

    assert discovery_id and upload_id and discovery_id != upload_id
    assert scheduler.add_job.call_count == 2


# ------------------------------------------------------------------- dispatch


def test_an_upload_seeds_the_upload_stage_list():
    scheduler = MagicMock()
    with patch("webui.services.run_service.db") as mock_db:
        mock_db.create_pipeline_run.return_value = MagicMock(id="run-1")
        dispatch_run(
            scheduler, {}, pillar_id=PILLAR,
            trigger_source=TRIGGER_UI_UPLOAD, kind=KIND_UPLOAD,
            source=UPLOAD_SOURCE_URL, url="https://example.com/p.pdf",
        )

    stage_names = mock_db.create_pipeline_run.call_args.args[3]
    assert stage_names == [s.value for s in UPLOAD_STAGES]
    # The six shared steps keep the names _process_paper already uses, so one label
    # change fixes both pipelines.
    assert stage_names[2:] == [
        "ingest", "summarize", "synthesize", "quiz", "persist", "vectors"
    ]


def test_the_upload_trigger_source_is_swept_on_restart():
    """A webui restart must be able to declare its own abandoned uploads interrupted;
    a run left 'running' forever is indistinguishable from a live one."""
    assert TRIGGER_UI_UPLOAD in run_service.WEBUI_TRIGGER_SOURCES


# -------------------------------------------------------------- execute_run


def _execute_upload(job_result=None, side_effect=None, **kwargs):
    events = {"run-1": threading.Event()}
    payload = {"source": UPLOAD_SOURCE_URL, "url": "https://x/p.pdf"}
    payload.update(kwargs)
    with (
        patch("webui.services.run_service.db") as mock_db,
        patch("webui.services.run_service.Orchestrator") as mock_orch,
        patch("webui.services.run_service.get_upload_service") as get_service,
    ):
        service = get_service.return_value
        if side_effect is not None:
            service.run_url_upload_job.side_effect = side_effect
            service.run_file_upload_job.side_effect = side_effect
        else:
            service.run_url_upload_job.return_value = job_result
            service.run_file_upload_job.return_value = job_result
        try:
            execute_run(
                "run-1", KIND_UPLOAD, PILLAR, events["run-1"], events, **payload
            )
        except BaseException:
            # execute_run records the failure and then re-raises so APScheduler's
            # EVENT_JOB_ERROR listener sees it too. What the run ROW says is what
            # these tests are about.
            pass
    return mock_db, mock_orch, service


def _job_result(errors=None, actions=("text_extraction",)):
    paper = MagicMock(id="arxiv:1706.03762", title="Attention Is All You Need")
    return UploadJobResult(
        paper=paper,
        outcome=PipelineOutcome(
            actions_triggered=list(actions), errors=list(errors or [])
        ),
        source="URL: https://x/p.pdf",
    )


def test_an_upload_run_does_not_build_an_orchestrator():
    """Constructing one dials Qdrant (VectorSearchTool.__init__ ->
    ensure_collections). An unreachable vector store must not stop a paper reaching
    the library — that failure belongs on the VECTORS stage, not on the whole run."""
    _, mock_orch, _ = _execute_upload(job_result=_job_result())
    mock_orch.assert_not_called()


def test_a_clean_upload_is_recorded_as_one_processed_paper():
    mock_db, _, _ = _execute_upload(job_result=_job_result())

    args, kwargs = mock_db.finish_pipeline_run.call_args
    assert args[1] == RunStatus.SUCCEEDED.value
    assert kwargs["papers_processed"] == 1
    assert kwargs["papers_failed"] == 0
    assert kwargs["error"] is None
    assert kwargs["result"]["added"] is True
    assert kwargs["result"]["title"] == "Attention Is All You Need"


def test_a_failed_step_fails_the_run_but_the_payload_still_says_the_paper_was_added():
    """Both facts, kept apart. A green run above a red stage row is the lie; so is
    "Failed" full stop on a run whose paper is sitting in the library."""
    mock_db, _, _ = _execute_upload(
        job_result=_job_result(errors=["summarizer: model refused"])
    )

    args, kwargs = mock_db.finish_pipeline_run.call_args
    assert args[1] == RunStatus.FAILED.value
    assert kwargs["papers_processed"] == 0
    assert kwargs["papers_failed"] == 1
    assert "model refused" in kwargs["error"]
    assert kwargs["result"]["added"] is True
    assert kwargs["result"]["errors"] == ["summarizer: model refused"]


def test_every_failed_step_is_reported_not_just_the_first():
    mock_db, _, _ = _execute_upload(
        job_result=_job_result(errors=["summarizer: a", "vector_storage: b"])
    )
    error = mock_db.finish_pipeline_run.call_args.kwargs["error"]
    assert "summarizer: a" in error and "vector_storage: b" in error


def test_an_upload_that_never_reached_the_library_fails_with_no_result():
    """The loud path. The job raises rather than returning, so there is no result
    payload and nothing can read `added` as true."""
    from nlp_pillars.services.upload_service import UploadError

    mock_db, _, _ = _execute_upload(side_effect=UploadError("404 from arxiv.org"))

    args, kwargs = mock_db.finish_pipeline_run.call_args
    assert args[1] == RunStatus.FAILED.value
    assert "404 from arxiv.org" in kwargs["error"]
    assert "result" not in kwargs


def test_a_cancelled_upload_is_recorded_as_cancelled_not_failed():
    mock_db, _, _ = _execute_upload(
        side_effect=RunCancelledError("cancelled before ingest")
    )
    assert mock_db.finish_pipeline_run.call_args.args[1] == RunStatus.CANCELLED.value


def test_the_file_source_reaches_the_file_job():
    _, _, service = _execute_upload(
        job_result=_job_result(),
        source=UPLOAD_SOURCE_FILE,
        saved_path="/app/data/uploads/abc.pdf",
        filename="paper.pdf",
        title="A Paper",
    )
    service.run_file_upload_job.assert_called_once()
    service.run_url_upload_job.assert_not_called()
    assert service.run_file_upload_job.call_args.args[1] == "/app/data/uploads/abc.pdf"


def test_an_upload_run_with_no_source_fails_loudly():
    mock_db, _, _ = _execute_upload(job_result=_job_result(), source=None)
    args, kwargs = mock_db.finish_pipeline_run.call_args
    assert args[1] == RunStatus.FAILED.value
    assert "source" in kwargs["error"]


# ------------------------------------------------------- stage reporting


class _Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, name, status, detail=None):
        self.events.append((name, status, detail))

    def status_of(self, stage):
        seen = [e for e in self.events if e[0] == stage.value]
        return seen[-1][1] if seen else None


@pytest.fixture
def service(tmp_path):
    from nlp_pillars.services.upload_service import UploadService

    return UploadService(upload_dir=str(tmp_path / "uploads"))


@pytest.fixture
def paper():
    from nlp_pillars.schemas import PaperRef

    return PaperRef(id="arxiv:1706.03762", title="A Paper", authors=[], url_pdf="x")


def _parsed(text="body text"):
    from nlp_pillars.schemas import PaperRef, ParsedPaper

    return ParsedPaper(
        paper_ref=PaperRef(id="arxiv:1706.03762", title="A Paper", authors=[]),
        full_text=text,
        chunks=[text],
    )


def test_a_failed_ingest_marks_every_later_stage_skipped_with_a_reason(
    service, paper
):
    """A row left 'pending' on a finished run is the shape a silent failure takes on
    this page: it reads as work that is still coming."""
    on_stage = _Recorder()
    with patch.object(service.ingest_agent, "ingest",
                      side_effect=RuntimeError("pdf is garbage")):
        outcome = service._run_full_pipeline(
            paper=paper, pillar_id="nlp-fundamentals", on_stage=on_stage,
        )

    assert not outcome.ok
    assert on_stage.status_of(StageName.INGEST) == StageStatus.FAILED.value
    for stage in (StageName.SUMMARIZE, StageName.SYNTHESIZE, StageName.QUIZ,
                  StageName.PERSIST, StageName.VECTORS):
        assert on_stage.status_of(stage) == StageStatus.SKIPPED.value
    reasons = [d for n, s, d in on_stage.events if s == StageStatus.SKIPPED.value]
    assert all(r for r in reasons), "a skipped stage must say why"


def test_turning_the_summarizer_off_skips_three_stages_rather_than_hiding_them(
    service, paper
):
    on_stage = _Recorder()
    with (
        patch.object(service.ingest_agent, "ingest", return_value=_parsed()),
        patch("nlp_pillars.services.upload_service.db"),
        patch("nlp_pillars.services.upload_service.vectors") as mock_vectors,
    ):
        mock_vectors.upsert_text.return_value = 7
        service._run_full_pipeline(
            paper=paper, pillar_id="nlp-fundamentals",
            run_summarizer=False, generate_quiz=False, on_stage=on_stage,
        )

    for stage in (StageName.SUMMARIZE, StageName.SYNTHESIZE, StageName.QUIZ):
        assert on_stage.status_of(stage) == StageStatus.SKIPPED.value
    assert on_stage.status_of(StageName.VECTORS) == StageStatus.COMPLETED.value


def test_zero_vectors_from_a_non_empty_paper_marks_the_stage_failed(service, paper):
    """upsert_text returns 0 for an empty document and a dead Qdrant alike."""
    on_stage = _Recorder()
    with (
        patch.object(service.ingest_agent, "ingest", return_value=_parsed()),
        patch("nlp_pillars.services.upload_service.db"),
        patch("nlp_pillars.services.upload_service.vectors") as mock_vectors,
    ):
        mock_vectors.upsert_text.return_value = 0
        outcome = service._run_full_pipeline(
            paper=paper, pillar_id="nlp-fundamentals",
            run_summarizer=False, generate_quiz=False, on_stage=on_stage,
        )

    assert not outcome.ok
    assert on_stage.status_of(StageName.VECTORS) == StageStatus.FAILED.value


def test_a_stage_callback_that_raises_does_not_lose_the_upload(service, paper):
    """Losing the progress display is bad; losing the paper with it is worse."""
    def exploding(name, status, detail=None):
        raise RuntimeError("the browser is on fire")

    with (
        patch.object(service.ingest_agent, "ingest", return_value=_parsed()),
        patch("nlp_pillars.services.upload_service.db"),
        patch("nlp_pillars.services.upload_service.vectors") as mock_vectors,
    ):
        mock_vectors.upsert_text.return_value = 7
        outcome = service._run_full_pipeline(
            paper=paper, pillar_id="nlp-fundamentals",
            run_summarizer=False, generate_quiz=False, on_stage=exploding,
        )

    assert outcome.ok, outcome.errors


def test_a_download_failure_closes_out_every_later_stage(service, paper):
    """A hard failure stops the upload, and the seven stages it will never reach must
    not sit on a finished run looking `pending` — that reads as work still coming."""
    from nlp_pillars.schemas import UploadUrlRequest
    from nlp_pillars.services.upload_service import UploadError

    on_stage = _Recorder()
    with patch("nlp_pillars.services.upload_service.download_pdf",
               side_effect=RuntimeError("404 Not Found")):
        with pytest.raises(UploadError):
            service.run_url_upload_job(
                "nlp-fundamentals",
                UploadUrlRequest(url="https://arxiv.org/pdf/9999.99999"),
                on_stage=on_stage,
            )

    assert on_stage.status_of(StageName.UPLOAD_FETCH) == StageStatus.FAILED.value
    for stage in UPLOAD_STAGES[1:]:
        assert on_stage.status_of(stage) == StageStatus.SKIPPED.value, stage


def test_closing_out_never_overwrites_a_stage_that_really_ran(service, paper):
    """Regression on the close-out itself. The upload job and the post-upload pipeline
    share one reporter precisely so a late failure cannot re-mark UPLOAD_FETCH — which
    genuinely completed — as skipped."""
    on_stage = _Recorder()
    with (
        patch.object(service.ingest_agent, "ingest", return_value=_parsed()),
        patch("nlp_pillars.services.upload_service.db"),
        patch("nlp_pillars.services.upload_service.vectors") as mock_vectors,
    ):
        mock_vectors.upsert_text.return_value = 7
        stages = upload_service_module._Stages(on_stage)
        stages.done(StageName.UPLOAD_FETCH, "1.2 MB")
        stages.done(StageName.UPLOAD_METADATA, "added")
        service._run_full_pipeline(
            paper=paper, pillar_id="nlp-fundamentals",
            run_summarizer=False, generate_quiz=False, stages=stages,
        )
        stages.stop_after_current("the upload was cancelled")

    assert on_stage.status_of(StageName.UPLOAD_FETCH) == StageStatus.COMPLETED.value
    assert on_stage.status_of(StageName.INGEST) == StageStatus.COMPLETED.value
    assert on_stage.status_of(StageName.VECTORS) == StageStatus.COMPLETED.value


def test_a_cancelled_upload_closes_out_the_stages_it_never_reached(service):
    """`start` raises before the stage is marked running, so without the close-out a
    cancelled run ends above a column of pending rows."""
    from nlp_pillars.schemas import UploadUrlRequest

    on_stage = _Recorder()
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(RunCancelledError):
        service.run_url_upload_job(
            "nlp-fundamentals",
            UploadUrlRequest(url="https://arxiv.org/pdf/1706.03762"),
            on_stage=on_stage,
            cancel=cancel,
        )

    for stage in UPLOAD_STAGES:
        assert on_stage.status_of(stage) == StageStatus.SKIPPED.value, stage


def test_a_tenacity_wrapper_never_reaches_the_user_as_a_future_repr():
    """Measured on a real 404: the panel read ``RetryError[<Future at 0x7f29…
    state=finished raised HTTPStatusError>]``, which names no status code.
    pdf_loader interpolates that repr into its OWN message, so _first_line's trim
    cannot see it — the reason is one hop down the exception chain."""
    from nlp_pillars.services.upload_service import _reason

    class _Retry(Exception):
        pass

    inner = RuntimeError("Client error '404 Not Found' for url 'https://x/y.pdf'")
    try:
        try:
            raise inner
        except RuntimeError as e:
            raise _Retry("RetryError[<Future at 0x7f29 state=finished raised X>]") from e
    except _Retry as retry:
        try:
            raise RuntimeError(f"Failed to download PDF from https://x/y.pdf: {retry}")
        except RuntimeError as outer:
            text = _reason(outer)

    assert "Future" not in text
    assert "404 Not Found" in text


def test_an_ordinary_message_is_left_alone():
    from nlp_pillars.services.upload_service import _reason

    assert _reason(RuntimeError("the model refused")) == "the model refused"


def test_cancellation_is_checked_at_a_stage_boundary(service, paper):
    cancel = threading.Event()
    cancel.set()

    with patch.object(service.ingest_agent, "ingest") as ingest:
        with pytest.raises(RunCancelledError):
            service._run_full_pipeline(
                paper=paper, pillar_id="nlp-fundamentals", cancel=cancel,
            )
    # Never mid-stage: the work is not begun, so nothing is left half-written.
    ingest.assert_not_called()
