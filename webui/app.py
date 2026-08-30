import logging
from contextlib import asynccontextmanager
from datetime import timezone
from typing import Any

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Must run before the local imports below: they pull in nlp_pillars.config, and
# db.get_client() reads os.environ directly rather than going through Settings.
load_dotenv()

from .routers import dashboard, papers, lessons, quiz, pipeline, analytics, pillars, discovery, podcast
from .routers.api import pillars as api_pillars, uploads as api_uploads, quiz as api_quiz, discovery as api_discovery, podcast as api_podcast, pipeline_runs as api_pipeline_runs, papers as api_papers
from .services import run_service

logger = logging.getLogger(__name__)

#: Long pipeline runs get their own small pool. APScheduler's default is 10 threads,
#: which here would mean ten Orchestrators hammering the LLM APIs at once; two is
#: enough for a single-user tool and keeps memory sane given each run holds a parsed
#: PDF.
_MAX_CONCURRENT_RUNS = 2


def create_app() -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Own the background scheduler for the life of the application.

        This is the first thing in webui to start a scheduler — the existing
        BackgroundJobsService runs only in the separate `scheduler` container. The two
        are independent on purpose: a webui restart must not disturb the nightly run,
        and vice versa.
        """
        scheduler = BackgroundScheduler(
            timezone=timezone.utc,
            executors={"default": ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_RUNS)},
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                # Belt and braces with the per-job value in run_service.dispatch_run:
                # APScheduler's built-in default is 1 second, and a job that reaches
                # the executor later than that is silently discarded.
                "misfire_grace_time": None,
            },
        )
        # Second line of defence. execute_run records its own failure, but a job that
        # never ran at all cannot.
        scheduler.add_listener(run_service.on_job_error, EVENT_JOB_ERROR | EVENT_JOB_MISSED)
        scheduler.start()

        app.state.scheduler = scheduler
        #: run_id -> threading.Event. The only way to stop a run: nothing can
        #: interrupt synchronous Python, so cancellation is checked at stage
        #: boundaries inside the Orchestrator.
        app.state.cancel_events = {}

        swept = run_service.sweep_interrupted_runs_on_startup()
        if swept:
            logger.warning(f"Marked {swept} abandoned pipeline run(s) as interrupted")

        try:
            yield
        finally:
            # Ask in-flight runs to stop at their next stage boundary, then wait.
            # shutdown(wait=True) does NOT interrupt a running job — it cannot — so
            # without setting the events first this would block until the current
            # stage finished naturally.
            for event in app.state.cancel_events.values():
                event.set()
            scheduler.shutdown(wait=True)

    app = FastAPI(title="NLP Workflow Web UI", version="0.1.0", lifespan=lifespan)

    app.mount("/static", StaticFiles(directory="webui/static"), name="static")
    templates = Jinja2Templates(directory="webui/templates")
    app.state.templates = templates

    # Include API routers
    # NOTE: webui/routers/api/script_download.py is deliberately NOT registered —
    # importing it aborts startup (FastAPI rejects its Optional[BackgroundTasks]
    # annotation at decoration time). See that module's docstring.
    app.include_router(api_pillars.router)
    app.include_router(api_uploads.router)
    app.include_router(api_quiz.router)
    app.include_router(api_discovery.router)
    app.include_router(api_podcast.router)
    app.include_router(api_pipeline_runs.router)
    app.include_router(api_papers.router)

    # Include page routers
    app.include_router(dashboard.router)
    app.include_router(papers.router)
    app.include_router(lessons.router)
    app.include_router(quiz.router)
    app.include_router(pipeline.router)
    app.include_router(analytics.router)
    app.include_router(pillars.router)
    app.include_router(discovery.router)
    app.include_router(podcast.router)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    return app


app = create_app()
