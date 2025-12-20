from typing import Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .routers import dashboard, papers, lessons, quiz, pipeline, analytics, pillars, discovery, podcast
from .routers.api import pillars as api_pillars, uploads as api_uploads, quiz as api_quiz, discovery as api_discovery, podcast as api_podcast


def create_app() -> FastAPI:
    app = FastAPI(title="NLP Workflow Web UI", version="0.1.0")

    app.mount("/static", StaticFiles(directory="webui/static"), name="static")
    templates = Jinja2Templates(directory="webui/templates")
    app.state.templates = templates

    # Include API routers
    app.include_router(api_pillars.router)
    app.include_router(api_uploads.router)
    app.include_router(api_quiz.router)
    app.include_router(api_discovery.router)
    app.include_router(api_podcast.router)

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


