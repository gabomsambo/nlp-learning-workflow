"""Async PostgREST client for the Web UI.

Provides typed helper methods to read data from the existing database via
PostgREST running at http://localhost:3000. This client is read-only and
optimized for the UI use cases (counts, recent activity, simple lists).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel

from nlp_pillars.pillar_utils import get_all_pillar_ids


class QuickStats(BaseModel):
    """Aggregated counts for dashboard quick stats."""

    total_papers: int
    total_lessons: int
    total_quiz_cards: int


class ActivityItem(BaseModel):
    """Recent activity feed item."""

    type: str
    id: str
    title: str
    created_at: Optional[str] = None
    pillar_id: Optional[str] = None


@dataclass
class _TableCount:
    name: str
    created_col: str


class PostgrestClient:
    """Minimal async client tailored for dashboard and browsing needs.

    Env vars:
      - POSTGREST_URL (default: http://localhost:3000)
      - SUPABASE_KEY (optional, used as Bearer token)
    """

    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None, client: Optional[httpx.AsyncClient] = None):
        url = base_url or os.getenv("POSTGREST_URL") or os.getenv("SUPABASE_URL") or "http://localhost:3000"
        url = url.rstrip("/")
        # Supabase cloud uses /rest/v1/ path, local PostgREST uses root
        if "supabase.co" in url and "/rest/v1" not in url:
            url = f"{url}/rest/v1"
        self.base_url = url
        auth_token = token or os.getenv("SUPABASE_KEY")

        headers = {"Accept": "application/json", "Prefer": "count=exact"}
        if auth_token:
            headers["apikey"] = auth_token
            headers["Authorization"] = f"Bearer {auth_token}"

        self._client = client or httpx.AsyncClient(headers=headers, timeout=20.0)

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], httpx.Response]:
        """Perform GET and return JSON with response for header access."""
        resp = await self._client.get(f"{self.base_url}/{path.lstrip('/')}", params=params or {})
        resp.raise_for_status()
        data = resp.json() if resp.text else []
        return data, resp

    @staticmethod
    def _parse_count_from_content_range(resp: httpx.Response) -> int:
        """Extract total count from Content-Range header when available."""
        cr = resp.headers.get("Content-Range")
        if not cr:
            return 0
        # Example: items 0-1/123
        try:
            total = cr.split("/")[-1]
            return int(total)
        except Exception:
            return 0

    async def get_quick_stats(self) -> QuickStats:
        """Return counts for papers, lessons, and quiz_cards.

        Uses Prefer: count=exact to read totals from Content-Range.
        """

        async def count_table(table: str) -> int:
            # Degrade to 0 when the database is unreachable, mirroring how
            # get_recent_activity() tolerates per-table failures.
            try:
                _, resp = await self._get(table, params={"select": "id", "limit": 1})
            except Exception:
                return 0
            return self._parse_count_from_content_range(resp)

        papers = await count_table("papers")
        lessons = await count_table("lessons")
        quiz = await count_table("quiz_cards")
        return QuickStats(total_papers=papers, total_lessons=lessons, total_quiz_cards=quiz)

    async def get_recent_activity(self, limit: int = 10) -> List[ActivityItem]:
        """Compile recent items from papers, lessons, and quiz_cards."""
        params_papers = {"select": "id,title,pillar_id,added_at", "order": "added_at.desc", "limit": limit}
        params_lessons = {"select": "id,paper_id,tl_dr,pillar_id,created_at", "order": "created_at.desc", "limit": limit}
        params_quiz = {"select": "id,paper_id,pillar_id,last_reviewed,created_at", "order": "created_at.desc", "limit": limit}

        results = await asyncio.gather(
            self._get("papers", params=params_papers),
            self._get("lessons", params=params_lessons),
            self._get("quiz_cards", params=params_quiz),
            return_exceptions=True,
        )

        papers: List[Dict[str, Any]] = []
        lessons: List[Dict[str, Any]] = []
        quiz_cards: List[Dict[str, Any]] = []

        try:
            if not isinstance(results[0], Exception):
                papers = results[0][0] or []
        except Exception:
            papers = []
        try:
            if not isinstance(results[1], Exception):
                lessons = results[1][0] or []
        except Exception:
            lessons = []
        try:
            if not isinstance(results[2], Exception):
                quiz_cards = results[2][0] or []
        except Exception:
            quiz_cards = []

        items: List[ActivityItem] = []
        for p in papers:
            items.append(ActivityItem(
                type="paper",
                id=str(p.get("id")),
                title=p.get("title", ""),
                created_at=p.get("added_at"),
                pillar_id=p.get("pillar_id"),
            ))
        for l in lessons:
            items.append(
                ActivityItem(type="lesson", id=str(l.get("id")), title=l.get("tl_dr", "Lesson"), created_at=l.get("created_at"), pillar_id=l.get("pillar_id"))
            )
        for q in quiz_cards:
            items.append(
                ActivityItem(type="quiz", id=str(q.get("id")), title=q.get("paper_id", "Quiz"), created_at=q.get("created_at") or q.get("last_reviewed"), pillar_id=q.get("pillar_id"))
            )

        # Sort by created_at desc, None at end
        items.sort(key=lambda x: (x.created_at is not None, x.created_at), reverse=True)
        return items[:limit]

    async def list_papers(self, pillar: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List papers with basic fields."""
        params: Dict[str, Any] = {"select": "id,title,authors,year,citation_count,pillar_id,added_at", "order": "added_at.desc", "limit": limit}
        if pillar:
            params["pillar_id"] = f"eq.{pillar}"
        # Degrade to an empty list when the database is unreachable.
        try:
            data, _ = await self._get("papers", params=params)
        except Exception:
            return []
        return data

    async def list_lessons(self, pillar: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List lessons with display fields."""
        params: Dict[str, Any] = {"select": "id,paper_id,tl_dr,pillar_id,created_at,difficulty,estimated_time,takeaways", "order": "created_at.desc", "limit": limit}
        if pillar:
            params["pillar_id"] = f"eq.{pillar}"
        # Degrade to an empty list when the database is unreachable.
        try:
            data, _ = await self._get("lessons", params=params)
        except Exception:
            return []
        return data

    async def list_podcast_scripts(self, pillar: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List podcast scripts with display fields."""
        params: Dict[str, Any] = {
            "select": "id,paper_id,pillar_id,title,word_count,created_at",
            "order": "created_at.desc",
            "limit": limit
        }
        if pillar:
            params["pillar_id"] = f"eq.{pillar}"
        data, _ = await self._get("podcast_scripts", params=params)
        return data

    async def get_podcast_script_details(self, script_id: str) -> Optional[Dict[str, Any]]:
        """Get full podcast script details by ID."""
        try:
            params = {"select": "*", "id": f"eq.{script_id}"}
            data, _ = await self._get("podcast_scripts", params=params)

            if not data:
                return None

            return data[0]

        except Exception as e:
            print(f"Error fetching podcast script details: {e}")
            return None

    async def list_quiz_cards(self, pillar: Optional[str] = None, difficulty: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List quiz cards with basic fields."""
        params: Dict[str, Any] = {"select": "id,paper_id,question,answer,difficulty,pillar_id,created_at", "order": "created_at.desc", "limit": limit}
        if pillar:
            params["pillar_id"] = f"eq.{pillar}"
        if difficulty is not None:
            params["difficulty"] = f"eq.{difficulty}"
        # Degrade to an empty list when the database is unreachable.
        try:
            data, _ = await self._get("quiz_cards", params=params)
        except Exception:
            return []
        return data

    async def counts_by_pillar(self) -> Dict[str, Dict[str, int]]:
        """Return counts per pillar for papers, lessons, quiz_cards using Content-Range."""
        # Get all pillar IDs dynamically from database
        pillar_ids = get_all_pillar_ids()
        result: Dict[str, Dict[str, int]] = {p: {"papers": 0, "lessons": 0, "quizzes": 0} for p in pillar_ids}

        async def table_count_for_pillar(table: str, pillar: str) -> int:
            _, resp = await self._get(table, params={"select": "id", "pillar_id": f"eq.{pillar}", "limit": 1})
            return self._parse_count_from_content_range(resp)

        # Run in batches per table for responsiveness
        for table, key in [("papers", "papers"), ("lessons", "lessons"), ("quiz_cards", "quizzes")]:
            tasks = [table_count_for_pillar(table, p) for p in pillar_ids]
            counts = await asyncio.gather(*tasks, return_exceptions=True)
            for pillar, c in zip(pillar_ids, counts):
                result[pillar][key] = 0 if isinstance(c, Exception) else int(c)

        return result

    async def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive paper details including notes, lessons, and quiz cards."""
        try:
            # Fetch paper metadata
            paper_params = {"select": "*", "id": f"eq.{paper_id}"}
            paper_data, _ = await self._get("papers", params=paper_params)

            if not paper_data:
                return None

            paper = paper_data[0]

            # Fetch related data in parallel
            notes_params = {"select": "*", "paper_id": f"eq.{paper_id}", "order": "created_at.desc"}
            lessons_params = {"select": "*", "paper_id": f"eq.{paper_id}", "order": "created_at.desc"}
            quiz_params = {"select": "*", "paper_id": f"eq.{paper_id}", "order": "created_at.desc"}

            results = await asyncio.gather(
                self._get("notes", params=notes_params),
                self._get("lessons", params=lessons_params),
                self._get("quiz_cards", params=quiz_params),
                return_exceptions=True,
            )

            notes = results[0][0] if not isinstance(results[0], Exception) else []
            lessons = results[1][0] if not isinstance(results[1], Exception) else []
            quiz_cards = results[2][0] if not isinstance(results[2], Exception) else []

            return {
                "paper": paper,
                "notes": notes,
                "lessons": lessons,
                "quiz_cards": quiz_cards
            }

        except Exception as e:
            print(f"Error fetching paper details: {e}")
            return None



    # ------------------------------------------------------------------
    # Pipeline run polling (migration 009)
    #
    # These are the READ side of run tracking. They live on the async client
    # because the browser polls them roughly once a second and a poll must never
    # occupy a worker thread. The WRITE side is synchronous and lives in
    # nlp_pillars/db.py, called from the pipeline's own thread. Do not cross them.
    # ------------------------------------------------------------------

    @staticmethod
    def _shape_run(row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise a run row: rename the embed and sort stages by seq.

        PostgREST returns the embedded child rows under the table name and in no
        guaranteed order, so the ordering is done here rather than in every caller
        and in the browser.
        """
        stages = row.pop("pipeline_run_stages", None) or []
        row["stages"] = sorted(stages, key=lambda s: s.get("seq", 0))
        return row

    async def get_pipeline_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get one run with its stages, or None if it does not exist.

        Uses PostgREST resource embedding so the run and all eleven stages arrive in
        a single request — at one poll per second, a second round trip per poll is
        not free.
        """
        try:
            params = {"select": "*,pipeline_run_stages(*)", "id": f"eq.{run_id}"}
            data, _ = await self._get("pipeline_runs", params=params)
        except Exception:
            return None
        if not data:
            return None
        return self._shape_run(data[0])

    async def get_active_pipeline_run(
        self, pillar_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get the in-flight run, optionally for one pillar.

        Powers "reattach after a refresh" when the browser has no stored run id.
        Filters on status rather than on a null finished_at, deliberately — see the
        eq(None) note in nlp_pillars/db.py.
        """
        try:
            params = {
                "select": "*,pipeline_run_stages(*)",
                "status": "in.(pending,running)",
                "order": "created_at.desc",
                "limit": 1,
            }
            if pillar_id:
                params["pillar_id"] = f"eq.{pillar_id}"
            data, _ = await self._get("pipeline_runs", params=params)
        except Exception:
            return None
        if not data:
            return None
        return self._shape_run(data[0])

    async def list_pipeline_runs(
        self, pillar_id: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Recent runs, newest first. Stages are not embedded — this is a summary."""
        params: Dict[str, Any] = {
            "select": "*",
            "order": "created_at.desc",
            "limit": limit,
        }
        if pillar_id:
            params["pillar_id"] = f"eq.{pillar_id}"
        try:
            data, _ = await self._get("pipeline_runs", params=params)
        except Exception:
            return []
        return data
