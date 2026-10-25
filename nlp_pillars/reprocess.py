"""Regenerate note, lesson and quiz cards for papers already in the library.

Reads each paper's stored text (``paper_texts``) and runs the orchestrator's
own summarize / synthesize / quiz / persist steps on it. It does not download,
ingest, chunk, embed or project anything, and it writes to ``notes``,
``lessons``, ``quiz_cards`` and the concept tables only: never to ``papers``,
the queue, pillars, podcasts or fact-checks.

Papers that already have a note are never regenerated here. The inserts only
append, so a second note, lesson and set of cards would sit beside the first
(and a quiz card carries review history). They are reported as ``has_note``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from . import db
from .llm_client import PAPER_TEXT_TOKEN_BUDGET, StructuredCall
from .llm_prices import price_tokens
from .llm_routes import QUIZ_ROUTE, SUMMARIZE_ROUTE, SYNTHESIZE_ROUTE
from .orchestrator import Orchestrator, _first_line
from .schemas import (
    PaperRef,
    ParsedPaper,
    RunStatus,
    StageName,
    StageStatus,
)

logger = logging.getLogger(__name__)

#: Claude tokens per tiktoken cl100k token on paper text. Measured 2026-10-25:
#: 60,000 tiktoken tokens of a paper was 89,261 Claude tokens.
CLAUDE_TOKENS_PER_TIKTOKEN = 1.5
#: System prompt + instructions + the note that lesson and quiz also carry.
PROMPT_OVERHEAD_TOKENS = 2_000
#: Typical output of the three calls, from stage details of earlier runs.
OUTPUT_TOKENS = {"summarize": 1_800, "synthesize": 2_500, "quiz": 1_800}

REPROCESS_STAGES = [
    StageName.SUMMARIZE,
    StageName.SYNTHESIZE,
    StageName.QUIZ,
    StageName.PERSIST,
]

# The run row needs a kind and a trigger the existing CHECK constraints admit;
# adding a value would be a hand-applied migration for no behavioural gain.
RUN_KIND = "process_selected"
RUN_TRIGGER = "scheduler"


@dataclass
class Candidate:
    paper: PaperRef
    text_tokens: int = 0
    estimate_usd: float = 0.0
    #: "ready", "has_note" or "no_text"
    state: str = "ready"
    detail: str = ""


@dataclass
class PaperOutcome:
    paper_id: str
    status: str  # "done" | "failed"
    cost_usd: float = 0.0
    detail: str = ""
    run_id: Optional[str] = None
    calls: List[str] = field(default_factory=list)
    cards: int = 0


@dataclass
class ReprocessReport:
    dry_run: bool
    candidates: List[Candidate] = field(default_factory=list)
    outcomes: List[PaperOutcome] = field(default_factory=list)

    @property
    def ready(self) -> List[Candidate]:
        return [c for c in self.candidates if c.state == "ready"]

    @property
    def estimate_usd(self) -> float:
        return round(sum(c.estimate_usd for c in self.ready), 6)

    @property
    def cost_usd(self) -> float:
        return round(sum(o.cost_usd for o in self.outcomes), 6)

    @property
    def failed(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "failed")

    def render(self) -> str:
        lines = [f"reprocess ({'dry-run' if self.dry_run else 'write'})"]
        for c in self.candidates:
            if c.state == "ready":
                if self.dry_run:
                    lines.append(
                        f"would process {c.paper.id} {c.text_tokens:,} text tokens "
                        f"est ${c.estimate_usd:.4f}  {c.paper.title[:60]}"
                    )
            else:
                lines.append(f"{c.state} {c.paper.id}: {c.detail}")
        if self.dry_run:
            lines.append(f"papers to process {len(self.ready)}")
            lines.append(f"estimated cost_usd total {self.estimate_usd:.4f}")
            lines.append("No model call. Nothing written.")
            return "\n".join(lines)
        for o in self.outcomes:
            lines.append(
                f"{o.status} {o.paper_id} cost_usd {o.cost_usd:.6f} "
                f"cards {o.cards} run {o.run_id or '-'}: {o.detail}"
            )
        lines.append(
            f"processed {len(self.outcomes) - self.failed} "
            f"failed {self.failed} actual cost_usd total {self.cost_usd:.6f}"
        )
        return "\n".join(lines)


def _count_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:  # the chunker's own degradation: ~4 characters a token
        return len(text) // 4


def estimate_paper_usd(text_tokens: int) -> float:
    """Cost of the three calls for a paper of ``text_tokens`` tiktoken tokens."""
    body = min(text_tokens, PAPER_TEXT_TOKEN_BUDGET)
    claude_in = int(body * CLAUDE_TOKENS_PER_TIKTOKEN) + PROMPT_OVERHEAD_TOKENS
    total = 0.0
    for route, name in (
        (SUMMARIZE_ROUTE, "summarize"),
        (SYNTHESIZE_ROUTE, "synthesize"),
        (QUIZ_ROUTE, "quiz"),
    ):
        total += price_tokens(route.resolved_model(), claude_in, OUTPUT_TOKENS[name])
    return total


def select_candidates(
    paper_id: Optional[str] = None, missing_only: bool = False
) -> List[Candidate]:
    """Library papers to regenerate, each with its state.

    Raises:
        db.PaperLookupError, db.PaperTextLookupError: a table could not be read.
    """
    papers = db.list_all_papers()
    if paper_id is not None:
        papers = [p for p in papers if p.id == paper_id]
        if not papers:
            raise ValueError(f"paper {paper_id} is not in the library")
    noted = set(db.list_noted_paper_ids())
    out: List[Candidate] = []
    for paper in papers:
        if paper.id in noted:
            if not missing_only:
                out.append(Candidate(
                    paper, state="has_note",
                    detail="already has a note; regenerating would duplicate "
                           "its lesson and quiz cards",
                ))
            continue
        if not paper.pillar_id:
            out.append(Candidate(paper, state="no_pillar", detail="paper has no pillar"))
            continue
        text = db.get_stored_paper_text(paper.id)
        if not text:
            out.append(Candidate(
                paper, state="no_text",
                detail="no stored paper text (paper_texts); run "
                       "`text backfill` first",
            ))
            continue
        tokens = _count_tokens(text)
        out.append(Candidate(
            paper, text_tokens=tokens, estimate_usd=estimate_paper_usd(tokens),
        ))
    return out


def _new_calls(agent, before) -> Optional[StructuredCall]:
    call = agent.last_call
    return call if isinstance(call, StructuredCall) and call is not before else None


def run_reprocess(
    *,
    paper_id: Optional[str] = None,
    missing_only: bool = False,
    dry_run: bool = False,
    orchestrator_factory: Callable[..., Orchestrator] = Orchestrator,
    agents: Optional[tuple] = None,
) -> ReprocessReport:
    """Select papers, then (unless ``dry_run``) regenerate them one at a time."""
    report = ReprocessReport(dry_run=dry_run)
    report.candidates = select_candidates(paper_id, missing_only)
    if dry_run or not report.ready:
        return report

    if agents is None:
        from .agents.quiz_agent import QuizAgent
        from .agents.summarizer_agent import SummarizerAgent
        from .agents.synthesis_agent import SynthesisAgent

        agents = (SummarizerAgent, SynthesisAgent, QuizAgent)
    watched = agents

    current_run: dict = {"id": None}

    def on_stage(name: str, status: str, detail: Optional[str] = None) -> None:
        if current_run["id"] and status != StageStatus.DROPPED.value:
            db.update_pipeline_run_stage(current_run["id"], name, status, detail)

    orchestrator = orchestrator_factory(enable_quiz=True, on_stage=on_stage)

    for cand in report.ready:
        report.outcomes.append(
            _reprocess_one(orchestrator, cand.paper, current_run, on_stage, watched)
        )
    return report


def _reprocess_one(orchestrator, paper: PaperRef, current_run: dict, on_stage, watched) -> PaperOutcome:
    pillar_id = paper.pillar_id
    outcome = PaperOutcome(paper.id, "failed")
    run_id = None
    try:
        run = db.create_pipeline_run(
            pillar_id, RUN_TRIGGER, RUN_KIND, [s.value for s in REPROCESS_STAGES],
        )
        run_id = run.id if run else None
        if run is None:
            outcome.detail = "no run row (another run is active on this pillar); "
    except Exception as exc:
        outcome.detail = f"no run row ({_first_line(exc)}); "
    outcome.run_id = run_id
    current_run["id"] = run_id
    if run_id:
        db.start_pipeline_run(run_id)

    befores = [a.last_call for a in watched]
    problem: Optional[str] = None
    try:
        text = db.get_stored_paper_text(paper.id)
        if not text:
            raise ValueError("stored paper text disappeared")
        parsed = ParsedPaper(paper_ref=paper, full_text=text, chunks=[])
        note, lesson, cards = orchestrator._generate_learning_content(
            pillar_id, paper, parsed
        )
        on_stage(StageName.PERSIST.value, StageStatus.RUNNING.value, paper.id)
        note_ok, lesson_ok, cards_added = orchestrator._store_learning_content(
            note, lesson, cards
        )
        wanted = len(cards or [])
        missing = [
            what for what, ok in (
                ("note", note_ok), ("lesson", lesson_ok),
                (f"quiz cards ({cards_added}/{wanted})", cards_added == wanted),
            ) if not ok
        ]
        if missing:
            raise RuntimeError(f"database refused: {', '.join(missing)}")
        outcome.cards = cards_added
        on_stage(
            StageName.PERSIST.value, StageStatus.COMPLETED.value,
            f"{paper.id}: note, lesson, {cards_added} cards stored",
        )
        outcome.status = "done"
        outcome.detail += f"{note.concepts and len(note.concepts) or 0} concepts"
    except Exception as exc:
        problem = _first_line(exc)
        logger.error("reprocess %s failed: %s", paper.id, exc, exc_info=True)
        orchestrator._mark_current_paper_stage_failed(problem)
        outcome.detail += problem
    finally:
        for agent, before in zip(watched, befores):
            call = _new_calls(agent, before)
            if call is not None:
                outcome.cost_usd += call.cost_usd
                outcome.calls.append(call.detail())
        outcome.cost_usd = round(outcome.cost_usd, 6)
        if run_id:
            status = RunStatus.SUCCEEDED if outcome.status == "done" else RunStatus.FAILED
            db.finish_pipeline_run(
                run_id, status.value,
                papers_processed=1 if outcome.status == "done" else 0,
                papers_failed=0 if outcome.status == "done" else 1,
                error=problem,
            )
        current_run["id"] = None
    return outcome
