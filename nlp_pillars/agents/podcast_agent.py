"""
Podcast Agent for generating educational podcast scripts from papers.
Uses a 4-prompt Ground Pack system with Claude Sonnet 4.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import NamedTuple, Optional

import httpx
from anthropic import AsyncAnthropic

from ..config import get_settings
from ..schemas import PaperNote, PaperRef, PodcastScript, SourceMaterial
from ..db import get_paper_by_id, get_notes_by_paper_id
from .ingest_agent import IngestAgent, IngestError

logger = logging.getLogger(__name__)

# Full text is passed to five separate Claude calls (the four Ground Pack
# prompts plus the final synthesis prompt), so this bound is multiplied by five
# on the bill. Measured end to end on an 18K-char paper: 37.8K input + 9.6K
# output tokens = $0.26 at Sonnet list price. A paper at this ceiling would be
# ~128K input tokens, so ~$0.53. Output is roughly constant; only the input
# side scales with the paper. _call_claude logs real per-call token usage.
MAX_FULL_TEXT_CHARS = 100000


class InsufficientSourceMaterialError(Exception):
    """There is nothing to write a podcast from, so nothing was spent trying.

    Raised by ``generate()`` *before* the first model call when the paper body
    could not be read AND the paper has no abstract AND there is no notes row.
    Previously this combination produced five model calls (~$0.27) and a fluent,
    confident script whose entire factual basis was the title, one author name
    and the placeholder "[Full text not available...]" — with a green success
    message and nothing in the response, the page or the stored row recording
    that the body was absent.

    The message carries the reason the body is missing (a dead ``file://`` path,
    a 404, a parse failure), because the caller has to put something actionable
    in front of the person who clicked the button.
    """


class FullTextResult(NamedTuple):
    """What ``_get_full_text`` found, and why it found nothing.

    ``_get_full_text`` used to return a bare ``""`` for every failure, which is
    exactly as informative as an empty paper — the reason died at the ``except``
    and the caller could not tell "this PDF 404s" from "this paper has no PDF".
    Keeping the reason is the whole point; do not collapse this back to ``str``.
    """

    text: str
    error: Optional[str] = None


# Ground Pack Prompts
PROMPT_1_FACTS_OUTLINE = """Role: Academic Research Assistant. Task: Analyze the provided NLP paper and generate a strict, chronological "Facts-Only Outline." Audience: A Computer Science & Linguistics graduate student. Do not simplify technical terms (e.g., keep terms like "ablation study," "perplexity," "embeddings," "zero-shot"). Instructions:

Hook/Problem: What specific gap in the literature is this paper addressing?

Methodology: briefly outline the model architecture, data processing pipeline, or algorithm used. Use the exact names given by the authors.

Experiments: List the specific experiments run.

Results: Summarize the outcome of each experiment qualitatively (e.g., "Model A outperformed Model B on the GLUE benchmark").

Conclusion: What is the primary takeaway? Output Format: Bullet points only. No prose."""

PROMPT_2_CORE_CONCEPTS = """Role: Technical Educator. Task: Identify the 3 most complex or novel concepts introduced in this paper (e.g., a specific new loss function, a novel attention mechanism, or a new sampling method). Audience: A CS/Linguistics graduate. Assume knowledge of Transformers, standard metrics (BLEU, ROUGE, F1), and basic ML concepts. Instructions: For each of the 3 concepts:

Concept Name: Exact term used in the paper.

Technical Definition: A rigorous 1-sentence definition using domain-specific vocabulary.

The "Podcast" Analogy: A conceptual metaphor to explain how it works effectively in an audio format. (e.g., "If standard attention is looking at the whole sentence, this sparse attention mechanism is like highlighting only the verbs..."). Output Format:

Concept 1: [Name]

Definition: ...

Analogy: ... (Repeat for 2 & 3)"""

PROMPT_3_METRICS_DATASETS = """Role: Data Analyst. Task: Extract all quantitative data, dataset names, and benchmarks from the paper. Instructions:

Datasets: List every dataset mentioned (e.g., SQuAD 2.0, Common Crawl).

Baselines: List the models compared against (e.g., BERT-Large, GPT-3).

Key Metrics: Extract the specific SOTA results. (e.g., "Achieved 89.4% accuracy on ImageNet").

Hyperparameters: List key constraints (e.g., parameter count, training tokens, GPU hours) if relevant to the main claim. Output Format: Markdown Table."""

PROMPT_4_LIMITATIONS = """Role: Peer Reviewer. Task: Critically analyze the "Discussion," "Limitations," and "Conclusion" sections. Instructions: Identify 3-4 critical weaknesses or open questions:

Scope Constraints: Where does the model fail? (e.g., long-context degradation, specific language families).

Resource Load: Is it computationally expensive?

Ambiguities: Did the authors mention any unexplained phenomena?

Future Work: What did the authors suggest needs to happen next? Output Format: Bullet points."""

FINAL_SYNTHESIS_PROMPT = """ROLE
Act as an expert science communicator and podcast host. Transform the provided research paper into ~30 minutes of dialogue.

AUDIENCE
A recent undergraduate with strong CS + Linguistics background, preparing for a PhD; technically literate but not yet a domain expert.

TONE
Engaging, enthusiastic, conversational—like a well-produced educational podcast (e.g., TWIML/Neutral/Lex vibe).

GROUNDING RULES (STRICT)
- Use ONLY information found in the provided paper and the Ground Pack.
- If the paper is silent on something, write: [VERIFY]: detail not specified in the paper.
- Include numbers/metrics ONLY if present in the paper; otherwise keep it qualitative.
- Prefer the exact phrasing of named datasets, tasks, and metrics from the paper.
- No external facts, no web knowledge.

OUTPUT FORMAT (STRICT)
- Use ONLY the speaker label [HOST] for all dialogue lines.
- Every line MUST begin with [HOST]: followed by a space. Examples:
  [HOST]: Hello, and welcome to today's episode.
  [HOST]: Let's dive into what makes this paper special.
- No headings, bullets, or prose outside of speaker lines.
- Optional cues are allowed as standalone lines in ALL CAPS inside square brackets: [MUSIC], [SFX], [PAUSE], [TRANSITION].
- Do NOT include parentheses stage directions.

LENGTH & FLOW
Aim for ~30 minutes spoken (approx 3,600-4,200 words at ~120-140 wpm). Keep lines natural and mostly 1-2 sentences.

STRUCTURE (DELIVER VIA SPEAKER LINES ONLY)
1) INTRO (2-3 min)
   - Hook (compelling question/fact)
   - Big Picture (why this matters; the "big problem")
   - Paper ID (title + lead authors; main goal in one sentence)
2) PROBLEM & BACKGROUND (5-7 min)
   - State of field before paper; key limitations/open questions
   - Core Concepts (2-3, explained plainly with brief analogies)
3) THE CONTRIBUTION—WHAT DID THEY DO? (7-10 min)
   - Method (intuition-first; model/dataset/exp design)
   - The "Aha!" (key innovation)
4) RESULTS & IMPLICATIONS (5-7 min)
   - Main findings (top takeaways with metrics if present)
   - Why it matters (broader implications; shifts/possibilities)
   - Limitations/Open questions
5) CONCLUSION (2-3 min)
   - Recap (1-2 sentences)
   - Final thought (forward-looking or question)
   - Outro: "Thanks for tuning in to this research breakdown. Today we covered..."

TTS FRIENDLINESS
- Read common acronyms as words (e.g., "NLP"); otherwise spell out on first use.
- For tricky numbers, write them how they should be spoken (e.g., "about three thousand nine hundred").
- Use [PAUSE] sparingly for emphasis and [TRANSITION] for section changes.

RESOURCES (GROUND PACK)
{ground_pack}

PAPER (verbatim for reference if needed):
--- BEGIN PAPER ---
{paper_content}
--- END PAPER ---

OUTPUT ONLY THE SCRIPT IN THE REQUIRED SPEAKER FORMAT."""


class PodcastAgent:
    """Generate podcast scripts using 4-prompt Ground Pack system."""

    # Unlike the OpenAI-backed agents, this class has no module-level singleton
    # (webui/routers/api/podcast.py constructs it per request), so the lazy-proxy
    # pattern used by SummarizerAgent/SynthesisAgent/QuizAgent does not apply
    # here: there is nothing built at import time to fail, and a missing key
    # already raises a ValueError that names the variable.
    def __init__(self):
        settings = get_settings()
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for podcast generation")

        # Configure client with timeout to prevent hangs
        self.client = AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=httpx.Timeout(120.0, connect=10.0)  # 2 min for long prompts
        )
        self.model = "claude-sonnet-4-5-20250929"

        # Used by _get_full_text to pull the paper body out of its PDF.
        self.ingest_agent = IngestAgent()

    async def _call_claude(self, system: str, user: str, max_tokens: int = 4000) -> str:
        """Make streaming API call to Claude."""
        full_text = ""

        async with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}]
        ) as stream:
            async for text in stream.text_stream:
                full_text += text

            # Real token counts per call. Full paper text goes to five calls, so
            # this is how the actual bill is observed rather than estimated.
            try:
                usage = (await stream.get_final_message()).usage
                logger.info(
                    f"Claude call usage: input={usage.input_tokens} "
                    f"output={usage.output_tokens} tokens"
                )
            except Exception as e:  # usage logging must never fail a generation
                logger.debug(f"Could not read token usage: {e}")

        return full_text

    async def _generate_facts_outline(self, paper_content: str) -> str:
        """Generate facts-only outline (Prompt 1)."""
        logger.info("Generating facts outline...")
        return await self._call_claude(
            system=PROMPT_1_FACTS_OUTLINE,
            user=f"Analyze this paper:\n\n{paper_content}"
        )

    async def _generate_core_concepts(self, paper_content: str) -> str:
        """Generate core concepts and analogies (Prompt 2)."""
        logger.info("Generating core concepts...")
        return await self._call_claude(
            system=PROMPT_2_CORE_CONCEPTS,
            user=f"Identify the 3 most complex or novel concepts from this paper:\n\n{paper_content}"
        )

    async def _generate_metrics_datasets(self, paper_content: str) -> str:
        """Generate metrics and datasets table (Prompt 3)."""
        logger.info("Generating metrics and datasets...")
        return await self._call_claude(
            system=PROMPT_3_METRICS_DATASETS,
            user=f"Extract all quantitative data, datasets, and benchmarks from this paper:\n\n{paper_content}"
        )

    async def _generate_limitations(self, paper_content: str) -> str:
        """Generate limitations and threats (Prompt 4)."""
        logger.info("Generating limitations...")
        return await self._call_claude(
            system=PROMPT_4_LIMITATIONS,
            user=f"Critically analyze the limitations and weaknesses of this paper:\n\n{paper_content}"
        )

    async def _generate_final_script(self, ground_pack: dict, paper_content: str) -> str:
        """Generate the final podcast script using Ground Pack results."""
        logger.info("Generating final podcast script...")

        # Format ground pack for prompt
        ground_pack_text = f"""
FACTS-ONLY OUTLINE:
{ground_pack['facts_outline']}

CORE CONCEPTS & ANALOGIES:
{ground_pack['core_concepts']}

METRICS & DATASETS:
{ground_pack['metrics_datasets']}

LIMITATIONS & THREATS:
{ground_pack['limitations']}
"""

        # Use the final synthesis prompt
        prompt = FINAL_SYNTHESIS_PROMPT.format(
            ground_pack=ground_pack_text,
            paper_content=paper_content
        )

        return await self._call_claude(
            system="You are an expert podcast scriptwriter.",
            user=prompt,
            max_tokens=16000  # Longer output for full script
        )

    def _extract_key_points(self, ground_pack: dict) -> list:
        """Extract key points from the facts outline."""
        facts = ground_pack.get('facts_outline', '')

        # Simple extraction of bullet points
        lines = facts.split('\n')
        key_points = []

        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                point = line[2:].strip()
                if len(point) > 10:  # Filter out very short points
                    key_points.append(point)

        # Limit to top 5 key points
        return key_points[:5]

    def _get_full_text(self, paper: PaperRef) -> FullTextResult:
        """Extract full text from paper PDF using IngestAgent.

        Blocking: this downloads and parses a PDF. Call it through
        ``asyncio.to_thread`` from async code — see ``generate``.

        Args:
            paper: Paper reference with url_pdf

        Returns:
            FullTextResult: the text, or empty text plus the reason it is empty.
            Failures are still not raised — an unreadable PDF is a legitimate
            state for a paper that has an abstract and notes — but the reason
            travels with the emptiness so ``generate`` can report it.
        """
        if not paper.url_pdf:
            logger.warning(f"No PDF URL for paper {paper.id}, cannot extract full text")
            return FullTextResult("", "the paper has no PDF URL")

        try:
            start_time = time.time()
            logger.info(f"Extracting full text from PDF: {paper.url_pdf}")

            parsed_paper = self.ingest_agent.ingest(paper)
            elapsed = time.time() - start_time

            logger.info(
                f"Extracted {len(parsed_paper.full_text)} chars from PDF in {elapsed:.1f}s"
            )
            if not parsed_paper.full_text.strip():
                return FullTextResult(
                    "", f"no text could be extracted from {paper.url_pdf}"
                )
            return FullTextResult(parsed_paper.full_text)

        except IngestError as e:
            logger.error(f"Failed to extract text for {paper.id}: {e}")
            return FullTextResult("", str(e))
        except Exception as e:
            logger.error(f"Unexpected error extracting text for {paper.id}: {e}")
            return FullTextResult("", f"{type(e).__name__}: {e}")

    @staticmethod
    def _assess_source_material(
        paper: PaperRef,
        notes: Optional[PaperNote],
        full_text: FullTextResult,
    ) -> SourceMaterial:
        """Decide whether there is enough to write a podcast from, and say so.

        Three levels of honesty, not two:

        - **full** — the paper body was read. The normal path.
        - **partial** — no body, but an abstract and/or an extracted notes row.
          This proceeds, deliberately: a notes row carries the problem, method,
          findings and limitations that prompts 3 and 4 ask for, and an abstract
          is real content written by the authors. Refusing here would also block
          every paper whose PDF host is briefly unreachable. But it is recorded
          — in the response, on the page and in the stored row — because the bug
          being fixed is precisely that "thin" and "complete" looked identical.
        - **nothing** — no body, no abstract, no notes. ``generate`` raises.

        Raises:
            InsufficientSourceMaterialError: level would be "nothing".
        """
        has_text = bool(full_text.text.strip())
        has_abstract = bool(paper.abstract and paper.abstract.strip())
        has_notes = notes is not None

        if has_text:
            return SourceMaterial(
                level="full",
                full_text_chars=len(full_text.text),
                has_abstract=has_abstract,
                has_notes=has_notes,
                warnings=[],
            )

        reason = full_text.error or "the paper body could not be read"

        if not has_abstract and not has_notes:
            raise InsufficientSourceMaterialError(
                f"Cannot generate a podcast for \"{paper.title}\": {reason}, and the "
                f"paper has no abstract and no extracted notes. There is nothing to "
                f"write from, so no model calls were made."
            )

        available = " and ".join(
            part for part, present in (("its abstract", has_abstract), ("extracted notes", has_notes))
            if present
        )
        return SourceMaterial(
            level="partial",
            full_text_chars=0,
            has_abstract=has_abstract,
            has_notes=has_notes,
            warnings=[
                f"The full text of this paper was not available ({reason}). The script "
                f"was written from {available} only, so it covers far less of the paper "
                f"than usual and cannot contain results, hyperparameters or numbers that "
                f"appear only in the body."
            ],
        )

    async def generate(self, paper_id: str, pillar_id: str) -> PodcastScript:
        """
        Generate complete podcast script for a paper.

        Args:
            paper_id: ID of the paper to generate script for
            pillar_id: Associated pillar ID

        Returns:
            PodcastScript with generated content, carrying a SourceMaterial record
            of what it was actually written from.

        Raises:
            ValueError: no such paper.
            InsufficientSourceMaterialError: the paper body could not be read and
                the paper has neither an abstract nor a notes row. Raised before
                the first model call, so nothing is spent.
        """
        total_start = time.time()
        logger.info(f"Starting podcast generation for paper {paper_id}")

        # Fetch paper data
        paper = get_paper_by_id(paper_id)
        if not paper:
            raise ValueError(f"Paper not found: {paper_id}")

        # Fetch notes for additional context
        notes = get_notes_by_paper_id(paper_id)

        # Get full text from the PDF. This is the whole point of the Ground Pack:
        # prompts 3 and 4 ask for hyperparameters, SOTA numbers and the
        # Limitations section, none of which exist in an abstract.
        #
        # to_thread because _get_full_text downloads and parses a PDF (~7s cold)
        # and generate() is awaited directly from a FastAPI route — running it
        # inline froze the event loop for every other request.
        full_text_result = await asyncio.to_thread(self._get_full_text, paper)

        # Decide whether there is anything to write from BEFORE spending a
        # single token. _assess_source_material raises when the body is missing
        # and there is no abstract and no notes; five calls against a title and
        # the string "[Full text not available...]" is $0.27 of confident
        # fiction with a green success message on top.
        source_material = self._assess_source_material(paper, notes, full_text_result)
        for warning in source_material.warnings:
            logger.warning(f"Podcast for {paper_id}: {warning}")

        full_text = full_text_result.text

        # Truncate if too long (keep under 100K chars for Claude context safety)
        if len(full_text) > MAX_FULL_TEXT_CHARS:
            logger.warning(
                f"Truncating full_text from {len(full_text)} to {MAX_FULL_TEXT_CHARS} chars"
            )
            full_text = full_text[:MAX_FULL_TEXT_CHARS] + "\n\n[TRUNCATED - paper continues...]"

        # Assemble paper content with FULL TEXT
        paper_content = f"""
Title: {paper.title}
Authors: {', '.join(paper.authors) if paper.authors else 'Unknown'}
Year: {paper.year or 'Unknown'}
Abstract: {paper.abstract or 'No abstract available'}

=== FULL PAPER TEXT ===
{full_text if full_text else '[Full text not available - using abstract and notes only]'}
=== END FULL TEXT ===
"""

        # Add notes content if available (supplements full text with structured data)
        if notes:
            paper_content += f"""
=== EXTRACTED NOTES ===
Problem Statement: {notes.problem}
Methodology: {notes.method}
Key Findings: {', '.join(notes.findings) if notes.findings else 'N/A'}
Limitations: {', '.join(notes.limitations) if notes.limitations else 'N/A'}
Future Work: {', '.join(notes.future_work) if notes.future_work else 'N/A'}
Key Terms: {', '.join(notes.key_terms) if notes.key_terms else 'N/A'}
=== END NOTES ===
"""

        logger.info(f"Paper content assembled: {len(paper_content)} chars")

        # Run 4 Ground Pack prompts sequentially
        ground_pack = {
            "facts_outline": await self._generate_facts_outline(paper_content),
            "core_concepts": await self._generate_core_concepts(paper_content),
            "metrics_datasets": await self._generate_metrics_datasets(paper_content),
            "limitations": await self._generate_limitations(paper_content)
        }

        # Generate final script
        script = await self._generate_final_script(ground_pack, paper_content)

        # Create PodcastScript
        podcast_script = PodcastScript(
            paper_id=paper_id,
            pillar_id=pillar_id,
            title=f"Deep Dive: {paper.title}",
            script=script,
            word_count=len(script.split()),
            key_points=self._extract_key_points(ground_pack),
            ground_pack=ground_pack,
            source_material=source_material,
            created_at=datetime.now()
        )

        total_elapsed = time.time() - total_start
        logger.info(
            f"Generated podcast script with {podcast_script.word_count} words "
            f"in {total_elapsed:.1f}s"
        )
        return podcast_script
