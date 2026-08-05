"""Podcast Agent for generating educational podcast scripts from papers.
Uses a 4-prompt Ground Pack system with Claude Sonnet 4.
"""

import logging
import time
from datetime import datetime

import httpx
from anthropic import AsyncAnthropic

from ..config import get_settings
from ..db import get_notes_by_paper_id, get_paper_by_id
from ..schemas import PaperRef, PodcastScript
from .ingest_agent import IngestAgent, IngestError

logger = logging.getLogger(__name__)


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

        # Initialize IngestAgent for full text extraction
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

    def _get_full_text(self, paper: PaperRef) -> str:
        """Extract full text from paper PDF using IngestAgent.

        Args:
            paper: Paper reference with url_pdf

        Returns:
            Full text content or empty string on failure

        """
        if not paper.url_pdf:
            logger.warning(f"No PDF URL for paper {paper.id}, cannot extract full text")
            return ""

        try:
            start_time = time.time()
            logger.info(f"Extracting full text from PDF: {paper.url_pdf}")

            parsed_paper = self.ingest_agent.ingest(paper)
            elapsed = time.time() - start_time

            logger.info(
                f"Extracted {len(parsed_paper.full_text)} chars from PDF in {elapsed:.1f}s"
            )
            return parsed_paper.full_text

        except IngestError as e:
            logger.error(f"Failed to extract text for {paper.id}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error extracting text for {paper.id}: {e}")
            return ""

    async def generate(self, paper_id: str, pillar_id: str) -> PodcastScript:
        """Generate complete podcast script for a paper.

        Args:
            paper_id: ID of the paper to generate script for
            pillar_id: Associated pillar ID

        Returns:
            PodcastScript with generated content

        """
        total_start = time.time()
        logger.info(f"Starting podcast generation for paper {paper_id}")

        # Fetch paper data
        paper = get_paper_by_id(paper_id)
        if not paper:
            raise ValueError(f"Paper not found: {paper_id}")

        # Fetch notes for additional context
        notes = get_notes_by_paper_id(paper_id)

        # CRITICAL: Get full text from PDF (this was the missing piece!)
        full_text = self._get_full_text(paper)

        # Truncate if too long (keep under 100K chars for Claude context safety)
        max_chars = 100000
        if len(full_text) > max_chars:
            logger.warning(f"Truncating full_text from {len(full_text)} to {max_chars} chars")
            full_text = full_text[:max_chars] + "\n\n[TRUNCATED - paper continues...]"

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

        # Log content size for debugging
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
            created_at=datetime.now()
        )

        total_elapsed = time.time() - total_start
        logger.info(
            f"Generated podcast script with {podcast_script.word_count} words "
            f"in {total_elapsed:.1f}s"
        )
        return podcast_script
