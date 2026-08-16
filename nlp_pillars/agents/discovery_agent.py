"""Discovery Agent - real LLM implementation using instructor + OpenAI.

Generates the search queries that feed paper discovery. Until 2026-08-16 this
module was a stub: `run()` was a classmethod that string-mangled the pillar goal
into three templated queries ("<keywords> research 2024", "<area> methods", ...)
and never called a model. The four `atomic_agents` stand-in classes below carry
a comment about being temporary "until atomic_agents v2.0+ is available" —
atomic-agents 2.9.1 is pinned and installed now, but see the note on those
classes for why they are still here.

Conventions match summarizer/synthesis/quiz (see AGENTS.md, "Agent LLM
conventions"):

- **No hand-rolled retry.** instructor re-prompts with validation feedback
  (`max_retries` defaults to 3) and raises `InstructorRetryException`, which is
  *not* a `pydantic.ValidationError`. One `create()` call, failures wrapped
  `from e`.
- **Lazy construction.** `orchestrator` imports `DiscoveryAgent` at module load,
  so nothing may build an OpenAI client at import time.
"""

import logging
from typing import Any, Generic, List, Optional, TypeVar

import instructor
from openai import OpenAI

logger = logging.getLogger(__name__)

T = TypeVar('T')
U = TypeVar('U')


# These four are stand-ins for atomic_agents types, and they are NOT dead code:
# summarizer_agent, synthesis_agent, quiz_agent, ingest_agent and summarizer all
# do `from .discovery_agent import SystemPromptGenerator` (summarizer imports all
# four). This module is therefore the de-facto home of the prompt scaffolding for
# every agent in the package. Deleting or renaming them breaks five importers
# that have nothing to do with discovery.
class SystemPromptGenerator:
    def __init__(self, background: List[str], steps: List[str], output_instructions: List[str]):
        self.background = background
        self.steps = steps
        self.output_instructions = output_instructions

    def render(self) -> str:
        """Flatten into a single system message."""
        parts = []
        if self.background:
            parts.append("Background:\n" + "\n".join(f"- {b}" for b in self.background))
        if self.steps:
            parts.append("Steps:\n" + "\n".join(f"- {s}" for s in self.steps))
        if self.output_instructions:
            parts.append("Output:\n" + "\n".join(f"- {o}" for o in self.output_instructions))
        return "\n\n".join(parts)


class ChatHistory:
    def __init__(self):
        self.messages = []

    def clear(self):
        self.messages = []


class AgentConfig:
    def __init__(self, client: Any, model: str, system_prompt_generator: SystemPromptGenerator, history: ChatHistory):
        self.client = client
        self.model = model
        self.system_prompt_generator = system_prompt_generator
        self.history = history


class AtomicAgent(Generic[T, U]):
    def __init__(self, config: AgentConfig):
        self.config = config

    def run(self, input_data: T) -> U:
        raise NotImplementedError("This is a mock implementation for testing only")


from ..config import PILLAR_CONFIGS, get_pillar_config, get_settings  # noqa: E402
from ..schemas import (  # noqa: E402
    DiscoveryInput,
    DiscoveryOutput,
    PillarConfig,
)

# Discovery exists to give the search stage several angles on a pillar. Fewer
# than three and the downstream `_search_candidates` fan-out collapses to
# roughly one source's opinion, so this is enforced rather than requested.
MIN_QUERIES = 3


class DiscoveryError(Exception):
    """Raised when query generation fails."""

    pass


class _DiscoveryCompletion:
    """One structured-output call. Tests replace this wholesale as `agent.agent`.

    Kept as its own object rather than a method so a test can swap in a Mock
    whose `.run()` returns a canned DiscoveryOutput without also having to stand
    up an OpenAI client.
    """

    def __init__(self, client, model: str, system_prompt: SystemPromptGenerator):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt

    def run(self, input_data: DiscoveryInput) -> DiscoveryOutput:
        return self.client.chat.completions.create(
            model=self.model,
            response_model=DiscoveryOutput,
            # Higher than the summarizer's 0.2 on purpose: the whole value of
            # this agent is queries that differ from each other. Near-zero
            # temperature reliably produces three rephrasings of one idea.
            temperature=0.7,
            messages=[
                {"role": "system", "content": self.system_prompt.render()},
                {"role": "user", "content": _build_user_message(input_data)},
            ],
        )


def _build_user_message(input_data: DiscoveryInput) -> str:
    pillar = input_data.pillar
    topics = DiscoveryAgent._blend_topics(list(pillar.focus_areas), input_data.priority_topics)

    parts = [
        f"Pillar: {pillar.name} (id: {pillar.id})",
        f"Goal: {pillar.goal}",
        "",
        "Focus areas, most relevant first:",
        *(f"- {t}" for t in topics),
    ]
    if input_data.priority_topics:
        parts += [
            "",
            "The user asked to steer toward these topics. Let them guide the "
            "queries, but keep every query inside the pillar's remit:",
            *(f"- {t}" for t in input_data.priority_topics),
        ]
    if input_data.recent_papers:
        # Ids, not titles — that is all DiscoveryInput carries. Enough for the
        # model to avoid re-proposing an id it can see, not enough for it to
        # reason about content.
        parts += [
            "",
            "Already processed recently; do not aim queries at these:",
            *(f"- {p}" for p in input_data.recent_papers[:20]),
        ]
    parts += [
        "",
        f"Produce at least {MIN_QUERIES} search queries.",
    ]
    return "\n".join(parts)


def _make_client():
    """Create the instructor-wrapped OpenAI client.

    Raises:
        ValueError: if OPENAI_API_KEY is unset, naming the key rather than
            failing later with an AttributeError on None.

    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError(
            "DiscoveryAgent is not initialized: "
            "OPENAI_API_KEY environment variable is required"
        )
    return instructor.from_openai(OpenAI(api_key=settings.openai_api_key))


class DiscoveryAgent:
    """Generates targeted search queries for research paper discovery."""

    def __init__(self, model: Optional[str] = None):
        """Build the agent.

        Args:
            model: Override the model. Defaults to `Settings.default_model`,
                which is `gpt-4o` unless DEFAULT_MODEL says otherwise.

        """
        settings = get_settings()
        self.model = model or settings.default_model
        self.system_prompt = SystemPromptGenerator(
            background=[
                "You plan literature searches for a working NLP researcher.",
                "You know arXiv, ACL Anthology and Semantic Scholar conventions.",
            ],
            steps=[
                "Read the pillar goal and its focus areas.",
                "Pick distinct angles: core method, a recent development, an "
                "application or evaluation. Do not paraphrase one idea three times.",
                "Phrase each query the way papers on that topic describe "
                "themselves, not as a question.",
            ],
            output_instructions=[
                f"Return at least {MIN_QUERIES} queries.",
                "Every query must stay within the pillar's remit.",
                "Set pillar_id on every query to the pillar id you were given.",
                "Explain the spread of angles in `rationale`.",
                "Return valid DiscoveryOutput JSON only (no extra text).",
            ],
        )
        self.agent = _DiscoveryCompletion(_make_client(), self.model, self.system_prompt)

    def discover(
        self,
        pillar: PillarConfig,
        recent_papers: Optional[List[str]] = None,
        priority_topics: Optional[List[str]] = None,
    ) -> DiscoveryOutput:
        """Generate queries for an already-resolved pillar.

        Raises:
            ValueError: if the model returned fewer than MIN_QUERIES.
            DiscoveryError: if the completion itself failed.

        """
        input_data = DiscoveryInput(
            pillar=pillar,
            recent_papers=recent_papers or [],
            priority_topics=priority_topics or [],
        )

        try:
            result = self.agent.run(input_data)
        except Exception as e:
            logger.error(f"Discovery failed for pillar {pillar.id}: {e}")
            raise DiscoveryError(f"Instructor completion failed: {e}") from e

        if len(result.queries) < MIN_QUERIES:
            raise ValueError(
                f"DiscoveryAgent must generate at least {MIN_QUERIES} queries, "
                f"got {len(result.queries)}"
            )

        # The prompt asks for pillar_id and the model usually complies, but a
        # wrong one here would write papers into another pillar's namespace —
        # that is the isolation boundary, so it is enforced, not trusted.
        for query in result.queries:
            query.pillar_id = pillar.id

        logger.info(f"Discovery generated {len(result.queries)} queries for pillar {pillar.id}")
        return result

    def discover_for_pillar_id(
        self,
        pillar_id: str,
        priority_topics: Optional[List[str]] = None,
    ) -> DiscoveryOutput:
        """Resolve a pillar id, then discover for it.

        Args:
            pillar_id: A live pillar slug. Retired P1-P5 slugs raise, exactly as
                `get_pillar_config` does — discovering into a pillar that no
                longer exists would write rows nothing can read.

        """
        config = get_pillar_config(pillar_id)
        pillar = PillarConfig(
            id=config.get("id", pillar_id),
            name=config["name"],
            goal=config["goal"],
            papers_per_day=config.get("papers_per_day", 2),
            focus_areas=config.get("focus_areas", []),
        )
        return self.discover(pillar, priority_topics=priority_topics)

    def get_priority_topics_for_pillar(self, pillar_id: str) -> List[str]:
        """Return a pillar's focus areas from the static config.

        Reads PILLAR_CONFIGS directly rather than going through
        `get_pillar_config`, which tries the database first: this is prompt
        seasoning, not data, and it should not depend on the database being up.
        Returns [] for an unknown or retired slug.
        """
        return list(PILLAR_CONFIGS.get(pillar_id, {}).get("focus_areas", []))

    @classmethod
    def run(cls, input_data: DiscoveryInput) -> DiscoveryOutput:
        """Discover, via a lazily built shared agent.

        Deliberately a classmethod. `orchestrator` calls `DiscoveryAgent.run(x)`
        on the class (two sites) while `cli` builds an instance and calls
        `agent.run(x)`; a classmethod satisfies both, where a plain instance
        method would bind `x` to `self` for the orchestrator and an eagerly
        constructed singleton would need an OpenAI client at import time.

        The consequence worth knowing: `agent.run(...)` does NOT use `agent`. It
        uses the shared instance. Call `agent.discover(...)` if you need the
        object you built — for example one constructed with a different model.
        """
        return cls._shared().discover(
            input_data.pillar,
            recent_papers=input_data.recent_papers,
            priority_topics=input_data.priority_topics,
        )

    _shared_instance: Optional["DiscoveryAgent"] = None

    @classmethod
    def _shared(cls) -> "DiscoveryAgent":
        if cls._shared_instance is None:
            cls._shared_instance = cls()
        return cls._shared_instance

    @staticmethod
    def _blend_topics(pillar_areas: List[str], user_topics: List[str]) -> List[str]:
        """Order focus areas so user-requested ones come first.

        Predates the LLM rewrite, where it built query strings directly. It now
        orders the focus-area list handed to the model. The rule is unchanged and
        is the point of the function: user topics *guide*, the pillar *anchors* —
        a user topic never removes a pillar's own areas from consideration.

        Args:
            pillar_areas: Pillar's focus areas.
            user_topics: User-provided topic hints.

        Returns:
            Blended list prioritising matches and user guidance.

        """
        if not user_topics:
            return pillar_areas

        pillar_lower = [a.lower() for a in pillar_areas]
        result = []
        matched_pillar_indices = set()

        # First pass: find user topics that match pillar areas
        for topic in user_topics:
            topic_lower = topic.lower()
            matched = False

            for i, area in enumerate(pillar_lower):
                # Check for overlap between user topic and pillar area
                if topic_lower in area or area in topic_lower or \
                   any(word in area for word in topic_lower.split()):
                    # Boost matching pillar area to front
                    if i not in matched_pillar_indices:
                        result.append(pillar_areas[i])
                        matched_pillar_indices.add(i)
                    matched = True
                    break

            if not matched:
                # User topic doesn't match a pillar area; keep it, and the loop
                # below still contributes every pillar area, so context survives.
                result.append(topic)

        # Always include pillar areas not yet added (to maintain pillar context)
        for i, area in enumerate(pillar_areas):
            if i not in matched_pillar_indices and area not in result:
                result.append(area)

        return result

    @staticmethod
    def _extract_keywords(goal: str) -> str:
        """Reduce a pillar goal to its content words.

        Left in place because the stub's templated-query behaviour is still the
        documented fallback in `orchestrator._run_discovery`, and callers outside
        this package may use it.
        """
        stopwords = {
            'master', 'understand', 'learn', 'develop', 'expertise', 'explore',
            'consider', 'in', 'and', 'the', 'of', 'for', 'with', 'to', 'between',
            'a', 'an', 'on', 'is', 'are', 'that', 'this', 'be'
        }
        words = [w.lower() for w in goal.split() if w.lower() not in stopwords]
        # Return first 4 meaningful words to keep query focused
        return ' '.join(words[:4])
