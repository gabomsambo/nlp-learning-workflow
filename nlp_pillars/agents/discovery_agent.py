"""
Discovery Agent - Fixed version with working implementation.
"""

from typing import List, Any, Generic, TypeVar

# Temporary mock classes for development (until atomic_agents v2.0+ is available)
T = TypeVar('T')
U = TypeVar('U')

class SystemPromptGenerator:
    def __init__(self, background: List[str], steps: List[str], output_instructions: List[str]):
        self.background = background
        self.steps = steps
        self.output_instructions = output_instructions

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
from ..schemas import (
    DiscoveryInput,
    DiscoveryOutput,
    SearchQuery
)
from typing import List


class DiscoveryAgent:
    """Agent that generates targeted search queries for research paper discovery."""

    @classmethod
    def run(cls, input_data: DiscoveryInput) -> DiscoveryOutput:
        """Generate search queries based on pillar configuration.

        This implementation blends user topics with pillar context - user topics
        guide but don't override the pillar's focus areas.
        """
        pillar = input_data.pillar
        user_topics = input_data.priority_topics

        # ALWAYS start with pillar context
        base_keywords = cls._extract_keywords(pillar.goal)
        focus_areas = pillar.focus_areas.copy()

        # Blend user topics with pillar focus areas (guide, not override)
        if user_topics:
            blended_topics = cls._blend_topics(focus_areas, user_topics)
        else:
            blended_topics = focus_areas

        # Generate queries that ALWAYS include pillar context
        queries = []

        # Query 1: Goal + user guidance
        q1_parts = [base_keywords]
        if user_topics:
            q1_parts.append(user_topics[0])
        queries.append(SearchQuery(
            pillar_id=pillar.id,
            query=f"{' '.join(q1_parts)} research 2024",
            max_results=10
        ))

        # Query 2: Blended focus areas
        if len(blended_topics) >= 2:
            queries.append(SearchQuery(
                pillar_id=pillar.id,
                query=f"{blended_topics[0]} {blended_topics[1]} methods",
                max_results=10
            ))
        elif blended_topics:
            queries.append(SearchQuery(
                pillar_id=pillar.id,
                query=f"{blended_topics[0]} methods techniques",
                max_results=10
            ))

        # Query 3: Deep dive into blended topic
        if blended_topics:
            area_idx = min(2, len(blended_topics) - 1)
            queries.append(SearchQuery(
                pillar_id=pillar.id,
                query=f"{blended_topics[area_idx]} benchmarks evaluation",
                max_results=10
            ))

        rationale = f"Generated pillar-specific queries for {pillar.name}: targeting {pillar.goal}"
        if user_topics:
            rationale += f" with user guidance: {', '.join(user_topics)}"

        return DiscoveryOutput(
            queries=queries[:3],
            rationale=rationale
        )

    @staticmethod
    def _blend_topics(pillar_areas: List[str], user_topics: List[str]) -> List[str]:
        """Blend user topics with pillar areas. User guides, pillar anchors.

        This ensures user input guides discovery within pillar context,
        but never completely overrides the pillar's focus areas.

        Args:
            pillar_areas: Pillar's focus areas
            user_topics: User-provided topic hints

        Returns:
            Blended list of topics prioritizing matches and user guidance
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
                # User topic doesn't match pillar area directly
                # Add it but keep pillar context by including nearby pillar areas
                result.append(topic)

        # Always include pillar areas not yet added (to maintain pillar context)
        for i, area in enumerate(pillar_areas):
            if i not in matched_pillar_indices and area not in result:
                result.append(area)

        return result

    @staticmethod
    def _extract_keywords(goal: str) -> str:
        """Extract key phrases from goal for search queries.

        Removes common stopwords and returns meaningful search terms.
        """
        stopwords = {
            'master', 'understand', 'learn', 'develop', 'expertise', 'explore',
            'consider', 'in', 'and', 'the', 'of', 'for', 'with', 'to', 'between',
            'a', 'an', 'on', 'is', 'are', 'that', 'this', 'be'
        }
        words = [w.lower() for w in goal.split() if w.lower() not in stopwords]
        # Return first 4 meaningful words to keep query focused
        return ' '.join(words[:4])
