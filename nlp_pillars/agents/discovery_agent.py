"""
Discovery Agent - Generates targeted search queries for research paper discovery.
"""

from typing import List, Optional
import instructor
from openai import OpenAI
# TODO: Replace with proper atomic_agents imports when v2.0+ is available
# from atomic_agents import AtomicAgent, AgentConfig
# from atomic_agents.context import SystemPromptGenerator, ChatHistory

# Temporary mock classes for development
from typing import Any, Generic, TypeVar, List
from pydantic import BaseModel

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
    SearchQuery, 
    PillarID,
    PillarConfig
)
from ..config import get_settings, PILLAR_CONFIGS


class DiscoveryAgent:
    """Agent that generates targeted search queries for research paper discovery."""
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize the Discovery Agent.
        
        Args:
            model: Optional model name override. Defaults to config setting.
        """
        settings = get_settings()
        self.model = model or settings.default_model
        
        # Create the system prompt generator
        self.system_prompt = SystemPromptGenerator(
            background=[
                "You are an expert NLP research strategist specializing in scientific paper discovery.",
                "You have deep knowledge of current research trends across all areas of natural language processing.",
                "You understand how to craft targeted search queries that find high-quality, relevant academic papers.",
                "You are familiar with academic search engines and their query syntax patterns.",
                "You stay current with emerging research areas and can identify knowledge gaps."
            ],
            steps=[
                "Analyze the given pillar configuration to understand the learning focus and goals",
                "Review recently processed papers to identify covered topics and potential gaps",
                "Consider the pillar's specific focus areas and current research priorities", 
                "Generate 3-5 diverse search queries that target different aspects of the pillar",
                "Ensure queries are specific enough to find relevant papers but broad enough to discover new perspectives",
                "Balance between core foundational topics and cutting-edge emerging research",
                "Include both theoretical and applied research perspectives where appropriate"
            ],
            output_instructions=[
                "Generate exactly 3-5 SearchQuery objects with varied focus areas",
                "Make each query distinctive - avoid redundant or overlapping searches",
                "Use precise academic terminology and concepts relevant to the pillar",
                "Set appropriate max_results (typically 5-15 per query)",
                "Include relevant filters when beneficial (categories, time ranges, etc.)",
                "Provide a clear, educational rationale explaining the strategy behind your query selection",
                "Ensure queries will find papers suitable for the pillar's learning objectives",
                "Consider both seminal works and recent developments (last 2-3 years)"
            ]
        )
        
        # Initialize the OpenAI client with instructor  
        client = instructor.from_openai(OpenAI(api_key=settings.openai_api_key))
        
        # Create the Atomic Agent
        self.agent = AtomicAgent[DiscoveryInput, DiscoveryOutput](
            config=AgentConfig(
                client=client,
                model=self.model,
                system_prompt_generator=self.system_prompt,
                history=ChatHistory(),
            )
        )
    
    def discover(
        self,
        pillar: PillarConfig,
        recent_papers: Optional[List[str]] = None,
        priority_topics: Optional[List[str]] = None
    ) -> DiscoveryOutput:
        """
        Generate search queries for discovering relevant research papers.
        
        Args:
            pillar: Pillar configuration with learning goals and focus areas
            recent_papers: Recently processed paper titles/IDs for context
            priority_topics: High-priority research areas to focus on
            
        Returns:
            DiscoveryOutput containing SearchQuery objects and rationale
        """
        # Prepare input
        agent_input = DiscoveryInput(
            pillar=pillar,
            recent_papers=recent_papers or [],
            priority_topics=priority_topics or []
        )
        
        # Run the agent
        result = self.agent.run(agent_input)
        
        # Validate result has required number of queries
        if len(result.queries) < 3:
            raise ValueError(f"Discovery agent must generate at least 3 queries, got {len(result.queries)}")
        
        if len(result.queries) > 5:
            # Truncate to max 5 queries
            result.queries = result.queries[:5]
        
        # Ensure all queries have the correct pillar_id
        for query in result.queries:
            query.pillar_id = pillar.id
        
        return result
    
    def discover_for_pillar_id(
        self,
        pillar_id: PillarID,
        recent_papers: Optional[List[str]] = None,
        priority_topics: Optional[List[str]] = None
    ) -> DiscoveryOutput:
        """
        Generate search queries for a specific pillar by ID.
        
        Args:
            pillar_id: The pillar ID to discover papers for
            recent_papers: Recently processed paper titles/IDs for context
            priority_topics: High-priority research areas to focus on
            
        Returns:
            DiscoveryOutput containing SearchQuery objects and rationale
        """
        # Get pillar configuration
        pillar_config_dict = PILLAR_CONFIGS.get(pillar_id.value)
        if not pillar_config_dict:
            raise ValueError(f"Unknown pillar ID: {pillar_id}")
        
        # Create PillarConfig object
        pillar = PillarConfig(
            id=pillar_id,
            name=pillar_config_dict["name"],
            goal=pillar_config_dict["goal"],
            focus_areas=pillar_config_dict["focus_areas"]
        )
        
        return self.discover(pillar, recent_papers, priority_topics)
    
    def discover_batch(
        self,
        pillars: List[PillarConfig],
        max_queries_per_pillar: int = 3
    ) -> List[DiscoveryOutput]:
        """
        Generate search queries for multiple pillars.
        
        Args:
            pillars: List of pillar configurations
            max_queries_per_pillar: Maximum queries to generate per pillar
            
        Returns:
            List of DiscoveryOutput objects, one per pillar
        """
        results = []
        
        for pillar in pillars:
            try:
                # Temporarily limit queries for batch processing
                original_instructions = self.system_prompt.output_instructions
                
                # Update instructions for batch mode
                batch_instructions = [instr.replace("3-5", f"exactly {max_queries_per_pillar}") 
                                    for instr in original_instructions]
                self.system_prompt.output_instructions = batch_instructions
                
                # Discover for this pillar
                result = self.discover(pillar)
                results.append(result)
                
                # Restore original instructions
                self.system_prompt.output_instructions = original_instructions
                
            except Exception as e:
                print(f"Warning: Discovery failed for pillar {pillar.id}: {e}")
                continue
        
        return results
    
    def get_priority_topics_for_pillar(self, pillar_id: PillarID) -> List[str]:
        """
        Get current priority research topics for a specific pillar.
        
        Args:
            pillar_id: The pillar to get priority topics for
            
        Returns:
            List of priority research topics
        """
        # This could be enhanced to query a database or use more sophisticated
        # topic modeling, but for now we'll use the focus areas
        pillar_config = PILLAR_CONFIGS.get(pillar_id.value, {})
        return pillar_config.get("focus_areas", [])
    
    def reset_context(self):
        """Reset the agent's conversation history."""
        self.agent.config.history.clear()


# Query generation helpers
class QueryStrategies:
    """Helper class with common query generation strategies."""
    
    @staticmethod
    def build_foundational_query(pillar: PillarConfig) -> SearchQuery:
        """Build a query targeting foundational research in the pillar area."""
        focus_terms = " ".join(pillar.focus_areas[:2])  # Use first 2 focus areas
        
        return SearchQuery(
            pillar_id=pillar.id,
            query=f"{focus_terms} foundational research survey",
            filters={"time_range": "5years", "categories": ["cs.CL", "cs.LG"]},
            max_results=10
        )
    
    @staticmethod
    def build_recent_advances_query(pillar: PillarConfig) -> SearchQuery:
        """Build a query targeting recent advances in the pillar area."""
        focus_terms = " ".join(pillar.focus_areas[1:3])  # Use middle focus areas
        
        return SearchQuery(
            pillar_id=pillar.id,
            query=f"{focus_terms} recent advances 2023 2024",
            filters={"time_range": "year", "categories": ["cs.CL"]},
            max_results=8
        )
    
    @staticmethod
    def build_applications_query(pillar: PillarConfig) -> SearchQuery:
        """Build a query targeting practical applications."""
        focus_terms = pillar.focus_areas[-1]  # Use last focus area
        
        return SearchQuery(
            pillar_id=pillar.id,
            query=f"{focus_terms} applications real-world deployment",
            filters={"categories": ["cs.CL", "cs.AI"]},
            max_results=7
        )


# Example usage
if __name__ == "__main__":
    from datetime import datetime
    
    # Create agent
    agent = DiscoveryAgent()
    
    # Example pillar configuration
    pillar = PillarConfig(
        id=PillarID.P2,
        name="Models & Architectures",
        goal="Understand cutting-edge model architectures and emerging paradigms",
        focus_areas=["Transformer variants", "Long-context models", "Multimodal architectures"],
        created_at=datetime.now()
    )
    
    # Example recent papers
    recent_papers = [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners"
    ]
    
    try:
        # Discover new papers
        result = agent.discover(
            pillar=pillar,
            recent_papers=recent_papers,
            priority_topics=["state space models", "mixture of experts"]
        )
        
        print(f"Generated {len(result.queries)} search queries:")
        print(f"Rationale: {result.rationale}")
        print()
        
        for i, query in enumerate(result.queries, 1):
            print(f"Query {i}:")
            print(f"  Text: {query.query}")
            print(f"  Max results: {query.max_results}")
            print(f"  Filters: {query.filters}")
            print()
            
    except Exception as e:
        print(f"Discovery failed: {e}")
