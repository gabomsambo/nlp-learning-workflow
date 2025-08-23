"""
Summarizer Agent - Extracts structured information from research papers.
"""

from typing import List, Optional
import instructor
from openai import OpenAI
# TODO: Replace with proper atomic_agents imports when v2.0+ is available  
# from atomic_agents import AtomicAgent, AgentConfig
# from atomic_agents.context import SystemPromptGenerator, ChatHistory

# Using the temporary mock classes from discovery_agent
from .discovery_agent import AtomicAgent, AgentConfig, SystemPromptGenerator, ChatHistory

from ..schemas import ParsedPaper, PaperNote, PillarID, SummarizerInput
from ..config import get_settings


class SummarizerAgent:
    """Agent that summarizes research papers into structured notes."""
    
    def __init__(self, model: Optional[str] = None):
        """Initialize the Summarizer Agent.
        
        Args:
            model: Optional model name override. Defaults to config setting.
        """
        settings = get_settings()
        self.model = model or settings.default_model
        
        # Create the system prompt generator
        self.system_prompt = SystemPromptGenerator(
            background=[
                "You are an expert NLP researcher specializing in extracting key information from academic papers.",
                "You have deep knowledge of machine learning, linguistics, and computational methods.",
                "You are skilled at identifying the core contributions and limitations of research work.",
                "You maintain objectivity and accuracy in your summaries."
            ],
            steps=[
                "Carefully read the provided research paper text",
                "Identify the main problem or research question being addressed",
                "Extract the methodology - what approach, model, or technique is proposed",
                "List the key findings with specific metrics or results when available",
                "Identify stated limitations or weaknesses acknowledged by the authors",
                "Note future work directions mentioned in the paper",
                "Extract important technical terms and concepts introduced or used"
            ],
            output_instructions=[
                "Be concise but comprehensive in your extraction",
                "Use clear, technical language appropriate for researchers",
                "Include specific numbers, metrics, or results when mentioned",
                "Distinguish between claimed contributions and actual validated results",
                "List items in order of importance",
                "Limit findings and limitations to 3-5 items each",
                "Extract 5-10 key technical terms"
            ]
        )
        
        # Initialize the OpenAI client with instructor
        client = instructor.from_openai(OpenAI(api_key=settings.openai_api_key))
        
        # Create the Atomic Agent
        self.agent = AtomicAgent[SummarizerInput, PaperNote](
            config=AgentConfig(
                client=client,
                model=self.model,
                system_prompt_generator=self.system_prompt,
                history=ChatHistory(),
            )
        )
    
    def summarize(
        self, 
        parsed_paper: ParsedPaper,
        pillar_id: PillarID,
        recent_notes: Optional[List[str]] = None
    ) -> PaperNote:
        """Summarize a research paper into structured notes.
        
        Args:
            parsed_paper: The parsed paper content
            pillar_id: The learning pillar this paper belongs to
            recent_notes: Recent paper summaries for context consistency
        
        Returns:
            Structured PaperNote with extracted information
        """
        # Prepare input
        agent_input = SummarizerInput(
            parsed_paper=parsed_paper,
            pillar_id=pillar_id,
            recent_notes=recent_notes or []
        )
        
        # Run the agent
        result = self.agent.run(agent_input)
        
        # Add paper ID and pillar ID to the result
        result.paper_id = parsed_paper.paper_ref.id
        result.pillar_id = pillar_id
        
        return result
    
    def summarize_batch(
        self,
        papers: List[ParsedPaper],
        pillar_id: PillarID
    ) -> List[PaperNote]:
        """Summarize multiple papers, maintaining context between them.
        
        Args:
            papers: List of parsed papers to summarize
            pillar_id: The learning pillar these papers belong to
        
        Returns:
            List of structured PaperNotes
        """
        notes = []
        recent_summaries = []
        
        for paper in papers:
            # Use previous summaries as context
            note = self.summarize(
                parsed_paper=paper,
                pillar_id=pillar_id,
                recent_notes=recent_summaries[-3:]  # Last 3 summaries for context
            )
            
            notes.append(note)
            
            # Add a brief summary to context for next paper
            summary = f"{paper.paper_ref.title}: {note.problem} - {note.method}"
            recent_summaries.append(summary[:200])  # Limit length
        
        return notes
    
    def reset_context(self):
        """Reset the agent's conversation history."""
        self.agent.config.history.clear()


# Example usage
if __name__ == "__main__":
    # This would typically be called from the orchestrator
    from ..schemas import PaperRef
    
    # Create sample data
    sample_paper_ref = PaperRef(
        id="2401.12345",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        year=2017,
        venue="NeurIPS",
        url_pdf="https://arxiv.org/pdf/1706.03762.pdf"
    )
    
    sample_parsed_paper = ParsedPaper(
        paper_ref=sample_paper_ref,
        full_text="[Paper text would go here...]",
        chunks=["chunk1", "chunk2"],
        figures_count=5,
        tables_count=3,
        references=["ref1", "ref2"]
    )
    
    # Initialize agent
    agent = SummarizerAgent()
    
    # Summarize paper
    note = agent.summarize(
        parsed_paper=sample_parsed_paper,
        pillar_id=PillarID.P2,  # Models & Architectures
        recent_notes=[]
    )
    
    print(f"Problem: {note.problem}")
    print(f"Method: {note.method}")
    print(f"Key findings: {note.findings}")
