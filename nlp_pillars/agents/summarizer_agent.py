"""
SummarizerAgent using Atomic Agents v2.0 + Instructor + OpenAI.

Implements structured paper summarization with validation and retry logic.
"""

import logging
import os
from typing import List
import instructor
from openai import OpenAI
from pydantic import ValidationError

# TODO: Replace with proper atomic_agents imports when v2.0+ is available
# from atomic_agents import AtomicAgent, AgentConfig
# from atomic_agents.context import SystemPromptGenerator, ChatHistory

# Using the temporary mock classes from discovery_agent
from .discovery_agent import AtomicAgent, AgentConfig, SystemPromptGenerator, ChatHistory

from ..schemas import SummarizerInput, PaperNote, ParsedPaper, PillarID
from ..config import get_settings

logger = logging.getLogger(__name__)


class SummarizerValidationError(Exception):
    """Raised when SummarizerAgent fails validation after retry."""
    pass


def _make_client() -> instructor.Instructor:
    """Create the Instructor-wrapped OpenAI client with sensible defaults."""
    settings = get_settings()
    
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    client = instructor.from_openai(
        OpenAI(
            api_key=settings.openai_api_key,
            timeout=60.0,  # 60 second timeout
        )
    )
    
    return client


# Create SystemPromptGenerator with specified content
system_prompt = SystemPromptGenerator(
    background=[
        "NLP research summarizer; faithful, structured",
        "Cite only what's supported by the text; avoid hallucinations"
    ],
    steps=[
        "Extract: problem, method, findings, limitations, future_work, key_terms",
        "Be concise but precise; prefer bullet points for lists"
    ],
    output_instructions=[
        "Return valid PaperNote JSON only (no extra text)"
    ]
)


class SummarizerAgentImpl:
    """Implementation class for the Summarizer Agent with retry logic."""
    
    def __init__(self, client: instructor.Instructor, model: str):
        self.client = client
        self.model = model
        
        # Create the base AtomicAgent configuration
        self.config = AgentConfig(
            client=client,
            model=model,
            system_prompt_generator=system_prompt,
            history=ChatHistory(),
        )
        
        # Create the AtomicAgent instance
        self.agent = AtomicAgent[SummarizerInput, PaperNote](self.config)
    
    def run(self, input_data: SummarizerInput) -> PaperNote:
        """
        Run the summarizer agent with validation and retry logic.
        
        Args:
            input_data: SummarizerInput containing parsed paper and context
            
        Returns:
            PaperNote object with structured summary
            
        Raises:
            SummarizerValidationError: If validation fails after retry
        """
        logger.info(f"Starting summarization for paper: {input_data.parsed_paper.paper_ref.title}")
        
        # Try first attempt
        try:
            result = self._attempt_summarization(input_data)
            logger.info("Summarization completed successfully on first attempt")
            return result
            
        except ValidationError as e:
            logger.warning(f"First attempt failed validation: {e}")
            logger.info("Attempting retry with corrective message")
            
            # Retry with corrective suffix
            try:
                result = self._attempt_summarization(
                    input_data, 
                    retry_suffix="Your last output was invalid JSON for PaperNote. Return strictly valid JSON per schema."
                )
                logger.info("Summarization completed successfully on retry")
                return result
                
            except ValidationError as retry_error:
                logger.error(f"Retry also failed validation: {retry_error}")
                raise SummarizerValidationError(
                    f"Failed to generate valid PaperNote after retry. "
                    f"Original error: {e}. Retry error: {retry_error}"
                )
    
    def _attempt_summarization(self, input_data: SummarizerInput, retry_suffix: str = "") -> PaperNote:
        """
        Attempt summarization with optional retry suffix.
        
        Args:
            input_data: SummarizerInput containing parsed paper and context
            retry_suffix: Optional corrective message to append
            
        Returns:
            PaperNote object
            
        Raises:
            ValidationError: If the response doesn't validate as PaperNote
        """
        # Build the prompt content
        paper = input_data.parsed_paper
        recent_notes_context = ""
        
        if input_data.recent_notes:
            recent_notes_context = f"\nRecent paper summaries for consistency:\n" + "\n".join(
                f"- {note[:200]}..." if len(note) > 200 else f"- {note}"
                for note in input_data.recent_notes[-3:]  # Last 3 for context
            )
        
        prompt_content = f"""
Paper Title: {paper.paper_ref.title}
Authors: {', '.join(paper.paper_ref.authors)}
Venue: {paper.paper_ref.venue or 'Unknown'}
Year: {paper.paper_ref.year or 'Unknown'}

Paper Content:
{paper.full_text[:8000]}  # Limit content to avoid token limits
{recent_notes_context}
{retry_suffix}
""".strip()
        
        # Use instructor client for structured output
        try:
            result = self.client.chat.completions.create(
                model=self.model,
                response_model=PaperNote,
                temperature=0.2,  # Lower temperature for higher faithfulness
                messages=[
                    {"role": "system", "content": self._build_system_message()},
                    {"role": "user", "content": prompt_content}
                ]
            )
            
            # Ensure required fields are populated from input
            result.paper_id = paper.paper_ref.id
            result.pillar_id = input_data.pillar_id
            
            return result
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            else:
                # Wrap other errors in custom exception for consistent handling
                raise SummarizerValidationError(f"Instructor completion failed: {e}")
    
    def _build_system_message(self) -> str:
        """Build the system message from SystemPromptGenerator."""
        background_text = "\n".join(f"- {bg}" for bg in system_prompt.background)
        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(system_prompt.steps))
        instructions_text = "\n".join(f"- {inst}" for inst in system_prompt.output_instructions)
        
        return f"""Background:
{background_text}

Steps:
{steps_text}

Output Instructions:
{instructions_text}

You must return a valid PaperNote JSON object with all required fields."""


# Create the ready-to-use SummarizerAgent instance
try:
    _client = _make_client()
    _settings = get_settings()
    _model = _settings.default_model
    
    SummarizerAgent = SummarizerAgentImpl(_client, _model)
    
except Exception as e:
    logger.warning(f"Failed to initialize SummarizerAgent: {e}")
    SummarizerAgent = None


def summarize(parsed_paper: ParsedPaper, pillar_id: PillarID, recent_notes: List[str] = None) -> PaperNote:
    """
    Convenience function to summarize a paper using the default SummarizerAgent.
    
    Args:
        parsed_paper: ParsedPaper object with paper content
        pillar_id: PillarID for the target learning pillar
        recent_notes: List of recent paper summaries for consistency
        
    Returns:
        PaperNote object with structured summary
        
    Raises:
        SummarizerValidationError: If summarization fails after retry
        ValueError: If SummarizerAgent is not initialized
    """
    if SummarizerAgent is None:
        raise ValueError("SummarizerAgent is not initialized. Check OPENAI_API_KEY configuration.")
    
    input_data = SummarizerInput(
        parsed_paper=parsed_paper,
        pillar_id=pillar_id,
        recent_notes=recent_notes or []
    )
    
    return SummarizerAgent.run(input_data)


# Example usage and testing
if __name__ == "__main__":
    import json
    from ..schemas import PaperRef
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example paper data
    sample_paper_ref = PaperRef(
        id="2301.12345",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        venue="NeurIPS",
        year=2017,
        url_pdf="https://arxiv.org/pdf/1706.03762.pdf"
    )
    
    sample_parsed_paper = ParsedPaper(
        paper_ref=sample_paper_ref,
        full_text="""This paper introduces the Transformer architecture, a novel neural network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. 
        
        The Transformer achieves superior performance on machine translation tasks while being more parallelizable and requiring significantly less time to train. The model relies entirely on an attention mechanism to draw global dependencies between input and output.
        
        Key contributions include: 1) A new simple network architecture based solely on attention mechanisms, 2) Superior translation quality on WMT 2014 English-German and English-French tasks, 3) Highly parallelizable training leading to faster convergence.
        
        The attention mechanism allows modeling of dependencies without regard to their distance in the input or output sequences. This makes the Transformer particularly effective for sequence-to-sequence tasks.
        
        Limitations include potential difficulty with very long sequences due to quadratic complexity of self-attention, and the need for large amounts of training data to achieve optimal performance.""",
        chunks=["Transformer attention mechanisms...", "Translation experiments...", "Architecture details..."],
        figures_count=3,
        tables_count=2,
        references=["Bahdanau et al., 2015", "Vaswani et al., 2017"]
    )
    
    try:
        # Test summarization
        note = summarize(
            parsed_paper=sample_parsed_paper,
            pillar_id=PillarID.P2,  # Models & Architectures
            recent_notes=["Previous paper on RNNs showed limited parallelization..."]
        )
        
        print("Summarization successful!")
        print(f"Problem: {note.problem}")
        print(f"Method: {note.method}")
        print(f"Findings: {note.findings}")
        print(f"Limitations: {note.limitations}")
        print(f"Key terms: {note.key_terms}")
        
    except Exception as e:
        print(f"Summarization failed: {e}")
