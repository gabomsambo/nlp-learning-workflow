"""
SynthesisAgent using Atomic Agents v2.0 + Instructor.

Synthesizes paper notes into actionable lessons following pillar configurations.
"""

import logging
from typing import List, Optional
import instructor
from openai import OpenAI
from pydantic import ValidationError

# TODO: Replace with proper atomic_agents imports when v2.0+ is available
# from atomic_agents import AtomicAgent, AgentConfig
# from atomic_agents.context import SystemPromptGenerator, ChatHistory

# Using the temporary mock classes from discovery_agent
from .discovery_agent import AtomicAgent, AgentConfig, SystemPromptGenerator, ChatHistory

from ..schemas import SynthesisInput, Lesson, PaperNote, PillarConfig
from ..config import get_settings

logger = logging.getLogger(__name__)


class SynthesisValidationError(Exception):
    """Raised when SynthesisAgent fails validation after retry."""
    pass


def _make_client() -> instructor.Instructor:
    """Create the Instructor-wrapped OpenAI client with sensible defaults."""
    settings = get_settings()
    
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    client = instructor.from_openai(
        OpenAI(
            api_key=settings.openai_api_key,
            timeout=60.0,
        )
    )
    
    return client


# Create SystemPromptGenerator with specified content
system_prompt = SystemPromptGenerator(
    background=[
        "You are an NLP learning synthesis assistant.",
        "Produce concise, actionable lessons grounded in the provided paper note and pillar config."
    ],
    steps=[
        "Create a single-sentence tl_dr.",
        "Generate 3–5 specific takeaways.",
        "Propose 2–3 practical practice_ideas.",
        "Add 0–3 connections to related work (IDs or short titles if provided).",
        "Respect the pillar goal/focus areas for emphasis."
    ],
    output_instructions=[
        "Return valid Lesson JSON only."
    ]
)


class SynthesisAgentImpl:
    """Implementation class for the Synthesis Agent with retry logic."""
    
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
        self.agent = AtomicAgent[SynthesisInput, Lesson](self.config)
    
    def run(self, input_data: SynthesisInput) -> Lesson:
        """
        Run the synthesis agent with validation and retry logic.
        
        Args:
            input_data: SynthesisInput containing paper note and pillar config
            
        Returns:
            Lesson object with synthesized learning content
            
        Raises:
            SynthesisValidationError: If validation fails after retry
        """
        logger.info(f"Starting synthesis for paper: {input_data.paper_note.paper_id}")
        
        # Try first attempt
        try:
            result = self._attempt_synthesis(input_data)
            logger.info("Synthesis completed successfully on first attempt")
            return result
            
        except ValidationError as e:
            logger.warning(f"First attempt failed validation: {e}")
            logger.info("Attempting retry with corrective message")
            
            # Retry with corrective suffix
            try:
                result = self._attempt_synthesis(
                    input_data, 
                    retry_suffix="Your last output was invalid JSON for Lesson. Return strictly valid JSON per schema."
                )
                logger.info("Synthesis completed successfully on retry")
                return result
                
            except ValidationError as retry_error:
                logger.error(f"Retry also failed validation: {retry_error}")
                raise SynthesisValidationError(
                    f"Failed to generate valid Lesson after retry. "
                    f"Original error: {e}. Retry error: {retry_error}"
                )
    
    def _attempt_synthesis(self, input_data: SynthesisInput, retry_suffix: str = "") -> Lesson:
        """
        Attempt synthesis with optional retry suffix.
        
        Args:
            input_data: SynthesisInput containing paper note and pillar config
            retry_suffix: Optional corrective message to append
            
        Returns:
            Lesson object
            
        Raises:
            ValidationError: If the response doesn't validate as Lesson
        """
        # Build the prompt content
        paper_note = input_data.paper_note
        pillar_config = input_data.pillar_config
        related_lessons = input_data.related_lessons
        
        # Build related lessons context
        related_context = ""
        if related_lessons:
            related_context = "\nRelated lessons for connections:\n" + "\n".join(
                f"- {lesson.paper_id}: {lesson.tl_dr}"
                for lesson in related_lessons[-5:]  # Last 5 for context
            )
        
        prompt_content = f"""
Paper Note:
Title: {paper_note.paper_id}
Problem: {paper_note.problem}
Method: {paper_note.method}
Key Findings: {', '.join(paper_note.findings)}
Limitations: {', '.join(paper_note.limitations)}
Future Work: {', '.join(paper_note.future_work)}
Key Terms: {', '.join(paper_note.key_terms)}

Pillar Configuration:
Name: {pillar_config.name}
Goal: {pillar_config.goal}
Focus Areas: {', '.join(pillar_config.focus_areas)}
{related_context}

Create a lesson that:
- Has a single-sentence tl_dr capturing the essence
- Contains 3-5 specific takeaways aligned with the pillar focus
- Provides 2-3 practical practice_ideas for applying the concepts
- Includes 0-3 connections to related work (use related lesson paper_ids or short titles)
- Emphasizes aspects relevant to the pillar goal

{retry_suffix}
""".strip()
        
        # Use instructor client for structured output
        try:
            result = self.client.chat.completions.create(
                model=self.model,
                response_model=Lesson,
                temperature=0.2,  # Low temperature for deterministic output
                messages=[
                    {"role": "system", "content": self._build_system_message()},
                    {"role": "user", "content": prompt_content}
                ]
            )
            
            # Ensure required fields are populated from input
            result.paper_id = paper_note.paper_id
            result.pillar_id = paper_note.pillar_id
            
            return result
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            else:
                # Wrap other errors in custom exception for consistent handling
                raise SynthesisValidationError(f"Instructor completion failed: {e}")
    
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

You must return a valid Lesson JSON object with all required fields."""


# Create the ready-to-use SynthesisAgent instance
try:
    _client = _make_client()
    _settings = get_settings()
    _model = _settings.default_model
    
    SynthesisAgent = SynthesisAgentImpl(_client, _model)
    
except Exception as e:
    logger.warning(f"Failed to initialize SynthesisAgent: {e}")
    SynthesisAgent = None


def synthesize(
    paper_note: PaperNote, 
    pillar_config: PillarConfig, 
    related_lessons: Optional[List[Lesson]] = None
) -> Lesson:
    """
    Convenience function to synthesize a lesson using the default SynthesisAgent.
    
    Args:
        paper_note: PaperNote object with structured paper content
        pillar_config: PillarConfig defining the learning pillar
        related_lessons: Optional list of related lessons for context
        
    Returns:
        Lesson object with synthesized learning content
        
    Raises:
        SynthesisValidationError: If synthesis fails after retry
        ValueError: If SynthesisAgent is not initialized
    """
    if SynthesisAgent is None:
        raise ValueError("SynthesisAgent is not initialized. Check OPENAI_API_KEY configuration.")
    
    input_data = SynthesisInput(
        paper_note=paper_note,
        pillar_config=pillar_config,
        related_lessons=related_lessons or []
    )
    
    return SynthesisAgent.run(input_data)


# Example usage and testing
if __name__ == "__main__":
    import json
    from ..schemas import PaperRef, PillarID
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example data structures would go here for testing
    print("SynthesisAgent module loaded successfully")
