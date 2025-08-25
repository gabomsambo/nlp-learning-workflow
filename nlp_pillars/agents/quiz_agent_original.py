"""
QuizAgent using Atomic Agents v2.0 + Instructor.

Generates quiz questions from paper notes with proper difficulty and type distribution.
"""

import logging
from typing import List, Dict
import instructor
from openai import OpenAI
from pydantic import ValidationError

# TODO: Replace with proper atomic_agents imports when v2.0+ is available
# from atomic_agents import AtomicAgent, AgentConfig
# from atomic_agents.context import SystemPromptGenerator, ChatHistory

# Using the temporary mock classes from discovery_agent
from .discovery_agent import AtomicAgent, AgentConfig, SystemPromptGenerator, ChatHistory

from ..schemas import QuizGeneratorInput, QuizCard, PaperNote, DifficultyLevel, QuestionType
from ..config import get_settings

logger = logging.getLogger(__name__)


class QuizValidationError(Exception):
    """Raised when QuizAgent fails validation after retry."""
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
        "You create high-quality quiz questions from research notes.",
        "Generate clear, unambiguous questions with single best answers."
    ],
    steps=[
        "Use the paper_note to generate questions.",
        "Honor difficulty_mix counts.",
        "Distribute types across FACTUAL, CONCEPTUAL, APPLICATION.",
        "Answers should be concise (1–3 sentences or a short list)."
    ],
    output_instructions=[
        "Return a JSON array of QuizCard objects only."
    ]
)


class QuizAgentImpl:
    """Implementation class for the Quiz Agent with retry logic."""
    
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
        self.agent = AtomicAgent[QuizGeneratorInput, List[QuizCard]](self.config)
    
    def run(self, input_data: QuizGeneratorInput) -> List[QuizCard]:
        """
        Run the quiz agent with validation and retry logic.
        
        Args:
            input_data: QuizGeneratorInput containing paper note and quiz parameters
            
        Returns:
            List of QuizCard objects with specified difficulty and type distribution
            
        Raises:
            QuizValidationError: If validation fails after retry
        """
        logger.info(f"Starting quiz generation for paper: {input_data.paper_note.paper_id}")
        logger.info(f"Requested {input_data.num_questions} questions with mix: {input_data.difficulty_mix}")
        
        # Try first attempt
        try:
            result = self._attempt_quiz_generation(input_data)
            logger.info(f"Quiz generation completed successfully on first attempt: {len(result)} questions")
            return result
            
        except ValidationError as e:
            logger.warning(f"First attempt failed validation: {e}")
            logger.info("Attempting retry with corrective message")
            
            # Retry with corrective suffix
            try:
                result = self._attempt_quiz_generation(
                    input_data, 
                    retry_suffix="Your last output was invalid JSON for QuizCard array. Return strictly valid JSON array per schema."
                )
                logger.info(f"Quiz generation completed successfully on retry: {len(result)} questions")
                return result
                
            except ValidationError as retry_error:
                logger.error(f"Retry also failed validation: {retry_error}")
                raise QuizValidationError(
                    f"Failed to generate valid QuizCard array after retry. "
                    f"Original error: {e}. Retry error: {retry_error}"
                )
    
    def _attempt_quiz_generation(self, input_data: QuizGeneratorInput, retry_suffix: str = "") -> List[QuizCard]:
        """
        Attempt quiz generation with optional retry suffix.
        
        Args:
            input_data: QuizGeneratorInput containing paper note and quiz parameters
            retry_suffix: Optional corrective message to append
            
        Returns:
            List of QuizCard objects
            
        Raises:
            ValidationError: If the response doesn't validate as List[QuizCard]
        """
        # Build the prompt content
        paper_note = input_data.paper_note
        num_questions = input_data.num_questions
        difficulty_mix = input_data.difficulty_mix
        
        # Convert difficulty_mix to detailed requirements
        difficulty_requirements = []
        for level, count in difficulty_mix.items():
            if count > 0:
                difficulty_requirements.append(f"{count} {level} questions")
        
        # Build type distribution guidance
        question_types = ["FACTUAL", "CONCEPTUAL", "APPLICATION"]
        type_guidance = "Distribute question types across FACTUAL (recall), CONCEPTUAL (understanding), and APPLICATION (apply knowledge)"
        
        prompt_content = f"""
Paper Note to Create Quiz From:
Paper ID: {paper_note.paper_id}
Problem: {paper_note.problem}
Method: {paper_note.method}
Key Findings: {', '.join(paper_note.findings)}
Limitations: {', '.join(paper_note.limitations)}
Key Terms: {', '.join(paper_note.key_terms)}

Quiz Requirements:
- Generate exactly {num_questions} questions
- Difficulty distribution: {', '.join(difficulty_requirements)}
- {type_guidance}

Question Type Guidelines:
- FACTUAL: Test recall of specific facts, definitions, or data from the paper
- CONCEPTUAL: Test understanding of concepts, relationships, or explanations
- APPLICATION: Test ability to apply methods or findings to new scenarios

Answer Guidelines:
- Keep answers concise (1-3 sentences or short lists)
- Ensure answers are unambiguous and directly supported by the paper content
- Avoid questions that require external knowledge not in the paper

{retry_suffix}
""".strip()
        
        # Use instructor client for structured output
        try:
            result = self.client.chat.completions.create(
                model=self.model,
                response_model=List[QuizCard],
                temperature=0.2,  # Low temperature for deterministic output
                messages=[
                    {"role": "system", "content": self._build_system_message()},
                    {"role": "user", "content": prompt_content}
                ]
            )
            
            # Validate and post-process the result
            validated_result = self._validate_and_fix_quiz(result, input_data)
            
            return validated_result
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            else:
                # Wrap other errors in custom exception for consistent handling
                raise QuizValidationError(f"Instructor completion failed: {e}")
    
    def _validate_and_fix_quiz(self, quiz_cards: List[QuizCard], input_data: QuizGeneratorInput) -> List[QuizCard]:
        """
        Validate and fix quiz cards to meet requirements.
        
        Args:
            quiz_cards: Raw quiz cards from LLM
            input_data: Original input requirements
            
        Returns:
            Validated and fixed quiz cards
        """
        paper_note = input_data.paper_note
        num_questions = input_data.num_questions
        difficulty_mix = input_data.difficulty_mix
        
        # Ensure we have exactly num_questions
        if len(quiz_cards) > num_questions:
            quiz_cards = quiz_cards[:num_questions]
        elif len(quiz_cards) < num_questions:
            # If we have too few, this is a validation error
            raise ValidationError(f"Expected {num_questions} questions, got {len(quiz_cards)}")
        
        # Set required fields and defaults for all cards
        for i, card in enumerate(quiz_cards):
            card.paper_id = paper_note.paper_id
            card.pillar_id = paper_note.pillar_id
            card.id = f"{paper_note.paper_id}_q{i+1}"
            
            # Ensure difficulty is set to a valid DifficultyLevel
            if hasattr(card.difficulty, 'value'):
                # It's already a DifficultyLevel enum
                pass
            elif isinstance(card.difficulty, str):
                # Convert string to enum
                difficulty_map = {
                    'easy': DifficultyLevel.EASY,
                    'medium': DifficultyLevel.MEDIUM,
                    'hard': DifficultyLevel.HARD
                }
                card.difficulty = difficulty_map.get(card.difficulty.lower(), DifficultyLevel.MEDIUM)
            elif isinstance(card.difficulty, int):
                # Convert int to enum
                if card.difficulty == 1:
                    card.difficulty = DifficultyLevel.EASY
                elif card.difficulty == 3:
                    card.difficulty = DifficultyLevel.HARD
                else:
                    card.difficulty = DifficultyLevel.MEDIUM
            
            # Ensure question_type is set to a valid QuestionType
            if hasattr(card.question_type, 'value'):
                # It's already a QuestionType enum
                pass
            elif isinstance(card.question_type, str):
                # Convert string to enum
                type_map = {
                    'factual': QuestionType.FACTUAL,
                    'conceptual': QuestionType.CONCEPTUAL,
                    'application': QuestionType.APPLICATION
                }
                card.question_type = type_map.get(card.question_type.lower(), QuestionType.FACTUAL)
        
        # Try to honor difficulty_mix (best effort)
        self._adjust_difficulty_distribution(quiz_cards, difficulty_mix)
        
        # Ensure type diversity (best effort)
        self._ensure_type_diversity(quiz_cards)
        
        return quiz_cards
    
    def _adjust_difficulty_distribution(self, quiz_cards: List[QuizCard], difficulty_mix: Dict[str, int]):
        """Adjust difficulty distribution to match requested mix (best effort)."""
        # Map difficulty names to enums
        difficulty_map = {
            'easy': DifficultyLevel.EASY,
            'medium': DifficultyLevel.MEDIUM,
            'hard': DifficultyLevel.HARD
        }
        
        # Calculate target counts
        targets = {}
        for level_name, count in difficulty_mix.items():
            if level_name in difficulty_map:
                targets[difficulty_map[level_name]] = count
        
        # Assign difficulties to match targets (best effort)
        assigned = 0
        for difficulty_level, target_count in targets.items():
            for _ in range(min(target_count, len(quiz_cards) - assigned)):
                if assigned < len(quiz_cards):
                    quiz_cards[assigned].difficulty = difficulty_level
                    assigned += 1
        
        # Fill remaining with MEDIUM if any
        while assigned < len(quiz_cards):
            quiz_cards[assigned].difficulty = DifficultyLevel.MEDIUM
            assigned += 1
    
    def _ensure_type_diversity(self, quiz_cards: List[QuizCard]):
        """Ensure question type diversity across the quiz."""
        if len(quiz_cards) < 3:
            return  # Too few questions to ensure diversity
        
        types = [QuestionType.FACTUAL, QuestionType.CONCEPTUAL, QuestionType.APPLICATION]
        
        # Assign types to ensure at least 2 different types
        for i, card in enumerate(quiz_cards):
            card.question_type = types[i % len(types)]
    
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

You must return a valid JSON array of QuizCard objects with all required fields."""


# Create the ready-to-use QuizAgent instance
try:
    _client = _make_client()
    _settings = get_settings()
    _model = _settings.default_model
    
    QuizAgent = QuizAgentImpl(_client, _model)
    
except Exception as e:
    logger.warning(f"Failed to initialize QuizAgent: {e}")
    QuizAgent = None


def generate_quiz(
    paper_note: PaperNote,
    num_questions: int = 5,
    difficulty_mix: Dict[str, int] = None
) -> List[QuizCard]:
    """
    Convenience function to generate a quiz using the default QuizAgent.
    
    Args:
        paper_note: PaperNote object with structured paper content
        num_questions: Number of questions to generate (1-10)
        difficulty_mix: Dictionary specifying difficulty distribution
        
    Returns:
        List of QuizCard objects
        
    Raises:
        QuizValidationError: If quiz generation fails after retry
        ValueError: If QuizAgent is not initialized
    """
    if QuizAgent is None:
        raise ValueError("QuizAgent is not initialized. Check OPENAI_API_KEY configuration.")
    
    if difficulty_mix is None:
        difficulty_mix = {"easy": 2, "medium": 2, "hard": 1}
    
    input_data = QuizGeneratorInput(
        paper_note=paper_note,
        num_questions=num_questions,
        difficulty_mix=difficulty_mix
    )
    
    return QuizAgent.run(input_data)


# Example usage and testing
if __name__ == "__main__":
    import json
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example data structures would go here for testing
    print("QuizAgent module loaded successfully")
