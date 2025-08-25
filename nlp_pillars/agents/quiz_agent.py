"""
Quiz Agent - Fixed version with working implementation.
"""

from typing import List
from ..schemas import QuizCard, QuizGeneratorInput


class QuizAgent:
    """Agent that generates quiz cards for spaced repetition learning."""
    
    @classmethod
    def run(cls, input_data: QuizGeneratorInput) -> List[QuizCard]:
        """Generate quiz cards from the paper note.
        
        This is a simplified implementation that creates quiz questions
        without requiring the full AtomicAgent infrastructure.
        """
        note = input_data.paper_note
        cards = []
        
        # Generate quiz cards based on the paper content
        cards.append(QuizCard(
            paper_id=note.paper_id,
            pillar_id=note.pillar_id,
            question="What is the main problem addressed in this paper?",
            answer=note.problem,
            difficulty=1,
            tags=["problem", "overview"]
        ))
        
        cards.append(QuizCard(
            paper_id=note.paper_id,
            pillar_id=note.pillar_id,
            question="What methodology was used in this research?",
            answer=note.method,
            difficulty=2,
            tags=["methodology", "technical"]
        ))
        
        cards.append(QuizCard(
            paper_id=note.paper_id,
            pillar_id=note.pillar_id,
            question="What are the key findings of this paper?",
            answer="; ".join(note.findings) if isinstance(note.findings, list) else str(note.findings),
            difficulty=2,
            tags=["findings", "results"]
        ))
        
        if note.limitations:
            cards.append(QuizCard(
                paper_id=note.paper_id,
                pillar_id=note.pillar_id,
                question="What are the limitations of this approach?",
                answer="; ".join(note.limitations) if isinstance(note.limitations, list) else str(note.limitations),
                difficulty=3,
                tags=["limitations", "critical-thinking"]
            ))
        
        if note.future_work:
            cards.append(QuizCard(
                paper_id=note.paper_id,
                pillar_id=note.pillar_id,
                question="What future work is suggested?",
                answer="; ".join(note.future_work) if isinstance(note.future_work, list) else str(note.future_work),
                difficulty=2,
                tags=["future-work", "research-directions"]
            ))
        
        return cards[:input_data.num_questions]  # Return requested number of questions
