"""
Supabase DAO layer for NLP Learning Workflow.

Provides data access operations with strict pillar isolation.
All operations require pillar_id filtering to ensure data separation.
"""

import logging
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from supabase import create_client, Client

from .schemas import PaperRef, PaperNote, Lesson, QuizCard, PillarID

logger = logging.getLogger(__name__)

# Module-level client singleton for testing
_client: Optional[Client] = None


def get_client() -> Client:
    """
    Get or create the Supabase client singleton.
    
    Returns:
        Supabase client instance
        
    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_KEY environment variables are missing
    """
    global _client
    
    if _client is not None:
        return _client
    
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    
    if not url:
        raise ValueError("SUPABASE_URL environment variable is required")
    if not key:
        raise ValueError("SUPABASE_KEY environment variable is required")
    
    _client = create_client(url, key)
    return _client


def set_client(client: Client) -> None:
    """Set the client singleton (for testing)."""
    global _client
    _client = client


# =====================================
# Serialization Helpers
# =====================================

def _paper_ref_to_dict(pillar_id: PillarID, paper: PaperRef) -> Dict[str, Any]:
    """Convert PaperRef to papers table row."""
    data = {
        'id': paper.id,
        'pillar_id': pillar_id.value,
        'title': paper.title,
        'authors': paper.authors,  # Already a list, goes to JSONB
        'venue': paper.venue,
        'year': paper.year,
        'url_pdf': paper.url_pdf,
        'abstract': paper.abstract,
        'citation_count': paper.citation_count or 0
    }
    # Remove None values
    return {k: v for k, v in data.items() if v is not None}


def _dict_to_paper_ref(row: Dict[str, Any]) -> PaperRef:
    """Convert papers table row to PaperRef."""
    return PaperRef(
        id=row['id'],
        title=row['title'],
        authors=row.get('authors', []),
        venue=row.get('venue'),
        year=row.get('year'),
        url_pdf=row.get('url_pdf'),
        abstract=row.get('abstract'),
        citation_count=row.get('citation_count', 0)
    )


def _paper_note_to_dict(note: PaperNote) -> Dict[str, Any]:
    """Convert PaperNote to notes table row."""
    return {
        'paper_id': note.paper_id,
        'pillar_id': note.pillar_id.value,
        'problem': note.problem,
        'method': note.method,
        'findings': note.findings,
        'limitations': note.limitations,
        'future_work': note.future_work,
        'key_terms': note.key_terms,
        'related_papers': note.related_papers,
        'confidence_score': note.confidence_score
    }


def _dict_to_paper_note(row: Dict[str, Any]) -> PaperNote:
    """Convert notes table row to PaperNote."""
    return PaperNote(
        paper_id=row['paper_id'],
        pillar_id=PillarID(row['pillar_id']),
        problem=row['problem'],
        method=row['method'],
        findings=row.get('findings', []),
        limitations=row.get('limitations', []),
        future_work=row.get('future_work', []),
        key_terms=row.get('key_terms', []),
        related_papers=row.get('related_papers', []),
        confidence_score=row.get('confidence_score', 0.8)
    )


def _lesson_to_dict(lesson: Lesson) -> Dict[str, Any]:
    """Convert Lesson to lessons table row."""
    return {
        'paper_id': lesson.paper_id,
        'pillar_id': lesson.pillar_id.value,
        'tl_dr': lesson.tl_dr,
        'takeaways': lesson.takeaways,
        'practice_ideas': lesson.practice_ideas,
        'connections': lesson.connections,
        'difficulty': lesson.difficulty.value,
        'estimated_time': lesson.estimated_time
    }


def _dict_to_lesson(row: Dict[str, Any]) -> Lesson:
    """Convert lessons table row to Lesson."""
    from .schemas import DifficultyLevel
    return Lesson(
        paper_id=row['paper_id'],
        pillar_id=PillarID(row['pillar_id']),
        tl_dr=row['tl_dr'],
        takeaways=row.get('takeaways', []),
        practice_ideas=row.get('practice_ideas', []),
        connections=row.get('connections', []),
        difficulty=DifficultyLevel(row.get('difficulty', 2)),
        estimated_time=row.get('estimated_time', 10)
    )


def _quiz_card_to_dict(card: QuizCard) -> Dict[str, Any]:
    """Convert QuizCard to quiz_cards table row."""
    return {
        'paper_id': card.paper_id,
        'pillar_id': card.pillar_id.value,
        'question': card.question,
        'answer': card.answer,
        'difficulty': card.difficulty.value,
        'question_type': card.question_type.value,
        'interval': card.interval,
        'repetitions': card.repetitions,
        'ease_factor': card.ease_factor,
        'due_date': card.due_date.isoformat() if card.due_date else None,
        'last_reviewed': card.last_reviewed.isoformat() if card.last_reviewed else None
    }


def _dict_to_quiz_card(row: Dict[str, Any]) -> QuizCard:
    """Convert quiz_cards table row to QuizCard."""
    from .schemas import DifficultyLevel, QuestionType
    
    # Parse datetime strings
    due_date = None
    if row.get('due_date'):
        due_date = datetime.fromisoformat(row['due_date'].replace('Z', '+00:00'))
    
    last_reviewed = None
    if row.get('last_reviewed'):
        last_reviewed = datetime.fromisoformat(row['last_reviewed'].replace('Z', '+00:00'))
    
    return QuizCard(
        id=row.get('id'),
        paper_id=row['paper_id'],
        pillar_id=PillarID(row['pillar_id']),
        question=row['question'],
        answer=row['answer'],
        difficulty=DifficultyLevel(row.get('difficulty', 2)),
        question_type=QuestionType(row.get('question_type', 'factual')),
        interval=row.get('interval', 1),
        repetitions=row.get('repetitions', 0),
        ease_factor=row.get('ease_factor', 2.5),
        due_date=due_date,
        last_reviewed=last_reviewed
    )


# =====================================
# Public DAO Functions
# =====================================

def upsert_paper(pillar_id: PillarID, paper: PaperRef) -> None:
    """
    Upsert a paper into the papers table with pillar isolation.
    
    Args:
        pillar_id: Target pillar for the paper
        paper: Paper metadata to upsert
        
    Raises:
        ValueError: If pillar_id is None or paper is invalid
    """
    if not pillar_id:
        raise ValueError("pillar_id is required for paper upsert")
    
    if not paper.id:
        raise ValueError("paper.id is required for upsert")
    
    logger.info(f"Upserting paper {paper.id} to pillar {pillar_id.value}")
    
    try:
        client = get_client()
        row_data = _paper_ref_to_dict(pillar_id, paper)
        
        # Upsert based on id, but maintain pillar_id constraint
        result = client.table('papers').upsert(
            row_data,
            on_conflict='id'
        ).execute()
        
        logger.info(f"Successfully upserted paper {paper.id}")
        
    except Exception as e:
        logger.error(f"Failed to upsert paper {paper.id}: {e}")
        raise ValueError(f"Failed to upsert paper {paper.id}: {e}")


def mark_processed(pillar_id: PillarID, paper_id: str) -> None:
    """
    Mark a paper as processed within the specified pillar.
    
    Args:
        pillar_id: Pillar containing the paper
        paper_id: ID of the paper to mark as processed
        
    Raises:
        ValueError: If pillar_id or paper_id is missing
    """
    if not pillar_id:
        raise ValueError("pillar_id is required")
    
    if not paper_id:
        raise ValueError("paper_id is required")
    
    logger.info(f"Marking paper {paper_id} as processed in pillar {pillar_id.value}")
    
    try:
        client = get_client()
        now = datetime.now(timezone.utc).isoformat()
        
        result = client.table('papers').update({
            'processed': True,
            'processed_at': now
        }).eq('id', paper_id).eq('pillar_id', pillar_id.value).execute()
        
        if not result.data:
            logger.warning(f"No paper found with id {paper_id} in pillar {pillar_id.value}")
        else:
            logger.info(f"Successfully marked paper {paper_id} as processed")
            
    except Exception as e:
        logger.error(f"Failed to mark paper {paper_id} as processed: {e}")
        raise ValueError(f"Failed to mark paper {paper_id} as processed: {e}")


def insert_note(note: PaperNote) -> None:
    """
    Insert a paper note with pillar isolation.
    
    Args:
        note: Paper note to insert
        
    Raises:
        ValueError: If note is invalid or missing required fields
    """
    if not note.pillar_id:
        raise ValueError("note.pillar_id is required")
    
    if not note.paper_id:
        raise ValueError("note.paper_id is required")
    
    logger.info(f"Inserting note for paper {note.paper_id} in pillar {note.pillar_id.value}")
    
    try:
        client = get_client()
        row_data = _paper_note_to_dict(note)
        
        result = client.table('notes').insert(row_data).execute()
        
        logger.info(f"Successfully inserted note for paper {note.paper_id}")
        
    except Exception as e:
        logger.error(f"Failed to insert note for paper {note.paper_id}: {e}")
        raise ValueError(f"Failed to insert note for paper {note.paper_id}: {e}")


def insert_lesson(lesson: Lesson) -> None:
    """
    Insert a lesson with pillar isolation.
    
    Args:
        lesson: Lesson to insert
        
    Raises:
        ValueError: If lesson is invalid or missing required fields
    """
    if not lesson.pillar_id:
        raise ValueError("lesson.pillar_id is required")
    
    if not lesson.paper_id:
        raise ValueError("lesson.paper_id is required")
    
    logger.info(f"Inserting lesson for paper {lesson.paper_id} in pillar {lesson.pillar_id.value}")
    
    try:
        client = get_client()
        row_data = _lesson_to_dict(lesson)
        
        result = client.table('lessons').insert(row_data).execute()
        
        logger.info(f"Successfully inserted lesson for paper {lesson.paper_id}")
        
    except Exception as e:
        logger.error(f"Failed to insert lesson for paper {lesson.paper_id}: {e}")
        raise ValueError(f"Failed to insert lesson for paper {lesson.paper_id}: {e}")


def insert_quiz_cards(cards: List[QuizCard]) -> None:
    """
    Insert quiz cards with pillar isolation using bulk insert.
    
    Args:
        cards: List of quiz cards to insert
        
    Raises:
        ValueError: If cards are invalid or missing required fields
    """
    if not cards:
        return
    
    # Validate all cards have required fields
    for i, card in enumerate(cards):
        if not card.pillar_id:
            raise ValueError(f"cards[{i}].pillar_id is required")
        if not card.paper_id:
            raise ValueError(f"cards[{i}].paper_id is required")
    
    pillar_id = cards[0].pillar_id
    logger.info(f"Inserting {len(cards)} quiz cards for pillar {pillar_id.value}")
    
    try:
        client = get_client()
        rows_data = [_quiz_card_to_dict(card) for card in cards]
        
        # Bulk insert
        result = client.table('quiz_cards').insert(rows_data).execute()
        
        logger.info(f"Successfully inserted {len(cards)} quiz cards")
        
    except Exception as e:
        logger.error(f"Failed to insert quiz cards: {e}")
        raise ValueError(f"Failed to insert quiz cards: {e}")


def get_recent_notes(pillar_id: PillarID, limit: int = 5) -> List[PaperNote]:
    """
    Get recent notes for a pillar.
    
    Args:
        pillar_id: Pillar to get notes from
        limit: Maximum number of notes to return
        
    Returns:
        List of recent PaperNote objects
        
    Raises:
        ValueError: If pillar_id is missing
    """
    if not pillar_id:
        raise ValueError("pillar_id is required")
    
    logger.info(f"Getting {limit} recent notes for pillar {pillar_id.value}")
    
    try:
        client = get_client()
        
        result = client.table('notes').select('*').eq(
            'pillar_id', pillar_id.value
        ).order('created_at', desc=True).limit(limit).execute()
        
        notes = [_dict_to_paper_note(row) for row in result.data]
        
        logger.info(f"Retrieved {len(notes)} recent notes for pillar {pillar_id.value}")
        return notes
        
    except Exception as e:
        logger.error(f"Failed to get recent notes for pillar {pillar_id.value}: {e}")
        raise ValueError(f"Failed to get recent notes for pillar {pillar_id.value}: {e}")


def queue_add_candidates(pillar_id: PillarID, papers: List[PaperRef]) -> int:
    """
    Add paper candidates to the queue, deduplicating against existing papers and queue.
    
    Args:
        pillar_id: Target pillar for the papers
        papers: List of paper candidates to add
        
    Returns:
        Number of papers actually inserted (after deduplication)
        
    Raises:
        ValueError: If pillar_id is missing
    """
    if not pillar_id:
        raise ValueError("pillar_id is required")
    
    if not papers:
        return 0
    
    logger.info(f"Adding {len(papers)} paper candidates to queue for pillar {pillar_id.value}")
    
    try:
        client = get_client()
        
        # Get existing paper IDs in this pillar
        existing_papers = client.table('papers').select('id').eq(
            'pillar_id', pillar_id.value
        ).execute()
        existing_paper_ids = {row['id'] for row in existing_papers.data}
        
        # Get existing queue IDs in this pillar
        existing_queue = client.table('paper_queue').select('paper_id').eq(
            'pillar_id', pillar_id.value
        ).execute()
        existing_queue_ids = {row['paper_id'] for row in existing_queue.data}
        
        # Filter out duplicates
        new_papers = []
        for paper in papers:
            if paper.id not in existing_paper_ids and paper.id not in existing_queue_ids:
                new_papers.append(paper)
        
        if not new_papers:
            logger.info("No new papers to add after deduplication")
            return 0
        
        # Insert new papers into queue
        queue_rows = []
        for paper in new_papers:
            queue_rows.append({
                'pillar_id': pillar_id.value,
                'paper_id': paper.id,
                'title': paper.title,
                'priority': 5,  # Default priority
                'source': 'unknown',
                'processed': False
            })
        
        result = client.table('paper_queue').insert(queue_rows).execute()
        
        logger.info(f"Successfully added {len(new_papers)} papers to queue for pillar {pillar_id.value}")
        return len(new_papers)
        
    except Exception as e:
        logger.error(f"Failed to add candidates to queue for pillar {pillar_id.value}: {e}")
        raise ValueError(f"Failed to add candidates to queue for pillar {pillar_id.value}: {e}")


def queue_pop_next(pillar_id: PillarID, limit: int = 1) -> List[PaperRef]:
    """
    Pop the next papers from the queue for processing.
    
    Args:
        pillar_id: Pillar to pop papers from
        limit: Maximum number of papers to pop
        
    Returns:
        List of PaperRef objects for processing
        
    Raises:
        ValueError: If pillar_id is missing
    """
    if not pillar_id:
        raise ValueError("pillar_id is required")
    
    logger.info(f"Popping {limit} papers from queue for pillar {pillar_id.value}")
    
    try:
        client = get_client()
        
        # Get next papers from queue (highest priority, then newest)
        queue_result = client.table('paper_queue').select('*').eq(
            'pillar_id', pillar_id.value
        ).eq('processed', False).order(
            'priority', desc=True
        ).order('added_at', desc=True).limit(limit).execute()
        
        if not queue_result.data:
            logger.info(f"No papers in queue for pillar {pillar_id.value}")
            return []
        
        queue_rows = queue_result.data
        paper_ids = [row['paper_id'] for row in queue_rows]
        queue_ids = [row['id'] for row in queue_rows]
        
        # Try to get full paper data from papers table
        papers_result = client.table('papers').select('*').in_('id', paper_ids).execute()
        papers_by_id = {row['id']: row for row in papers_result.data}
        
        # Build PaperRef list (with fallback for missing papers)
        paper_refs = []
        for queue_row in queue_rows:
            paper_id = queue_row['paper_id']
            if paper_id in papers_by_id:
                # Use full paper data
                paper_refs.append(_dict_to_paper_ref(papers_by_id[paper_id]))
            else:
                # Fallback to minimal PaperRef from queue
                paper_refs.append(PaperRef(
                    id=paper_id,
                    title=queue_row['title'],
                    authors=[]
                ))
        
        # Mark queue items as processed
        client.table('paper_queue').update({
            'processed': True
        }).in_('id', queue_ids).execute()
        
        logger.info(f"Successfully popped {len(paper_refs)} papers from queue for pillar {pillar_id.value}")
        return paper_refs
        
    except Exception as e:
        logger.error(f"Failed to pop papers from queue for pillar {pillar_id.value}: {e}")
        raise ValueError(f"Failed to pop papers from queue for pillar {pillar_id.value}: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example operations would go here for testing
    print("Database DAO module loaded successfully")
