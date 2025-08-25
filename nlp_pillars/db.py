"""
Fixed Supabase DAO layer for NLP Learning Workflow.
Modified to work with vanilla PostgREST (without /rest/v1/ prefix).
"""

import logging
import os
import httpx
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from .schemas import PaperRef, PaperNote, Lesson, QuizCard, PillarID

logger = logging.getLogger(__name__)


class PostgRESTClient:
    """Simple PostgREST client that works with vanilla PostgREST."""
    
    def __init__(self, url: str, key: str):
        self.base_url = url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
            'Prefer': 'return=representation'
        }
        self.client = httpx.Client(headers=self.headers, timeout=30.0)
    
    def table(self, table_name: str):
        """Return a table query builder."""
        return TableQuery(self.client, self.base_url, table_name)


class TableQuery:
    """Simple query builder for PostgREST."""
    
    def __init__(self, client: httpx.Client, base_url: str, table_name: str):
        self.client = client
        self.url = f"{base_url}/{table_name}"
        self.params = {}
    
    def select(self, columns: str = '*'):
        """Select columns."""
        self.params['select'] = columns
        return self
    
    def eq(self, column: str, value: Any):
        """Filter by equality."""
        self.params[column] = f'eq.{value}'
        return self
    
    def order(self, column: str, desc: bool = False):
        """Order results."""
        direction = '.desc' if desc else ''
        self.params['order'] = f'{column}{direction}'
        return self
    
    def limit(self, count: int):
        """Limit results."""
        self.params['limit'] = str(count)
        return self
    
    def insert(self, data: Dict[str, Any]):
        """Insert data."""
        logger.debug(f"Inserting data to {self.url}: {json.dumps(data, indent=2)}")
        try:
            response = self.client.post(self.url, json=data)
            if response.status_code >= 400:
                logger.error(f"Insert failed with status {response.status_code}: {response.text}")
                logger.error(f"Request payload was: {json.dumps(data, indent=2)}")
            response.raise_for_status()
            return {'data': response.json() if response.text else None, 'error': None}
        except httpx.HTTPStatusError as e:
            # Handle HTTP 409 Conflict gracefully
            if e.response.status_code == 409:
                logger.info(f"Conflict detected (409): Resource already exists")
                return {'data': None, 'error': {'code': 409, 'message': 'Resource already exists', 'details': e.response.text}}
            # Re-raise other HTTP errors
            error_data = {}
            try:
                error_data = e.response.json()
            except:
                error_data = {'message': str(e), 'status_code': e.response.status_code}
            return {'data': None, 'error': error_data}
    
    def update(self, data: Dict[str, Any]):
        """Update data."""
        response = self.client.patch(self.url, json=data, params=self.params)
        response.raise_for_status()
        return {'data': response.json() if response.text else None, 'error': None}
    
    def execute(self):
        """Execute the query."""
        try:
            response = self.client.get(self.url, params=self.params)
            response.raise_for_status()
            return {'data': response.json(), 'error': None}
        except httpx.HTTPStatusError as e:
            error_data = {}
            try:
                error_data = e.response.json()
            except:
                error_data = {'message': str(e)}
            return {'data': None, 'error': error_data}
        except Exception as e:
            return {'data': None, 'error': {'message': str(e)}}


# Module-level client singleton
_client: Optional[PostgRESTClient] = None


def get_client() -> PostgRESTClient:
    """Get or create the PostgREST client singleton."""
    global _client
    
    if _client is not None:
        return _client
    
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    
    if not url:
        raise ValueError("SUPABASE_URL environment variable is required")
    if not key:
        raise ValueError("SUPABASE_KEY environment variable is required")
    
    _client = PostgRESTClient(url, key)
    return _client


def set_client(client: PostgRESTClient) -> None:
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
        'created_at': note.created_at.isoformat() if note.created_at else None
    }


def _dict_to_paper_note(row: Dict[str, Any]) -> PaperNote:
    """Convert notes table row to PaperNote."""
    return PaperNote(
        paper_id=row['paper_id'],
        pillar_id=PillarID(row['pillar_id']),
        problem=row['problem'],
        method=row['method'],
        findings=row['findings'],
        limitations=row.get('limitations'),
        future_work=row.get('future_work'),
        key_terms=row.get('key_terms', []),
        created_at=datetime.fromisoformat(row['created_at']) if row.get('created_at') else None
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
        'difficulty': lesson.difficulty,
        'estimated_time': lesson.estimated_time,
        'created_at': lesson.created_at.isoformat() if lesson.created_at else None
    }


def _dict_to_lesson(row: Dict[str, Any]) -> Lesson:
    """Convert lessons table row to Lesson."""
    return Lesson(
        paper_id=row['paper_id'],
        pillar_id=PillarID(row['pillar_id']),
        title=f"Lesson from {row['paper_id']}",  # Generate title as it's not stored in DB
        content=row.get('tl_dr', ''),  # Use tl_dr as content fallback
        tl_dr=row['tl_dr'],
        takeaways=row.get('takeaways', []),
        practice_ideas=row.get('practice_ideas', []),
        connections=row.get('connections', []),
        examples=[],  # Not stored in DB, provide empty list
        podcast_script=None,  # Not stored in lessons table
        difficulty=row.get('difficulty', 2),
        estimated_time=row.get('estimated_time', 10),
        created_at=datetime.fromisoformat(row['created_at']) if row.get('created_at') else None
    )


def _quiz_card_to_dict(card: QuizCard) -> Dict[str, Any]:
    """Convert QuizCard to quiz_cards table row."""
    return {
        'paper_id': card.paper_id,
        'pillar_id': card.pillar_id.value,
        'question': card.question,
        'answer': card.answer,
        'difficulty': card.difficulty,
        'question_type': card.question_type.value if hasattr(card.question_type, 'value') else card.question_type,
        'interval': card.interval,
        'repetitions': card.repetitions,
        'ease_factor': card.ease_factor,
        'due_date': card.due_date.isoformat() if card.due_date else None,
        'last_reviewed': card.last_reviewed.isoformat() if card.last_reviewed else None,
        'created_at': datetime.now(timezone.utc).isoformat()
    }


def _dict_to_quiz_card(row: Dict[str, Any]) -> QuizCard:
    """Convert quiz_cards table row to QuizCard."""
    from .schemas import QuestionType
    return QuizCard(
        id=row.get('id'),
        paper_id=row['paper_id'],
        pillar_id=PillarID(row['pillar_id']),
        question=row['question'],
        answer=row['answer'],
        difficulty=row['difficulty'],
        question_type=QuestionType(row.get('question_type', 'factual')),
        tags=[],  # Not stored in DB, provide empty list
        interval=row.get('interval', 1),
        repetitions=row.get('repetitions', 0),
        ease_factor=row.get('ease_factor', 2.5),
        due_date=datetime.fromisoformat(row['due_date']) if row.get('due_date') else datetime.now(timezone.utc),
        last_reviewed=datetime.fromisoformat(row['last_reviewed']) if row.get('last_reviewed') else None,
        review_count=row.get('repetitions', 0),  # Map repetitions to review_count
        interval_days=row.get('interval', 1)  # Map interval to interval_days
    )


# =====================================
# Paper Operations
# =====================================

def add_paper(pillar_id: PillarID, paper: PaperRef) -> bool:
    """
    Add a paper to the papers table for a specific pillar.
    
    Args:
        pillar_id: The pillar this paper belongs to
        paper: Paper reference to add
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_client()
        data = _paper_ref_to_dict(pillar_id, paper)
        
        response = client.table('papers').insert(data)
        
        if response['error']:
            # Check if it's a duplicate key error (HTTP 409 Conflict)
            error_info = response['error']
            if isinstance(error_info, dict) and error_info.get('code') == 409:
                logger.info(f"Paper {paper.id} already exists for pillar {pillar_id.value} (HTTP 409), skipping insert")
                return True  # Consider it successful if already exists
            
            error_str = str(error_info).lower()
            if 'duplicate' in error_str or 'conflict' in error_str or 'already exists' in error_str:
                logger.info(f"Paper {paper.id} already exists for pillar {pillar_id.value}, skipping insert")
                return True  # Consider it successful if already exists
            logger.error(f"Failed to add paper {paper.id} for pillar {pillar_id.value}: {response['error']}")
            return False
            
        logger.info(f"Added paper {paper.id} for pillar {pillar_id.value}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to add paper {paper.id} for pillar {pillar_id.value}: {e}")
        return False


def get_papers(pillar_id: PillarID, limit: int = 10) -> List[PaperRef]:
    """
    Get papers for a specific pillar.
    
    Args:
        pillar_id: The pillar to get papers for
        limit: Maximum number of papers to return
        
    Returns:
        List of paper references
    """
    try:
        client = get_client()
        response = (client.table('papers')
                   .select('*')
                   .eq('pillar_id', pillar_id.value)
                   .order('created_at', desc=True)
                   .limit(limit)
                   .execute())
        
        if response['error']:
            logger.error(f"Failed to get papers for pillar {pillar_id.value}: {response['error']}")
            return []
        
        if not response['data']:
            return []
            
        return [_dict_to_paper_ref(row) for row in response['data']]
        
    except Exception as e:
        logger.error(f"Failed to get papers for pillar {pillar_id.value}: {e}")
        return []


def paper_exists(pillar_id: PillarID, paper_id: str) -> bool:
    """
    Check if a paper exists for a specific pillar.
    
    Args:
        pillar_id: The pillar to check
        paper_id: Paper ID to check
        
    Returns:
        True if paper exists, False otherwise
    """
    try:
        client = get_client()
        response = (client.table('papers')
                   .select('id')
                   .eq('pillar_id', pillar_id.value)
                   .eq('id', paper_id)
                   .execute())
        
        if response['error']:
            logger.error(f"Failed to check paper {paper_id} for pillar {pillar_id.value}: {response['error']}")
            return False
            
        return bool(response['data'])
        
    except Exception as e:
        logger.error(f"Failed to check paper {paper_id} for pillar {pillar_id.value}: {e}")
        return False


# =====================================
# Notes Operations
# =====================================

def add_note(note: PaperNote) -> bool:
    """
    Add a note for a paper.
    
    Args:
        note: Note to add
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_client()
        data = _paper_note_to_dict(note)
        
        response = client.table('notes').insert(data)
        
        if response['error']:
            logger.error(f"Failed to add note for paper {note.paper_id}: {response['error']}")
            return False
            
        logger.info(f"Added note for paper {note.paper_id} in pillar {note.pillar_id.value}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to add note for paper {note.paper_id}: {e}")
        return False


def get_recent_notes(pillar_id: PillarID, limit: int = 5) -> List[PaperNote]:
    """
    Get recent notes for a specific pillar.
    
    Args:
        pillar_id: The pillar to get notes for
        limit: Maximum number of notes to return
        
    Returns:
        List of recent notes
    """
    try:
        client = get_client()
        response = (client.table('notes')
                   .select('*')
                   .eq('pillar_id', pillar_id.value)
                   .order('created_at', desc=True)
                   .limit(limit)
                   .execute())
        
        if response['error']:
            logger.error(f"Failed to get recent notes for pillar {pillar_id.value}: {response['error']}")
            raise Exception(f"Failed to get recent notes for pillar {pillar_id.value}: {response['error']}")
        
        if not response['data']:
            return []
            
        return [_dict_to_paper_note(row) for row in response['data']]
        
    except Exception as e:
        logger.error(f"Failed to get recent notes for pillar {pillar_id.value}: {e}")
        raise Exception(f"Failed to get recent notes for pillar {pillar_id.value}: {e}")


# =====================================
# Lessons Operations
# =====================================

def add_lesson(lesson: Lesson) -> bool:
    """
    Add a lesson for a paper.
    
    Args:
        lesson: Lesson to add
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_client()
        data = _lesson_to_dict(lesson)
        
        response = client.table('lessons').insert(data)
        
        if response['error']:
            logger.error(f"Failed to add lesson for paper {lesson.paper_id}: {response['error']}")
            return False
            
        logger.info(f"Added lesson for paper {lesson.paper_id} in pillar {lesson.pillar_id.value}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to add lesson for paper {lesson.paper_id}: {e}")
        return False


def get_lessons(pillar_id: PillarID, limit: int = 10) -> List[Lesson]:
    """
    Get lessons for a specific pillar.
    
    Args:
        pillar_id: The pillar to get lessons for
        limit: Maximum number of lessons to return
        
    Returns:
        List of lessons
    """
    try:
        client = get_client()
        response = (client.table('lessons')
                   .select('*')
                   .eq('pillar_id', pillar_id.value)
                   .order('created_at', desc=True)
                   .limit(limit)
                   .execute())
        
        if response['error']:
            logger.error(f"Failed to get lessons for pillar {pillar_id.value}: {response['error']}")
            return []
        
        if not response['data']:
            return []
            
        return [_dict_to_lesson(row) for row in response['data']]
        
    except Exception as e:
        logger.error(f"Failed to get lessons for pillar {pillar_id.value}: {e}")
        return []


# =====================================
# Quiz Cards Operations
# =====================================

def add_quiz_cards(cards: List[QuizCard]) -> int:
    """
    Add multiple quiz cards.
    
    Args:
        cards: List of quiz cards to add
        
    Returns:
        Number of cards successfully added
    """
    if not cards:
        return 0
        
    added = 0
    client = get_client()
    
    for card in cards:
        try:
            data = _quiz_card_to_dict(card)
            response = client.table('quiz_cards').insert(data)
            
            if response['error']:
                logger.error(f"Failed to add quiz card: {response['error']}")
            else:
                added += 1
                
        except Exception as e:
            logger.error(f"Failed to add quiz card: {e}")
    
    logger.info(f"Added {added}/{len(cards)} quiz cards")
    return added


def get_quiz_cards_for_review(pillar_id: PillarID, limit: int = 10) -> List[QuizCard]:
    """
    Get quiz cards due for review for a specific pillar.
    
    Args:
        pillar_id: The pillar to get cards for
        limit: Maximum number of cards to return
        
    Returns:
        List of quiz cards due for review
    """
    try:
        client = get_client()
        
        # For now, just get cards ordered by last_reviewed (oldest first)
        response = (client.table('quiz_cards')
                   .select('*')
                   .eq('pillar_id', pillar_id.value)
                   .order('last_reviewed')  # Nulls first, then oldest
                   .limit(limit)
                   .execute())
        
        if response['error']:
            logger.error(f"Failed to get quiz cards for pillar {pillar_id.value}: {response['error']}")
            return []
        
        if not response['data']:
            return []
            
        return [_dict_to_quiz_card(row) for row in response['data']]
        
    except Exception as e:
        logger.error(f"Failed to get quiz cards for pillar {pillar_id.value}: {e}")
        return []


def update_quiz_card_review(
    pillar_id: PillarID,
    paper_id: str,
    card_index: int,
    quality: int  # 0-5, where 5 is perfect recall
) -> bool:
    """
    Update quiz card after review using spaced repetition algorithm.
    
    Args:
        pillar_id: The pillar the card belongs to
        paper_id: Paper ID the card is from
        card_index: Index of the card in the paper's quiz set
        quality: Quality of recall (0-5)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_client()
        
        # First get the card
        response = (client.table('quiz_cards')
                   .select('*')
                   .eq('pillar_id', pillar_id.value)
                   .eq('paper_id', paper_id)
                   .limit(1)
                   .execute())
        
        if response['error'] or not response['data']:
            logger.error(f"Quiz card not found")
            return False
        
        card_data = response['data'][0]
        
        # Update using SM-2 algorithm
        ease_factor = card_data.get('ease_factor', 2.5)
        interval_days = card_data.get('interval_days', 1)
        review_count = card_data.get('review_count', 0)
        
        # Calculate new values
        if quality < 3:
            interval_days = 1
            review_count = 0
        else:
            if review_count == 0:
                interval_days = 1
            elif review_count == 1:
                interval_days = 6
            else:
                interval_days = int(interval_days * ease_factor)
            
            review_count += 1
            ease_factor = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
            ease_factor = max(1.3, ease_factor)
        
        # Update the card
        update_data = {
            'last_reviewed': datetime.now(timezone.utc).isoformat(),
            'review_count': review_count,
            'ease_factor': ease_factor,
            'interval_days': interval_days
        }
        
        response = (client.table('quiz_cards')
                   .update(update_data)
                   .eq('pillar_id', pillar_id.value)
                   .eq('paper_id', paper_id))
        
        if response['error']:
            logger.error(f"Failed to update quiz card: {response['error']}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to update quiz card: {e}")
        return False


# =====================================
# Paper Queue Operations
# =====================================

def add_to_paper_queue(pillar_id: PillarID, papers: List[PaperRef], priority: int = 0) -> int:
    """
    Add papers to the processing queue for a specific pillar.
    
    Args:
        pillar_id: The pillar to add papers for
        papers: List of papers to add to queue
        priority: Priority level (higher = processed first)
        
    Returns:
        Number of papers successfully added to queue
    """
    if not papers:
        return 0
    
    client = get_client()
    added = 0
    
    # First, get existing paper IDs for this pillar to avoid duplicates
    try:
        response = (client.table('papers')
                   .select('id')
                   .eq('pillar_id', pillar_id.value)
                   .execute())
        
        if response['error']:
            logger.error(f"Failed to get existing papers: {response['error']}")
            raise Exception(f"Failed to add candidates to queue for pillar {pillar_id.value}: {response['error']}")
        
        existing_ids = {row['id'] for row in (response['data'] or [])}
        
    except Exception as e:
        logger.error(f"Failed to check existing papers: {e}")
        raise Exception(f"Failed to add candidates to queue for pillar {pillar_id.value}: {e}")
    
    # Add papers that don't exist yet
    for paper in papers:
        if paper.id in existing_ids:
            logger.debug(f"Paper {paper.id} already exists for pillar {pillar_id.value}, skipping")
            continue
            
        try:
            queue_data = {
                'paper_id': paper.id,
                'pillar_id': pillar_id.value,
                'title': paper.title,
                'source': 'arxiv',  # Default source, could be enhanced to track actual source
                'priority': priority,
                'processed': False
            }
            
            response = client.table('paper_queue').insert(queue_data)
            
            if response['error']:
                if 'duplicate' in str(response['error']).lower():
                    logger.debug(f"Paper {paper.id} already in queue for pillar {pillar_id.value}")
                else:
                    logger.error(f"Failed to add paper {paper.id} to queue: {response['error']}")
            else:
                added += 1
                
        except Exception as e:
            logger.error(f"Failed to add paper {paper.id} to queue: {e}")
    
    logger.info(f"Added {added}/{len(papers)} papers to queue for pillar {pillar_id.value}")
    return added


def pop_from_paper_queue(pillar_id: PillarID, limit: int = 1) -> List[PaperRef]:
    """
    Get and mark papers from the queue as being processed.
    
    Args:
        pillar_id: The pillar to get papers for
        limit: Number of papers to pop from queue
        
    Returns:
        List of papers to process
    """
    try:
        client = get_client()
        
        # Get unprocessed papers ordered by priority
        response = (client.table('paper_queue')
                   .select('*')
                   .eq('pillar_id', pillar_id.value)
                   .eq('processed', False)
                   .order('priority', desc=True)
                   .order('added_at', desc=True)
                   .limit(limit)
                   .execute())
        
        if response['error']:
            logger.error(f"Failed to pop papers from queue: {response['error']}")
            raise Exception(f"Failed to pop papers from queue for pillar {pillar_id.value}: {response['error']}")
        
        if not response['data']:
            return []
        
        papers = []
        for row in response['data']:
            # Create a basic PaperRef with available data from queue
            # Note: Full paper data should be retrieved from papers table if needed
            paper = PaperRef(
                id=row['paper_id'],
                title=row['title'],
                authors=[],  # Not stored in queue, would need to fetch from papers table
                venue=None,  # Not stored in queue
                year=None,   # Not stored in queue
                url_pdf=f"https://arxiv.org/pdf/{row['paper_id']}.pdf",  # Construct arXiv URL
                abstract=None,  # Not stored in queue
                citation_count=0  # Not stored in queue
            )
            papers.append(paper)
            
            # Mark as processed
            update_response = (client.table('paper_queue')
                             .eq('pillar_id', pillar_id.value)
                             .eq('paper_id', row['paper_id'])
                             .update({'processed': True}))
            
            if update_response['error']:
                logger.error(f"Failed to mark paper {row['paper_id']} as processed: {update_response['error']}")
        
        return papers
        
    except Exception as e:
        logger.error(f"Failed to pop papers from queue: {e}")
        raise Exception(f"Failed to pop papers from queue for pillar {pillar_id.value}: {e}")


def get_queue_size(pillar_id: PillarID) -> int:
    """
    Get the number of unprocessed papers in queue for a pillar.
    
    Args:
        pillar_id: The pillar to check
        
    Returns:
        Number of unprocessed papers in queue
    """
    try:
        client = get_client()
        response = (client.table('paper_queue')
                   .select('paper_id')
                   .eq('pillar_id', pillar_id.value)
                   .eq('processed', False)
                   .execute())
        
        if response['error']:
            logger.error(f"Failed to get queue size: {response['error']}")
            return 0
            
        return len(response['data'] or [])
        
    except Exception as e:
        logger.error(f"Failed to get queue size: {e}")
        return 0


# =====================================
# Aliases for Orchestrator Compatibility
# =====================================

# Queue operations aliases
queue_add_candidates = add_to_paper_queue  # Alias for orchestrator
queue_pop_next = pop_from_paper_queue  # Alias for orchestrator

# Paper operations aliases
def upsert_paper(pillar_id: PillarID, paper: PaperRef) -> bool:
    """Alias for add_paper to match orchestrator expectations."""
    return add_paper(pillar_id, paper)

# Note operations aliases
def insert_note(note: PaperNote) -> bool:
    """Alias for add_note to match orchestrator expectations."""
    return add_note(note)

# Lesson operations aliases
def insert_lesson(lesson: Lesson) -> bool:
    """Alias for add_lesson to match orchestrator expectations."""
    return add_lesson(lesson)

# Quiz operations aliases
def insert_quiz_cards(cards: List[QuizCard]) -> int:
    """Alias for add_quiz_cards to match orchestrator expectations."""
    return add_quiz_cards(cards)

# Processing status operations
def mark_processed(pillar_id: PillarID, paper_id: str) -> bool:
    """Mark a paper as processed in the queue."""
    try:
        client = get_client()
        response = (client.table('paper_queue')
                   .eq('pillar_id', pillar_id.value)
                   .eq('paper_id', paper_id)
                   .update({'processed': True}))
        
        if response['error']:
            logger.error(f"Failed to mark paper {paper_id} as processed: {response['error']}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Failed to mark paper {paper_id} as processed: {e}")
        return False
