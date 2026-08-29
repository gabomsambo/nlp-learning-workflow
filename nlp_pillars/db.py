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

from .paper_ids import is_arxiv_id, resolvable_pdf_url
from .schemas import (
    PaperRef, PaperNote, Lesson, QuizCard, Pillar, PillarCreate, PillarUpdate,
    ReviewLog, UserFSRSParameters, FSRSRating, CardState, QuizReviewRequest, QuizReviewResponse,
    PaperCitation, PodcastScript, SourceMaterial,
    PipelineRun, PipelineRunStage, RunStatus, StageStatus
)

logger = logging.getLogger(__name__)


def _is_missing_column(error: Any, column: str) -> bool:
    """True if a PostgREST error means `column` does not exist on the table.

    PostgREST reports an unknown column as PGRST204 with a message naming it.
    Used so a database that has not had a hand-applied migration run against it
    degrades visibly instead of failing the write.
    """
    text = str(error).lower()
    return column.lower() in text and ("pgrst204" in text or "column" in text)


def _is_missing_url_pdf_column(error: Any) -> bool:
    """True if a PostgREST error means paper_queue.url_pdf does not exist.

    PostgREST reports an unknown column as PGRST204 with a message naming it.
    Databases created before docs/migrations/008_paper_queue_url_pdf.sql hit
    this; the caller degrades to arXiv-only queueing rather than failing.
    """
    text = str(error).lower()
    return "url_pdf" in text and ("pgrst204" in text or "column" in text)


class PostgRESTClient:
    """Simple PostgREST client that works with vanilla PostgREST."""

    def __init__(self, url: str, key: str):
        url = url.rstrip('/')
        # Supabase cloud uses /rest/v1/ path, local PostgREST uses root
        if "supabase.co" in url and "/rest/v1" not in url:
            url = f"{url}/rest/v1"
        self.base_url = url
        self.headers = {
            'apikey': key,
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
                logger.info("Conflict detected (409): Resource already exists")
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

    def delete(self):
        """Delete data."""
        response = self.client.delete(self.url, params=self.params)
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

def _paper_ref_to_dict(pillar_id: str, paper: PaperRef) -> Dict[str, Any]:
    """Convert PaperRef to papers table row.

    Raises:
        ValueError: if `pillar_id` or `paper.id` is missing. The None-filter at the
            bottom of this function exists to let optional metadata (venue, year,
            abstract) be omitted rather than written as NULL — but it applied to
            EVERY key, so a None pillar_id was quietly stripped and the row was
            inserted with no pillar at all. Pillar isolation is the one invariant
            this schema has; a write that cannot name its pillar is a bug in the
            caller, not a row to be written anyway.

            add_paper() catches this and returns False, so a daily run records the
            paper as failed and surfaces it, rather than aborting mid-run.

    """
    if not pillar_id:
        raise ValueError(
            f"pillar_id is required to write paper {paper.id!r}; "
            "refusing to insert a paper with no pillar"
        )
    if not paper.id:
        raise ValueError("paper.id is required; refusing to insert a paper with no id")

    data = {
        'id': paper.id,
        'pillar_id': pillar_id,
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
        'pillar_id': note.pillar_id,
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
        pillar_id=row['pillar_id'],
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
        'pillar_id': lesson.pillar_id,
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
        pillar_id=row['pillar_id'],
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


def _podcast_script_to_dict(script: PodcastScript) -> Dict[str, Any]:
    """Convert PodcastScript to podcast_scripts table row."""
    return {
        'paper_id': script.paper_id,
        'pillar_id': script.pillar_id,
        'title': script.title,
        'script': script.script,
        'word_count': script.word_count,
        'key_points': script.key_points,
        'ground_pack': script.ground_pack,
        # What the script was written from. Requires
        # docs/migrations/011_podcast_source_material.sql on databases created
        # before it; add_podcast_script() retries without the key and says so.
        'source_material': script.source_material.model_dump(),
        'created_at': script.created_at.isoformat() if script.created_at else None
    }


def _dict_to_podcast_script(row: Dict[str, Any]) -> PodcastScript:
    """Convert podcast_scripts table row to PodcastScript."""
    return PodcastScript(
        id=row.get('id'),
        paper_id=row['paper_id'],
        pillar_id=row['pillar_id'],
        title=row['title'],
        script=row.get('script', ''),
        word_count=row.get('word_count', 0),
        key_points=row.get('key_points', []),
        ground_pack=row.get('ground_pack', {}),
        # Absent on rows written before 011, and on any row a pre-011 database
        # accepted through the fallback insert. Defaults to level="full", which
        # is what every historical row was assumed to be anyway.
        source_material=SourceMaterial(**(row.get('source_material') or {})),
        created_at=datetime.fromisoformat(row['created_at']) if row.get('created_at') else None
    )


def _quiz_card_to_dict(card: QuizCard) -> Dict[str, Any]:
    """Convert QuizCard to quiz_cards table row."""
    data = {
        'paper_id': card.paper_id,
        'pillar_id': card.pillar_id,
        'question': card.question,
        'answer': card.answer,
        'difficulty': card.difficulty,
        'question_type': card.question_type.value if hasattr(card.question_type, 'value') else card.question_type,
        'user_id': card.user_id,

        # FSRS fields
        'difficulty_fsrs': card.difficulty_fsrs,
        'stability': card.stability,
        'retrievability': card.retrievability,
        'state': card.state.value if hasattr(card.state, 'value') else card.state,
        'lapses': card.lapses,
        'last_review_date': card.last_review_date.isoformat() if card.last_review_date else None,
        'next_review_date': card.next_review_date.isoformat() if card.next_review_date else None,

        # Legacy SM-2 fields (for backward compatibility)
        'interval': card.interval,
        'repetitions': card.repetitions,
        'ease_factor': card.ease_factor,
        'due_date': card.due_date.isoformat() if card.due_date else None,
        'last_reviewed': card.last_reviewed.isoformat() if card.last_reviewed else None,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    # Remove None values
    return {k: v for k, v in data.items() if v is not None}


def _dict_to_quiz_card(row: Dict[str, Any]) -> QuizCard:
    """Convert quiz_cards table row to QuizCard."""
    from .schemas import QuestionType
    return QuizCard(
        id=row.get('id'),
        paper_id=row['paper_id'],
        pillar_id=row['pillar_id'],
        question=row['question'],
        answer=row['answer'],
        difficulty=row['difficulty'],
        question_type=QuestionType(row.get('question_type', 'factual')),
        tags=[],  # Not stored in DB, provide empty list
        user_id=row.get('user_id', 'default_user'),

        # FSRS fields
        difficulty_fsrs=row.get('difficulty_fsrs', 0.0),
        stability=row.get('stability', 0.0),
        retrievability=row.get('retrievability'),
        state=CardState(row.get('state', 'new')),
        lapses=row.get('lapses', 0),
        last_review_date=datetime.fromisoformat(row['last_review_date']) if row.get('last_review_date') else None,
        next_review_date=datetime.fromisoformat(row['next_review_date']) if row.get('next_review_date') else None,

        # Legacy SM-2 fields (for backward compatibility)
        interval=row.get('interval', 1),
        repetitions=row.get('repetitions', 0),
        ease_factor=row.get('ease_factor', 2.5),
        due_date=datetime.fromisoformat(row['due_date']) if row.get('due_date') else datetime.now(timezone.utc),
        last_reviewed=datetime.fromisoformat(row['last_reviewed']) if row.get('last_reviewed') else None,
        review_count=row.get('repetitions', 0),  # Map repetitions to review_count
        interval_days=row.get('interval', 1)  # Map interval to interval_days
    )


def _review_log_to_dict(log: ReviewLog) -> Dict[str, Any]:
    """Convert ReviewLog to review_logs table row."""
    data = {
        'card_id': log.card_id,
        'user_id': log.user_id,
        'pillar_id': log.pillar_id,
        'paper_id': log.paper_id,
        'rating': log.rating.value if hasattr(log.rating, 'value') else log.rating,
        'review_timestamp': log.review_timestamp.isoformat(),
        'difficulty': log.difficulty,
        'stability': log.stability,
        'retrievability': log.retrievability,
        'previous_due_date': log.previous_due_date.isoformat() if log.previous_due_date else None,
        'days_overdue': log.days_overdue,
        'session_id': log.session_id,
        'response_time_ms': log.response_time_ms
    }
    # Remove None values
    return {k: v for k, v in data.items() if v is not None}


def _dict_to_review_log(row: Dict[str, Any]) -> ReviewLog:
    """Convert review_logs table row to ReviewLog."""
    return ReviewLog(
        id=row.get('id'),
        card_id=row['card_id'],
        user_id=row.get('user_id', 'default_user'),
        pillar_id=row['pillar_id'],
        paper_id=row['paper_id'],
        rating=FSRSRating(row['rating']),
        review_timestamp=datetime.fromisoformat(row['review_timestamp']),
        difficulty=row['difficulty'],
        stability=row['stability'],
        retrievability=row.get('retrievability'),
        previous_due_date=datetime.fromisoformat(row['previous_due_date']) if row.get('previous_due_date') else None,
        days_overdue=row.get('days_overdue', 0),
        session_id=row.get('session_id'),
        response_time_ms=row.get('response_time_ms')
    )


def _user_fsrs_parameters_to_dict(params: UserFSRSParameters) -> Dict[str, Any]:
    """Convert UserFSRSParameters to user_fsrs_parameters table row."""
    data = {
        'user_id': params.user_id,
        'pillar_id': params.pillar_id if params.pillar_id else None,
        'w0': params.w0, 'w1': params.w1, 'w2': params.w2, 'w3': params.w3,
        'w4': params.w4, 'w5': params.w5, 'w6': params.w6, 'w7': params.w7,
        'w8': params.w8, 'w9': params.w9, 'w10': params.w10, 'w11': params.w11,
        'w12': params.w12, 'w13': params.w13, 'w14': params.w14, 'w15': params.w15,
        'w16': params.w16, 'w17': params.w17,
        'review_count': params.review_count,
        'last_optimized': params.last_optimized.isoformat() if params.last_optimized else None,
        'optimization_score': params.optimization_score,
        'is_default': params.is_default,
        'updated_at': datetime.now(timezone.utc).isoformat()
    }
    # Remove None values
    return {k: v for k, v in data.items() if v is not None}


def _dict_to_user_fsrs_parameters(row: Dict[str, Any]) -> UserFSRSParameters:
    """Convert user_fsrs_parameters table row to UserFSRSParameters."""
    return UserFSRSParameters(
        id=row.get('id'),
        user_id=row.get('user_id', 'default_user'),
        pillar_id=row['pillar_id'] if row.get('pillar_id') else None,
        w0=row.get('w0', 0.4), w1=row.get('w1', 0.6), w2=row.get('w2', 2.4), w3=row.get('w3', 5.8),
        w4=row.get('w4', 4.93), w5=row.get('w5', 0.94), w6=row.get('w6', 0.86), w7=row.get('w7', 0.01),
        w8=row.get('w8', 1.49), w9=row.get('w9', 0.14), w10=row.get('w10', 0.94), w11=row.get('w11', 2.18),
        w12=row.get('w12', 0.05), w13=row.get('w13', 0.34), w14=row.get('w14', 0.67), w15=row.get('w15', 2.74),
        w16=row.get('w16', 0.0), w17=row.get('w17', 2.0),
        review_count=row.get('review_count', 0),
        last_optimized=datetime.fromisoformat(row['last_optimized']) if row.get('last_optimized') else None,
        optimization_score=row.get('optimization_score'),
        is_default=row.get('is_default', False),
        created_at=datetime.fromisoformat(row['created_at']) if row.get('created_at') else datetime.now(timezone.utc),
        updated_at=datetime.fromisoformat(row['updated_at']) if row.get('updated_at') else datetime.now(timezone.utc)
    )


# =====================================
# Paper Operations
# =====================================

def add_paper(pillar_id: str, paper: PaperRef) -> bool:
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
                logger.info(f"Paper {paper.id} already exists for pillar {pillar_id} (HTTP 409), skipping insert")
                return True  # Consider it successful if already exists

            error_str = str(error_info).lower()
            if 'duplicate' in error_str or 'conflict' in error_str or 'already exists' in error_str:
                logger.info(f"Paper {paper.id} already exists for pillar {pillar_id}, skipping insert")
                return True  # Consider it successful if already exists
            logger.error(f"Failed to add paper {paper.id} for pillar {pillar_id}: {response['error']}")
            return False

        logger.info(f"Added paper {paper.id} for pillar {pillar_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to add paper {paper.id} for pillar {pillar_id}: {e}")
        return False


def get_papers(pillar_id: str, limit: int = 10) -> List[PaperRef]:
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
                   .eq('pillar_id', pillar_id)
                   .order('added_at', desc=True)
                   .limit(limit)
                   .execute())

        if response['error']:
            logger.error(f"Failed to get papers for pillar {pillar_id}: {response['error']}")
            return []

        if not response['data']:
            return []

        return [_dict_to_paper_ref(row) for row in response['data']]

    except Exception as e:
        logger.error(f"Failed to get papers for pillar {pillar_id}: {e}")
        return []


def paper_exists(pillar_id: str, paper_id: str) -> bool:
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
                   .eq('pillar_id', pillar_id)
                   .eq('id', paper_id)
                   .execute())

        if response['error']:
            logger.error(f"Failed to check paper {paper_id} for pillar {pillar_id}: {response['error']}")
            return False

        return bool(response['data'])

    except Exception as e:
        logger.error(f"Failed to check paper {paper_id} for pillar {pillar_id}: {e}")
        return False


class PaperLookupError(Exception):
    """The paper list could not be read from the database.

    Distinct from "there are no papers", which is an empty list. Collapsing the
    two rendered the /podcast page's paper dropdown empty, with no error, when
    the database was simply unreachable.
    """


def get_all_papers(limit: int = 100) -> List[PaperRef]:
    """
    Get all papers across all pillars.

    Args:
        limit: Maximum number of papers to return

    Returns:
        List of paper references. Empty means there are none — nothing else.

    Raises:
        PaperLookupError: the database could not be read.
    """
    try:
        client = get_client()
        response = (client.table('papers')
                   .select('*')
                   .order('added_at', desc=True)
                   .limit(limit)
                   .execute())

        if response['error']:
            logger.error(f"Failed to get all papers: {response['error']}")
            raise PaperLookupError(f"Could not read the paper list: {response['error']}")

        if not response['data']:
            return []

        return [_dict_to_paper_ref(row) for row in response['data']]

    except PaperLookupError:
        raise
    except Exception as e:
        logger.error(f"Failed to get all papers: {e}")
        raise PaperLookupError(f"Could not read the paper list: {e}") from e


def get_paper_by_id(paper_id: str) -> Optional[PaperRef]:
    """
    Get a paper by its ID.

    Args:
        paper_id: Paper ID to fetch

    Returns:
        PaperRef if found, None otherwise
    """
    try:
        client = get_client()
        response = (client.table('papers')
                   .select('*')
                   .eq('id', paper_id)
                   .execute())

        if response['error']:
            logger.error(f"Failed to get paper {paper_id}: {response['error']}")
            return None

        if not response['data'] or len(response['data']) == 0:
            return None

        return _dict_to_paper_ref(response['data'][0])

    except Exception as e:
        logger.error(f"Failed to get paper {paper_id}: {e}")
        return None


# =====================================
# Notes Operations
# =====================================


def get_notes_by_paper_id(paper_id: str) -> Optional[PaperNote]:
    """
    Get notes for a paper by paper ID.

    Args:
        paper_id: Paper ID to get notes for

    Returns:
        PaperNote if found, None otherwise
    """
    try:
        client = get_client()
        response = (client.table('notes')
                   .select('*')
                   .eq('paper_id', paper_id)
                   .order('created_at', desc=True)
                   .limit(1)
                   .execute())

        if response['error']:
            logger.error(f"Failed to get notes for paper {paper_id}: {response['error']}")
            return None

        if not response['data'] or len(response['data']) == 0:
            return None

        return _dict_to_paper_note(response['data'][0])

    except Exception as e:
        logger.error(f"Failed to get notes for paper {paper_id}: {e}")
        return None

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

        logger.info(f"Added note for paper {note.paper_id} in pillar {note.pillar_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to add note for paper {note.paper_id}: {e}")
        return False


def get_recent_notes(pillar_id: str, limit: int = 5) -> List[PaperNote]:
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
                   .eq('pillar_id', pillar_id)
                   .order('created_at', desc=True)
                   .limit(limit)
                   .execute())

        if response['error']:
            logger.error(f"Failed to get recent notes for pillar {pillar_id}: {response['error']}")
            raise Exception(f"Failed to get recent notes for pillar {pillar_id}: {response['error']}")

        if not response['data']:
            return []

        return [_dict_to_paper_note(row) for row in response['data']]

    except Exception as e:
        logger.error(f"Failed to get recent notes for pillar {pillar_id}: {e}")
        raise Exception(f"Failed to get recent notes for pillar {pillar_id}: {e}")


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

        logger.info(f"Added lesson for paper {lesson.paper_id} in pillar {lesson.pillar_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to add lesson for paper {lesson.paper_id}: {e}")
        return False


def get_lessons(pillar_id: str, limit: int = 10) -> List[Lesson]:
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
                   .eq('pillar_id', pillar_id)
                   .order('created_at', desc=True)
                   .limit(limit)
                   .execute())

        if response['error']:
            logger.error(f"Failed to get lessons for pillar {pillar_id}: {response['error']}")
            return []

        if not response['data']:
            return []

        return [_dict_to_lesson(row) for row in response['data']]

    except Exception as e:
        logger.error(f"Failed to get lessons for pillar {pillar_id}: {e}")
        return []


# =====================================
# Podcast Script Operations
# =====================================

class PodcastScriptSaveError(Exception):
    """A generated podcast script could not be persisted.

    Raised rather than returned as None because the caller has an artifact in
    its hands that already cost ~$0.27 and four minutes to make, and the only
    correct response to a failed insert is to hand that artifact back to the
    user together with the real reason. The old `return None` made every
    failure — dead database, missing table, bad grant, constraint violation —
    into the same opaque 500 with the script silently discarded.
    """


class PodcastScriptLookupError(Exception):
    """Podcast scripts could not be read from the database.

    Distinct from "there is no such script", which is None, and from "there are
    no scripts", which is an empty list. Same precedent as PillarLookupError and
    PostgrestClient.get_pipeline_run: a transient failure must not tell the
    browser the thing is gone. Measured before this split: a malformed script id
    made PostgREST answer `400 invalid input syntax for type uuid`, which the
    route reported as 404 "Script not found".
    """


def add_podcast_script(script: PodcastScript) -> str:
    """
    Add a podcast script for a paper.

    Args:
        script: PodcastScript to add

    Returns:
        The new script's ID.

    Raises:
        PodcastScriptSaveError: the row could not be written. Never swallowed —
            see the class docstring; the caller owns an artifact that has
            already been paid for.
    """
    try:
        client = get_client()
        data = _podcast_script_to_dict(script)

        response = client.table('podcast_scripts').insert(data)

        if response['error'] and _is_missing_column(response['error'], 'source_material'):
            # Same degradation as paper_queue.url_pdf: keep the write working on
            # a database that has not had the migration applied, and say loudly
            # what was dropped. Losing the provenance record is much better than
            # losing the script.
            logger.error(
                "podcast_scripts has no source_material column — run "
                "docs/migrations/011_podcast_source_material.sql. Saving the "
                "script without its source-material record."
            )
            data.pop('source_material')
            response = client.table('podcast_scripts').insert(data)

        if response['error']:
            logger.error(f"Failed to add podcast script for paper {script.paper_id}: {response['error']}")
            raise PodcastScriptSaveError(str(response['error']))

        # Get the ID from the response
        if response['data'] and len(response['data']) > 0:
            script_id = response['data'][0].get('id')
            logger.info(f"Added podcast script {script_id} for paper {script.paper_id}")
            return script_id

        raise PodcastScriptSaveError(
            "the insert returned no row, so the script has no id and may not have been stored"
        )

    except PodcastScriptSaveError:
        raise
    except Exception as e:
        logger.error(f"Failed to add podcast script for paper {script.paper_id}: {e}")
        raise PodcastScriptSaveError(str(e)) from e


def get_podcast_scripts(pillar_id: Optional[str] = None, limit: int = 10) -> List[PodcastScript]:
    """
    Get podcast scripts, optionally filtered by pillar.

    Args:
        pillar_id: Optional pillar to filter by
        limit: Maximum number of scripts to return

    Returns:
        List of podcast scripts. Empty means there are none — nothing else.

    Raises:
        PodcastScriptLookupError: the database could not be read. An unreachable
            database used to render /podcast as "No podcast scripts generated
            yet" with no error anywhere.
    """
    try:
        client = get_client()
        query = client.table('podcast_scripts').select('*')

        if pillar_id:
            query = query.eq('pillar_id', pillar_id)

        response = (query
                   .order('created_at', desc=True)
                   .limit(limit)
                   .execute())

        if response['error']:
            logger.error(f"Failed to get podcast scripts: {response['error']}")
            raise PodcastScriptLookupError(
                f"Could not read the podcast scripts: {response['error']}"
            )

        if not response['data']:
            return []

        return [_dict_to_podcast_script(row) for row in response['data']]

    except PodcastScriptLookupError:
        raise
    except Exception as e:
        logger.error(f"Failed to get podcast scripts: {e}")
        raise PodcastScriptLookupError(f"Could not read the podcast scripts: {e}") from e


def get_podcast_script_by_id(script_id: str) -> Optional[PodcastScript]:
    """
    Get a specific podcast script by ID.

    Args:
        script_id: The script ID to fetch

    Returns:
        PodcastScript, or None if there is genuinely no such script.

    Raises:
        PodcastScriptLookupError: the query failed. A missing row is already
            `data == []` with HTTP 200, so every error reaching here is a real
            failure and reporting it as None made the route answer 404 — telling
            the browser a script is gone when the database merely hiccuped.
    """
    try:
        client = get_client()
        response = (client.table('podcast_scripts')
                   .select('*')
                   .eq('id', script_id)
                   .execute())

        if response['error']:
            logger.error(f"Failed to get podcast script {script_id}: {response['error']}")
            raise PodcastScriptLookupError(
                f"Could not read podcast script {script_id}: {response['error']}"
            )

        if not response['data'] or len(response['data']) == 0:
            return None

        return _dict_to_podcast_script(response['data'][0])

    except PodcastScriptLookupError:
        raise
    except Exception as e:
        logger.error(f"Failed to get podcast script {script_id}: {e}")
        raise PodcastScriptLookupError(
            f"Could not read podcast script {script_id}: {e}"
        ) from e


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


def get_quiz_cards_for_review(pillar_id: str, limit: int = 10) -> List[QuizCard]:
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
                   .eq('pillar_id', pillar_id)
                   .order('last_reviewed')  # Nulls first, then oldest
                   .limit(limit)
                   .execute())

        if response['error']:
            logger.error(f"Failed to get quiz cards for pillar {pillar_id}: {response['error']}")
            return []

        if not response['data']:
            return []

        return [_dict_to_quiz_card(row) for row in response['data']]

    except Exception as e:
        logger.error(f"Failed to get quiz cards for pillar {pillar_id}: {e}")
        return []


def review_quiz_card_fsrs(request: QuizReviewRequest) -> QuizReviewResponse:
    """
    Review a quiz card using FSRS algorithm.
    
    Args:
        request: Review request with card ID and rating
        
    Returns:
        QuizReviewResponse with updated card and next review date
    """
    try:
        from .services.fsrs_service import get_fsrs_service

        client = get_client()
        fsrs_service = get_fsrs_service()

        # Get the card
        response = (client.table('quiz_cards')
                   .select('*')
                   .eq('id', request.card_id)
                   .limit(1)
                   .execute())

        if response['error'] or not response['data']:
            return QuizReviewResponse(
                success=False,
                card=None,
                next_review_date=datetime.now(timezone.utc),
                message="Quiz card not found"
            )

        # Convert to QuizCard object
        card = _dict_to_quiz_card(response['data'][0])

        # Get user's FSRS parameters
        user_params = get_user_fsrs_parameters(card.user_id, card.pillar_id)

        # Schedule the review using FSRS
        updated_card, next_review_date = fsrs_service.schedule_card_review(
            card, request.rating, user_params
        )

        # Create review log
        review_log = fsrs_service.create_review_log(
            card, request.rating, request.response_time_ms, request.session_id
        )

        # Update the card in database
        update_data = _quiz_card_to_dict(updated_card)
        update_response = (client.table('quiz_cards')
                          .eq('id', request.card_id)
                          .update(update_data))

        if update_response['error']:
            logger.error(f"Failed to update quiz card: {update_response['error']}")
            return QuizReviewResponse(
                success=False,
                card=card,
                next_review_date=next_review_date,
                message="Failed to update card"
            )

        # Save review log
        log_data = _review_log_to_dict(review_log)
        log_response = client.table('review_logs').insert(log_data)

        if log_response['error']:
            logger.warning(f"Failed to save review log: {log_response['error']}")

        logger.info(f"Reviewed card {request.card_id} with FSRS rating {request.rating}")

        return QuizReviewResponse(
            success=True,
            card=updated_card,
            next_review_date=next_review_date,
            message="Card reviewed successfully"
        )

    except Exception as e:
        logger.error(f"Failed to review quiz card: {e}")
        return QuizReviewResponse(
            success=False,
            card=None,
            next_review_date=datetime.now(timezone.utc),
            message=f"Review failed: {str(e)}"
        )


def update_quiz_card_review(
    pillar_id: str,
    paper_id: str,
    card_index: int,
    quality: int  # 0-5, where 5 is perfect recall
) -> bool:
    """
    Update quiz card after review using spaced repetition algorithm.
    DEPRECATED: Use review_quiz_card_fsrs() for new implementations.
    
    This function maintains backward compatibility with the old SM-2 implementation.
    """
    try:
        # Convert old quality rating to FSRS rating
        if quality <= 2:
            rating = FSRSRating.AGAIN
        elif quality == 3:
            rating = FSRSRating.HARD
        elif quality == 4:
            rating = FSRSRating.GOOD
        else:
            rating = FSRSRating.EASY

        # Get the card by pillar and paper (legacy approach)
        client = get_client()
        response = (client.table('quiz_cards')
                   .select('*')
                   .eq('pillar_id', pillar_id)
                   .eq('paper_id', paper_id)
                   .limit(1)
                   .execute())

        if response['error'] or not response['data']:
            logger.error(f"Quiz card not found for pillar {pillar_id}, paper {paper_id}")
            return False

        card_id = response['data'][0]['id']

        # Use the new FSRS review function
        request = QuizReviewRequest(card_id=card_id, rating=rating)
        result = review_quiz_card_fsrs(request)

        return result.success

    except Exception as e:
        logger.error(f"Failed to update quiz card (legacy): {e}")
        return False


# =====================================
# Paper Queue Operations
# =====================================

def add_to_paper_queue(pillar_id: str, papers: List[PaperRef], priority: int = 5) -> int:
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

    # Get existing paper IDs and queued paper IDs to avoid duplicates
    try:
        # Check both existing papers and queued papers
        papers_response = (client.table('papers')
                          .select('id')
                          .eq('pillar_id', pillar_id)
                          .execute())

        queue_response = (client.table('paper_queue')
                         .select('paper_id')
                         .eq('pillar_id', pillar_id)
                         .eq('processed', False)
                         .execute())

        if papers_response['error']:
            logger.error(f"Failed to get existing papers: {papers_response['error']}")
            raise Exception(f"Failed to add candidates to queue for pillar {pillar_id}: {papers_response['error']}")

        if queue_response['error']:
            logger.error(f"Failed to get queued papers: {queue_response['error']}")
            raise Exception(f"Failed to add candidates to queue for pillar {pillar_id}: {queue_response['error']}")

        existing_ids = {row['id'] for row in (papers_response['data'] or [])}
        queued_ids = {row['paper_id'] for row in (queue_response['data'] or [])}
        all_existing_ids = existing_ids | queued_ids  # Union of both sets

    except Exception as e:
        logger.error(f"Failed to check existing papers: {e}")
        raise Exception(f"Failed to add candidates to queue for pillar {pillar_id}: {e}")

    # Add papers that don't exist yet
    for paper in papers:
        if paper.id in all_existing_ids:
            logger.debug(f"Paper {paper.id} already exists or is queued for pillar {pillar_id}, skipping")
            continue

        # The queue is the last place a candidate can be rejected cheaply.
        # Past this point the only feedback is an ingest failure, which reads as
        # "the pillar failed today" rather than "this candidate was never
        # downloadable". So refuse anything that cannot produce a PDF URL.
        pdf_url = resolvable_pdf_url(paper.id, paper.url_pdf)
        if not pdf_url:
            logger.warning(
                f"Not queueing paper {paper.id!r} ({paper.title[:60]!r}) for pillar "
                f"{pillar_id}: it is neither an arXiv id nor accompanied by a PDF "
                f"URL, so ingest could never download it"
            )
            continue

        try:
            queue_data = {
                'paper_id': paper.id,
                'pillar_id': pillar_id,
                'title': paper.title,
                # Record what the id actually is instead of asserting 'arxiv'
                # for everything — pop_from_paper_queue trusts this field, and
                # a wrong 'arxiv' is what turned unresolvable ids into
                # fabricated https://arxiv.org/pdf/<id>.pdf URLs.
                'source': 'arxiv' if is_arxiv_id(paper.id) else 'url',
                # Carry the resolved URL so a non-arXiv paper survives the round
                # trip. Requires docs/migrations/008_paper_queue_url_pdf.sql on
                # databases created before that migration; see the retry below.
                'url_pdf': pdf_url,
                'priority': priority,
                'processed': False
            }

            response = client.table('paper_queue').insert(queue_data)

            if response['error'] and _is_missing_url_pdf_column(response['error']):
                logger.error(
                    "paper_queue has no url_pdf column — run "
                    "docs/migrations/008_paper_queue_url_pdf.sql. Falling back to "
                    "queueing without it; non-arXiv candidates will be dropped."
                )
                if not is_arxiv_id(paper.id):
                    continue
                queue_data.pop('url_pdf')
                response = client.table('paper_queue').insert(queue_data)

            if response['error']:
                if 'duplicate' in str(response['error']).lower():
                    logger.debug(f"Paper {paper.id} already in queue for pillar {pillar_id}")
                else:
                    logger.error(f"Failed to add paper {paper.id} to queue: {response['error']}")
            else:
                added += 1

        except Exception as e:
            logger.error(f"Failed to add paper {paper.id} to queue: {e}")

    logger.info(f"Added {added}/{len(papers)} papers to queue for pillar {pillar_id}")
    return added


def pop_from_paper_queue(pillar_id: str, limit: int = 1) -> List[PaperRef]:
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

        # Get unprocessed papers ordered by priority (high first), then oldest first
        response = (client.table('paper_queue')
                   .select('*')
                   .eq('pillar_id', pillar_id)
                   .eq('processed', False)
                   .order('priority.desc,added_at.asc')  # Fixed: proper composite ordering
                   .limit(limit)
                   .execute())

        if response['error']:
            logger.error(f"Failed to pop papers from queue: {response['error']}")
            raise Exception(f"Failed to pop papers from queue for pillar {pillar_id}: {response['error']}")

        if not response['data']:
            return []

        papers = []
        for row in response['data']:
            # Create PaperRef with full metadata from ArXiv if available
            paper = _fetch_full_paper_metadata(
                row['paper_id'],
                row['title'],
                row.get('source', 'arxiv'),
                url_pdf=row.get('url_pdf'),
            )
            papers.append(paper)

            # Mark as processed
            update_response = (client.table('paper_queue')
                             .eq('pillar_id', pillar_id)
                             .eq('paper_id', row['paper_id'])
                             .update({'processed': True}))

            if update_response['error']:
                logger.error(f"Failed to mark paper {row['paper_id']} as processed: {update_response['error']}")

        return papers

    except Exception as e:
        logger.error(f"Failed to pop papers from queue: {e}")
        raise Exception(f"Failed to pop papers from queue for pillar {pillar_id}: {e}")


def get_queue_size(pillar_id: str) -> int:
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
                   .eq('pillar_id', pillar_id)
                   .eq('processed', False)
                   .execute())

        if response['error']:
            logger.error(f"Failed to get queue size: {response['error']}")
            return 0

        return len(response['data'] or [])

    except Exception as e:
        logger.error(f"Failed to get queue size: {e}")
        return 0


def _fetch_full_paper_metadata(
    paper_id: str,
    title: str,
    source: str,
    url_pdf: Optional[str] = None,
) -> PaperRef:
    """
    Fetch full paper metadata from ArXiv for a given paper ID.

    Args:
        paper_id: Paper ID (arXiv ID)
        title: Paper title from queue
        source: Source of the paper ('arxiv', 'url', ...)
        url_pdf: PDF URL stored on the queue row, if any

    Returns:
        PaperRef with full metadata if available, or basic metadata as fallback
    """
    # `is_arxiv_id` and not just `source == 'arxiv'`: rows written before the
    # source field was made honest claim 'arxiv' for everything, and querying
    # the arXiv API for a non-arXiv id is a guaranteed miss.
    if source == 'arxiv' and is_arxiv_id(paper_id):
        try:
            # Import here to avoid circular dependencies
            from .tools.arxiv_tool import ArXivTool
            from .schemas import SearchQuery

            arxiv_tool = ArXivTool()

            # Search for the specific paper by ID
            search_query = SearchQuery(
                pillar_id="models-architectures",  # Default pillar for metadata fetch
                query=f"id:{paper_id}",
                filters={},
                max_results=1
            )

            results = arxiv_tool.search(search_query)

            if results:
                # Found full metadata
                paper = results[0]
                logger.info(f"Retrieved full metadata for paper {paper_id}: {len(paper.authors)} authors, year {paper.year}")
                return paper
            else:
                logger.warning(f"No metadata found for arXiv paper {paper_id}, using basic data")

        except Exception as e:
            logger.warning(f"Failed to fetch metadata for paper {paper_id}: {e}")

    # Fallback to basic paper data. The stored URL wins; an arXiv URL is only
    # constructed when the id really is an arXiv id. This line used to read
    # `if source == 'arxiv'` — and since every row claimed 'arxiv', it turned
    # ids like `searxng_078015` into https://arxiv.org/pdf/searxng_078015.pdf,
    # which 404s at ingest.
    pdf_url = resolvable_pdf_url(paper_id, url_pdf)
    if not pdf_url:
        logger.warning(
            f"Queue row for paper {paper_id!r} (source={source!r}) has no "
            f"resolvable PDF URL; ingest will reject it"
        )

    return PaperRef(
        id=paper_id,
        title=title,
        authors=[],  # Empty metadata as fallback
        venue=None,
        year=None,
        url_pdf=pdf_url,
        abstract=None,
        citation_count=0
    )


# =====================================
# Aliases for Orchestrator Compatibility
# =====================================

# Queue operations aliases
queue_add_candidates = add_to_paper_queue  # Alias for orchestrator
queue_pop_next = pop_from_paper_queue  # Alias for orchestrator

# Paper operations aliases
def upsert_paper(pillar_id: str, paper: PaperRef) -> bool:
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
def mark_processed(pillar_id: str, paper_id: str) -> bool:
    """Mark a paper as processed in the queue."""
    try:
        client = get_client()
        response = (client.table('paper_queue')
                   .eq('pillar_id', pillar_id)
                   .eq('paper_id', paper_id)
                   .update({'processed': True}))

        if response['error']:
            logger.error(f"Failed to mark paper {paper_id} as processed: {response['error']}")
            return False

        return True
    except Exception as e:
        logger.error(f"Failed to mark paper {paper_id} as processed: {e}")
        return False


# =====================================
# Pillar CRUD Operations
# =====================================

def _generate_slug(name: str) -> str:
    """Generate a URL-friendly slug from a pillar name."""
    import re
    # Convert to lowercase and replace spaces/special chars with hyphens
    slug = re.sub(r'[^\w\s-]', '', name.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')


def _pillar_to_dict(pillar: PillarCreate) -> Dict[str, Any]:
    """Convert PillarCreate to pillars table row."""
    slug = _generate_slug(pillar.name)
    return {
        'id': slug,
        'name': pillar.name,
        'goal': pillar.goal,
        'focus_areas': pillar.focus_areas,
        'papers_per_day': pillar.papers_per_day
    }


def _dict_to_pillar(row: Dict[str, Any]) -> Pillar:
    """Convert pillars table row to Pillar."""
    return Pillar(
        id=row['id'],
        name=row['name'],
        goal=row['goal'],
        focus_areas=row.get('focus_areas', []),
        papers_per_day=row.get('papers_per_day', 2),
        created_at=datetime.fromisoformat(row['created_at']) if row.get('created_at') else datetime.now(timezone.utc),
        updated_at=datetime.fromisoformat(row['updated_at']) if row.get('updated_at') else datetime.now(timezone.utc),
        last_active=datetime.fromisoformat(row['last_active']) if row.get('last_active') else None
    )


class PillarCreateError(Exception):
    """Custom exception for pillar creation errors."""
    pass

class PillarExistsError(PillarCreateError):
    """Exception raised when pillar already exists."""
    pass

def create_pillar(pillar_data: PillarCreate) -> Pillar:
    """
    Create a new pillar.
    
    Args:
        pillar_data: Data for the new pillar
        
    Returns:
        Created pillar
        
    Raises:
        PillarExistsError: If pillar with same name already exists
        PillarCreateError: If creation fails for other reasons
    """
    try:
        client = get_client()
        data = _pillar_to_dict(pillar_data)

        response = client.table('pillars').insert(data)

        if response['error']:
            error_info = response['error']
            if isinstance(error_info, dict) and (error_info.get('code') == 409 or 'duplicate' in str(error_info).lower()):
                logger.error(f"Pillar with name '{pillar_data.name}' already exists")
                raise PillarExistsError(f"Pillar with name '{pillar_data.name}' already exists")
            logger.error(f"Failed to create pillar: {response['error']}")
            raise PillarCreateError(f"Failed to create pillar: {response['error']}")

        if response['data']:
            logger.info(f"Created pillar: {pillar_data.name} (id: {data['id']})")
            return _dict_to_pillar(response['data'][0])

        raise PillarCreateError("No data returned from database")

    except (PillarCreateError, PillarExistsError):
        raise
    except Exception as e:
        logger.error(f"Failed to create pillar: {e}")
        raise PillarCreateError(f"Failed to create pillar: {e}")


class PillarLookupError(Exception):
    """The pillar list could not be read from the database.

    Distinct from "there are no pillars", which is an empty list. Collapsing the two
    is why several callers' fallbacks were dead code: `cli.get_valid_pillars()` and
    `scheduler.run_daily_pipeline()` both wrap the call in `except Exception` and both
    could never fire, so an unreachable database made every pillar look invalid and
    made the scheduler report "no pillars in the database, nothing to do. Seed them
    with create_pillars.py" — advice that is actively wrong when the database is
    simply down.
    """


def get_pillars(limit: int = 50) -> List[Pillar]:
    """
    Get all pillars.

    Args:
        limit: Maximum number of pillars to return

    Returns:
        List of pillars. Empty means the table is empty — nothing else.

    Raises:
        PillarLookupError: the database could not be read. Callers that would rather
            degrade than fail should use get_pillars_or_empty(), which is explicit
            about the trade rather than hiding it in a return value.

    """
    try:
        client = get_client()
        response = (client.table('pillars')
                   .select('*')
                   .order('created_at', desc=False)
                   .limit(limit)
                   .execute())

        if response['error']:
            logger.error(f"Failed to get pillars: {response['error']}")
            raise PillarLookupError(
                f"Could not read the pillar list: {response['error']}"
            )

        if not response['data']:
            return []

        return [_dict_to_pillar(row) for row in response['data']]

    except PillarLookupError:
        raise
    except Exception as e:
        logger.error(f"Failed to get pillars: {e}")
        raise PillarLookupError(f"Could not read the pillar list: {e}") from e


def get_pillars_or_empty(limit: int = 50) -> List[Pillar]:
    """Pillars, or an empty list if the database could not be read.

    For the page routers, which render a pillar dropdown and would rather show an
    empty one than a 500. Separate from get_pillars() so the choice to swallow the
    failure is visible at the call site instead of being the silent default for
    everybody — including the CLI and the scheduler, which genuinely need to know.
    """
    try:
        return get_pillars(limit=limit)
    except PillarLookupError as e:
        logger.error(f"Falling back to an empty pillar list: {e}")
        return []


def get_pillar_by_id(pillar_id: str) -> Optional[Pillar]:
    """
    Get a pillar by its ID.
    
    Args:
        pillar_id: The pillar ID (slug)
        
    Returns:
        Pillar or None if not found
    """
    try:
        client = get_client()
        response = (client.table('pillars')
                   .select('*')
                   .eq('id', pillar_id)
                   .execute())

        if response['error']:
            logger.error(f"Failed to get pillar {pillar_id}: {response['error']}")
            return None

        if not response['data']:
            return None

        return _dict_to_pillar(response['data'][0])

    except Exception as e:
        logger.error(f"Failed to get pillar {pillar_id}: {e}")
        return None


def update_pillar(pillar_id: str, pillar_data: PillarUpdate) -> Optional[Pillar]:
    """
    Update an existing pillar.
    
    Args:
        pillar_id: The pillar ID to update
        pillar_data: Updated pillar data
        
    Returns:
        Updated pillar or None if failed
    """
    try:
        client = get_client()

        # Build update data from non-None fields
        update_data = {}
        if pillar_data.name is not None:
            update_data['name'] = pillar_data.name
            # Regenerate slug if name changes
            update_data['id'] = _generate_slug(pillar_data.name)
        if pillar_data.goal is not None:
            update_data['goal'] = pillar_data.goal
        if pillar_data.focus_areas is not None:
            update_data['focus_areas'] = pillar_data.focus_areas
        if pillar_data.papers_per_day is not None:
            update_data['papers_per_day'] = pillar_data.papers_per_day

        if not update_data:
            # No updates to apply
            return get_pillar_by_id(pillar_id)

        response = (client.table('pillars')
                   .eq('id', pillar_id)
                   .update(update_data))

        if response['error']:
            logger.error(f"Failed to update pillar {pillar_id}: {response['error']}")
            return None

        if response['data']:
            logger.info(f"Updated pillar: {pillar_id}")
            return _dict_to_pillar(response['data'][0])
        return None

    except Exception as e:
        logger.error(f"Failed to update pillar {pillar_id}: {e}")
        return None


def delete_pillar(pillar_id: str) -> bool:
    """
    Delete a pillar.
    
    Args:
        pillar_id: The pillar ID to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_client()

        # Check if pillar has associated data
        papers_response = (client.table('papers')
                          .select('id')
                          .eq('pillar_id', pillar_id)
                          .limit(1)
                          .execute())

        if papers_response['data']:
            logger.warning(f"Cannot delete pillar {pillar_id}: has associated papers")
            return False

        response = (client.table('pillars')
                   .eq('id', pillar_id)
                   .delete())

        if response.get('error'):
            logger.error(f"Failed to delete pillar {pillar_id}: {response['error']}")
            return False

        logger.info(f"Deleted pillar: {pillar_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to delete pillar {pillar_id}: {e}")
        return False


def pillar_exists(pillar_id: str) -> bool:
    """
    Check if a pillar exists.
    
    Args:
        pillar_id: The pillar ID to check
        
    Returns:
        True if pillar exists, False otherwise
    """
    try:
        client = get_client()
        response = (client.table('pillars')
                   .select('id')
                   .eq('id', pillar_id)
                   .execute())

        if response['error']:
            logger.error(f"Failed to check pillar {pillar_id}: {response['error']}")
            return False

        return bool(response['data'])

    except Exception as e:
        logger.error(f"Failed to check pillar {pillar_id}: {e}")
        return False


# =====================================
# FSRS Operations
# =====================================

def get_user_fsrs_parameters(user_id: str = "default_user", pillar_id: Optional[str] = None) -> Optional[UserFSRSParameters]:
    """
    Get FSRS parameters for a user, with pillar-specific fallback to global.
    
    Args:
        user_id: User identifier
        pillar_id: Optional pillar-specific parameters
        
    Returns:
        UserFSRSParameters or None if not found
    """
    try:
        client = get_client()

        # First try pillar-specific parameters if pillar_id provided
        if pillar_id:
            response = (client.table('user_fsrs_parameters')
                       .select('*')
                       .eq('user_id', user_id)
                       .eq('pillar_id', pillar_id)
                       .limit(1)
                       .execute())

            if not response['error'] and response['data']:
                return _dict_to_user_fsrs_parameters(response['data'][0])

        # Fallback to global parameters
        response = (client.table('user_fsrs_parameters')
                   .select('*')
                   .eq('user_id', user_id)
                   .eq('pillar_id', None)
                   .limit(1)
                   .execute())

        if response['error'] or not response['data']:
            logger.debug(f"No FSRS parameters found for user {user_id}, using defaults")
            return None

        return _dict_to_user_fsrs_parameters(response['data'][0])

    except Exception as e:
        logger.error(f"Failed to get FSRS parameters for user {user_id}: {e}")
        return None


def upsert_user_fsrs_parameters(params: UserFSRSParameters) -> bool:
    """
    Insert or update FSRS parameters for a user.
    
    Args:
        params: UserFSRSParameters to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_client()
        data = _user_fsrs_parameters_to_dict(params)

        # Try to update first
        response = (client.table('user_fsrs_parameters')
                   .eq('user_id', params.user_id)
                   .eq('pillar_id', params.pillar_id if params.pillar_id else None)
                   .update(data))

        if response['error']:
            # If update failed, try insert
            response = client.table('user_fsrs_parameters').insert(data)

            if response['error']:
                logger.error(f"Failed to upsert FSRS parameters: {response['error']}")
                return False

        logger.info(f"Upserted FSRS parameters for user {params.user_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to upsert FSRS parameters: {e}")
        return False


def get_review_logs(user_id: str = "default_user", pillar_id: Optional[str] = None,
                   limit: int = 1000) -> List[ReviewLog]:
    """
    Get review logs for optimization.
    
    Args:
        user_id: User identifier
        pillar_id: Optional pillar filter
        limit: Maximum number of logs to return
        
    Returns:
        List of ReviewLog entries
    """
    try:
        client = get_client()

        query = (client.table('review_logs')
                .select('*')
                .eq('user_id', user_id)
                .order('review_timestamp', desc=False)
                .limit(limit))

        if pillar_id:
            query = query.eq('pillar_id', pillar_id)

        response = query.execute()

        if response['error']:
            logger.error(f"Failed to get review logs: {response['error']}")
            return []

        if not response['data']:
            return []

        return [_dict_to_review_log(row) for row in response['data']]

    except Exception as e:
        logger.error(f"Failed to get review logs: {e}")
        return []


def get_cards_for_review(user_id: str = "default_user", pillar_id: Optional[str] = None,
                        limit: int = 10) -> List[QuizCard]:
    """
    Get quiz cards due for review using FSRS scheduling.
    
    Args:
        user_id: User identifier
        pillar_id: Optional pillar filter
        limit: Maximum number of cards to return
        
    Returns:
        List of QuizCard due for review
    """
    try:
        client = get_client()
        now = datetime.now(timezone.utc).isoformat()

        # Build query for cards due for review
        query = (client.table('quiz_cards')
                .select('*')
                .eq('user_id', user_id)
                .limit(limit))

        if pillar_id:
            query = query.eq('pillar_id', pillar_id)

        # For now, get cards where next_review_date is null or <= now
        # Note: PostgREST doesn't support complex date comparisons easily,
        # so we'll do this filtering in Python
        response = query.execute()

        if response['error']:
            logger.error(f"Failed to get cards for review: {response['error']}")
            return []

        if not response['data']:
            return []

        # Filter and sort cards
        cards = []
        for row in response['data']:
            card = _dict_to_quiz_card(row)

            # Include card if:
            # 1. Never reviewed (next_review_date is None)
            # 2. Due for review (next_review_date <= now)
            if card.next_review_date is None or card.next_review_date <= datetime.now(timezone.utc):
                cards.append(card)

        # Sort by due date (earliest first, None last)
        cards.sort(key=lambda c: c.next_review_date or datetime.max.replace(tzinfo=timezone.utc))

        return cards[:limit]

    except Exception as e:
        logger.error(f"Failed to get cards for review: {e}")
        return []


def add_review_log(log: ReviewLog) -> bool:
    """
    Add a review log entry.
    
    Args:
        log: ReviewLog to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_client()
        data = _review_log_to_dict(log)

        response = client.table('review_logs').insert(data)

        if response['error']:
            logger.error(f"Failed to add review log: {response['error']}")
            return False

        logger.debug(f"Added review log for card {log.card_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to add review log: {e}")
        return False


# =====================================
# Paper Citations Operations
# =====================================

def _paper_citation_to_dict(citation: PaperCitation) -> Dict[str, Any]:
    """Convert PaperCitation to paper_citations table row."""
    data = {
        'paper_id': citation.paper_id,
        'cited_paper_id': citation.cited_paper_id,
        'citation_direction': citation.citation_direction,
        'is_influential': citation.is_influential,
        'citation_context': citation.citation_context,
        'source': citation.source,
        'fetched_at': citation.fetched_at.isoformat() if citation.fetched_at else datetime.now(timezone.utc).isoformat()
    }
    # Remove None values
    return {k: v for k, v in data.items() if v is not None}


def _dict_to_paper_citation(row: Dict[str, Any]) -> PaperCitation:
    """Convert paper_citations table row to PaperCitation."""
    return PaperCitation(
        id=row.get('id'),
        paper_id=row['paper_id'],
        cited_paper_id=row['cited_paper_id'],
        citation_direction=row['citation_direction'],
        is_influential=row.get('is_influential', False),
        citation_context=row.get('citation_context'),
        source=row.get('source', 'semantic_scholar'),
        fetched_at=datetime.fromisoformat(row['fetched_at']) if row.get('fetched_at') else None
    )


def add_paper_citations(citations: List[PaperCitation]) -> int:
    """
    Add multiple paper citations to the database.

    Args:
        citations: List of PaperCitation objects to add

    Returns:
        Number of citations successfully added
    """
    if not citations:
        return 0

    try:
        client = get_client()
        added = 0

        for citation in citations:
            data = _paper_citation_to_dict(citation)

            response = client.table('paper_citations').insert(data)

            if response['error']:
                error_str = str(response['error']).lower()
                # Skip duplicates silently
                if 'duplicate' in error_str or 'conflict' in error_str or 'unique' in error_str:
                    continue
                logger.warning(f"Failed to add citation: {response['error']}")
                continue

            added += 1

        logger.info(f"Added {added} citations out of {len(citations)}")
        return added

    except Exception as e:
        logger.error(f"Failed to add paper citations: {e}")
        return 0


def get_citations_for_paper(paper_id: str, direction: str = None) -> List[PaperCitation]:
    """
    Get citations for a paper.

    Args:
        paper_id: Paper ID to get citations for
        direction: Optional filter for 'outgoing' or 'incoming'

    Returns:
        List of PaperCitation objects
    """
    try:
        client = get_client()
        query = client.table('paper_citations').select('*').eq('paper_id', paper_id)

        if direction:
            query = query.eq('citation_direction', direction)

        response = query.execute()

        if response['error']:
            logger.error(f"Failed to get citations: {response['error']}")
            return []

        citations = [_dict_to_paper_citation(row) for row in response['data']]
        logger.debug(f"Found {len(citations)} citations for paper {paper_id}")
        return citations

    except Exception as e:
        logger.error(f"Failed to get citations for paper {paper_id}: {e}")
        return []


def get_co_cited_papers(paper_ids: List[str], limit: int = 10) -> List[str]:
    """
    Get papers that are frequently cited together with the given papers.

    Args:
        paper_ids: List of paper IDs to find co-cited papers for
        limit: Maximum number of results

    Returns:
        List of paper IDs that are frequently co-cited
    """
    if not paper_ids:
        return []

    try:
        client = get_client()

        # Get all citations for the given papers
        all_cited = []
        for paper_id in paper_ids:
            citations = get_citations_for_paper(paper_id, direction='outgoing')
            all_cited.extend([c.cited_paper_id for c in citations])

        # Count occurrences
        from collections import Counter
        cited_counts = Counter(all_cited)

        # Exclude the input papers
        for paper_id in paper_ids:
            cited_counts.pop(paper_id, None)

        # Return most common
        return [paper_id for paper_id, count in cited_counts.most_common(limit)]

    except Exception as e:
        logger.error(f"Failed to get co-cited papers: {e}")
        return []


def citation_exists(paper_id: str, cited_paper_id: str, direction: str) -> bool:
    """
    Check if a citation relationship already exists.

    Args:
        paper_id: Source paper ID
        cited_paper_id: Target paper ID
        direction: Citation direction ('outgoing' or 'incoming')

    Returns:
        True if citation exists, False otherwise
    """
    try:
        client = get_client()

        response = client.table('paper_citations') \
            .select('id') \
            .eq('paper_id', paper_id) \
            .eq('cited_paper_id', cited_paper_id) \
            .eq('citation_direction', direction) \
            .execute()

        if response['error']:
            return False

        return len(response['data']) > 0

    except Exception as e:
        logger.error(f"Failed to check citation existence: {e}")
        return False


def get_citation_network_papers(paper_ids: List[str], limit: int = 20) -> List[str]:
    """
    Get paper IDs from the citation network of the given papers.

    Finds papers that cite or are cited by the given papers.

    Args:
        paper_ids: Seed paper IDs
        limit: Maximum number of results

    Returns:
        List of paper IDs in the citation network
    """
    if not paper_ids:
        return []

    try:
        client = get_client()
        network_papers = set()

        for paper_id in paper_ids:
            # Get papers this one cites (outgoing)
            outgoing = get_citations_for_paper(paper_id, direction='outgoing')
            for citation in outgoing:
                network_papers.add(citation.cited_paper_id)

            # Get papers that cite this one (incoming)
            incoming = get_citations_for_paper(paper_id, direction='incoming')
            for citation in incoming:
                network_papers.add(citation.cited_paper_id)

        # Exclude seed papers
        network_papers -= set(paper_ids)

        return list(network_papers)[:limit]

    except Exception as e:
        logger.error(f"Failed to get citation network: {e}")
        return []


# =====================================
# Pipeline Run Tracking (migration 009)
#
# These are the WRITE side of run tracking and they are deliberately synchronous:
# they are called from the worker thread that runs the pipeline, where there is no
# event loop. Reads for the polling endpoint go through the async client in
# webui/services/postgrest_client.py instead. Do not mix the two.
#
# httpx.Client is thread-safe and designed to be shared, so the module-global
# singleton behind get_client() is fine to use from both the request thread and the
# worker thread.
# =====================================


class PipelineRunCreateError(Exception):
    """A pipeline run could not be created, for a reason that is not a conflict.

    Deliberately distinct from create_pipeline_run() returning None. None means the
    partial unique index refused a second active run for the pillar, which is a normal
    outcome the caller reports as HTTP 409. This means something is actually broken —
    the table is missing, the grant is wrong, PostgREST is down — and must not be
    reported to the user as "a run is already in progress".
    """


def _dict_to_pipeline_run_stage(data: Dict[str, Any]) -> PipelineRunStage:
    """Convert a pipeline_run_stages row to a PipelineRunStage."""
    return PipelineRunStage(
        id=data.get('id'),
        run_id=data['run_id'],
        seq=data['seq'],
        name=data['name'],
        status=data.get('status', StageStatus.PENDING.value),
        detail=data.get('detail'),
        started_at=data.get('started_at'),
        finished_at=data.get('finished_at'),
    )


def _dict_to_pipeline_run(data: Dict[str, Any]) -> PipelineRun:
    """Convert a pipeline_runs row to a PipelineRun.

    Accepts an embedded ``pipeline_run_stages`` list if PostgREST resource embedding
    was used, and sorts it by ``seq`` so callers never have to.
    """
    raw_stages = data.get('pipeline_run_stages') or []
    stages = sorted(
        (_dict_to_pipeline_run_stage(s) for s in raw_stages),
        key=lambda s: s.seq,
    )
    return PipelineRun(
        id=data['id'],
        pillar_id=data['pillar_id'],
        trigger_source=data['trigger_source'],
        kind=data['kind'],
        status=data.get('status', RunStatus.PENDING.value),
        current_stage=data.get('current_stage'),
        papers_processed=data.get('papers_processed', 0),
        papers_failed=data.get('papers_failed', 0),
        error=data.get('error'),
        created_at=data.get('created_at'),
        started_at=data.get('started_at'),
        finished_at=data.get('finished_at'),
        heartbeat_at=data.get('heartbeat_at'),
        stages=stages,
        result=data.get('result'),
    )


def create_pipeline_run(
    pillar_id: str,
    trigger_source: str,
    kind: str,
    stage_names: List[str],
) -> Optional[PipelineRun]:
    """Create a run row plus its seeded stage rows.

    Returns None — rather than raising — when a run is already active for this
    pillar. That case is not an error: it is the `pipeline_runs_one_active_per_pillar`
    partial unique index doing its job, and PostgREST reports it as HTTP 409 with
    Postgres code 23505. The caller turns None into a 409 response.

    Args:
        pillar_id: Target pillar slug.
        trigger_source: 'ui_pipeline' | 'ui_select' | 'scheduler'.
        kind: 'run_daily' | 'process_selected'.
        stage_names: Ordered stage names; index + 1 becomes the stored ``seq``.

    Returns:
        The created PipelineRun with its seeded stages, or None if — and only if — a
        run is already active for this pillar.

    Raises:
        PipelineRunCreateError: creation failed for any other reason (missing table,
            permission denied, PostgREST unreachable). This used to return None like
            the conflict case, and the caller turned every None into HTTP 409, so a
            broken database told the user "a run is already in progress" — a
            comfortable lie that sent them looking for a run that did not exist.

    """
    try:
        client = get_client()
        response = client.table('pipeline_runs').insert({
            'pillar_id': pillar_id,
            'trigger_source': trigger_source,
            'kind': kind,
            'status': RunStatus.PENDING.value,
        })

        if response['error']:
            error_info = response['error']
            # 409 from TableQuery.insert, or 23505 straight from Postgres.
            if isinstance(error_info, dict) and (
                error_info.get('code') in (409, '409', '23505')
                or 'one_active_per_pillar' in str(error_info)
            ):
                logger.info(
                    f"Not starting a run for pillar {pillar_id}: one is already active"
                )
                return None
            logger.error(f"Failed to create pipeline run: {error_info}")
            raise PipelineRunCreateError(
                f"Could not create a pipeline run for {pillar_id}: {error_info}"
            )

        if not response['data']:
            logger.error("Pipeline run insert returned no row")
            raise PipelineRunCreateError(
                f"Could not create a pipeline run for {pillar_id}: "
                "the insert returned no row"
            )

        run = _dict_to_pipeline_run(response['data'][0])

        # Seed the stage skeleton so the UI can render every step as pending
        # immediately, instead of rows appearing one at a time.
        stage_rows = [
            {
                'run_id': run.id,
                'seq': i + 1,
                'name': name,
                'status': StageStatus.PENDING.value,
            }
            for i, name in enumerate(stage_names)
        ]
        # PostgREST accepts a JSON array for a bulk insert. TableQuery.insert types
        # its argument as a dict, but it passes the value straight to httpx `json=`,
        # so a list works and is one round trip instead of eleven.
        stages_response = client.table('pipeline_run_stages').insert(stage_rows)
        if stages_response['error']:
            logger.error(
                f"Created run {run.id} but failed to seed stages: "
                f"{stages_response['error']}"
            )
        elif stages_response['data']:
            run.stages = sorted(
                (_dict_to_pipeline_run_stage(s) for s in stages_response['data']),
                key=lambda s: s.seq,
            )

        logger.info(
            f"Created pipeline run {run.id} for pillar {pillar_id} "
            f"({kind}, via {trigger_source})"
        )
        return run

    except PipelineRunCreateError:
        # Already logged and already carries its reason — must stay above the broad
        # clause below, which would otherwise flatten it back into None.
        raise
    except Exception as e:
        logger.error(f"Failed to create pipeline run for pillar {pillar_id}: {e}")
        raise PipelineRunCreateError(
            f"Could not create a pipeline run for {pillar_id}: {e}"
        ) from e


def start_pipeline_run(run_id: str) -> bool:
    """Mark a run as running. Called from the worker thread as it picks the job up."""
    now = datetime.now(timezone.utc).isoformat()
    return _update_pipeline_run(run_id, {
        'status': RunStatus.RUNNING.value,
        'started_at': now,
        'heartbeat_at': now,
    })


def update_pipeline_run_stage(
    run_id: str,
    stage_name: str,
    status: str,
    detail: Optional[str] = None,
) -> bool:
    """Advance one stage of a run, and refresh the run's current_stage/heartbeat.

    Two writes on purpose: the stage row carries per-stage timing, and the
    denormalised ``current_stage`` on the run row means the poll endpoint can answer
    "where is it" without inspecting the stage list.

    Args:
        run_id: The run.
        stage_name: A StageName value.
        status: A StageStatus value.
        detail: Optional free text, e.g. "2/5 papers".

    """
    now = datetime.now(timezone.utc).isoformat()

    stage_patch: Dict[str, Any] = {'status': status}
    if detail is not None:
        stage_patch['detail'] = detail
    if status == StageStatus.RUNNING.value:
        stage_patch['started_at'] = now
    elif status in (StageStatus.COMPLETED.value, StageStatus.FAILED.value,
                    StageStatus.SKIPPED.value):
        stage_patch['finished_at'] = now

    ok = True
    try:
        client = get_client()
        # TableQuery.update() calls raise_for_status() WITHOUT catching it, unlike
        # insert()/execute(). Hence the try/except around every update in this module.
        response = (client.table('pipeline_run_stages')
                    .eq('run_id', run_id)
                    .eq('name', stage_name)
                    .update(stage_patch))
        if response['error']:
            logger.error(
                f"Failed to update stage {stage_name} of run {run_id}: "
                f"{response['error']}"
            )
            ok = False
    except Exception as e:
        logger.error(f"Failed to update stage {stage_name} of run {run_id}: {e}")
        ok = False

    run_patch: Dict[str, Any] = {'heartbeat_at': now}
    if status == StageStatus.RUNNING.value:
        run_patch['current_stage'] = stage_name
    if not _update_pipeline_run(run_id, run_patch):
        ok = False

    return ok


def delete_pipeline_run_stage(run_id: str, stage_name: str) -> bool:
    """Remove a seeded stage row for work the run has decided not to do.

    Stage rows are seeded from a fixed per-kind list when the run is created, before
    anything is known about the pillar. Discovery's citation step is conditional on
    the pillar having recent papers to follow, and that is only known once the run has
    started — so the row is created and then taken away, rather than left pending
    forever on a step that will never run. `seq` is left with a gap, which is fine:
    ordering reads seq, it does not count it.
    """
    try:
        client = get_client()
        response = (client.table('pipeline_run_stages')
                    .eq('run_id', run_id)
                    .eq('name', stage_name)
                    .delete())
        if response['error']:
            logger.error(
                f"Failed to delete stage {stage_name} of run {run_id}: "
                f"{response['error']}"
            )
            return False
        return True
    except Exception as e:
        # Cosmetic: a leftover pending row is a blemish, not a reason to lose a run.
        logger.error(f"Failed to delete stage {stage_name} of run {run_id}: {e}")
        return False


def finish_pipeline_run(
    run_id: str,
    status: str,
    papers_processed: int = 0,
    papers_failed: int = 0,
    error: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
) -> bool:
    """Write a terminal status onto a run.

    Args:
        run_id: The run to close out.
        status: A terminal RunStatus value — succeeded, failed, cancelled or
            interrupted. Passing a non-terminal status here is a bug; polling
            clients stop only on a terminal one.
        papers_processed: Count for the run row.
        papers_failed: Count for the run row.
        error: Optional message, truncated to 2000 chars on the row.
        result: Optional terminal payload for run kinds that produce one — today only
            a `discover` run, which stores the candidate list the user has to choose
            from. Written once here rather than incrementally, which is why it can be
            a JSONB column when the stage list could not be (PostgREST cannot
            partially update JSONB; see migration 009).

    """
    patch: Dict[str, Any] = {
        'status': status,
        'papers_processed': papers_processed,
        'papers_failed': papers_failed,
        'finished_at': datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        # Keep the row readable; the full traceback is in the logs.
        patch['error'] = error[:2000]
    if result is not None:
        patch['result'] = result
    return _update_pipeline_run(run_id, patch)


def get_pipeline_run(run_id: str) -> Optional[PipelineRun]:
    """Read a run and its stages. Synchronous — for the sweep and for tests.

    The polling endpoint uses the async client instead; see the module comment.
    """
    try:
        client = get_client()
        response = (client.table('pipeline_runs')
                    .select('*,pipeline_run_stages(*)')
                    .eq('id', run_id)
                    .execute())
        if response['error']:
            logger.error(f"Failed to read pipeline run {run_id}: {response['error']}")
            return None
        if not response['data']:
            return None
        return _dict_to_pipeline_run(response['data'][0])
    except Exception as e:
        logger.error(f"Failed to read pipeline run {run_id}: {e}")
        return None


def get_active_pipeline_run(pillar_id: str) -> Optional[PipelineRun]:
    """Return the pending/running run for a pillar, if there is one.

    Filters on ``status``, never on "finished_at IS NULL": TableQuery.eq() renders
    Python None as the string "None", so a null filter silently matches nothing.
    """
    try:
        client = get_client()
        query = client.table('pipeline_runs').select('*,pipeline_run_stages(*)')
        query.params['pillar_id'] = f'eq.{pillar_id}'
        query.params['status'] = 'in.(pending,running)'
        response = query.limit(1).execute()
        if response['error']:
            logger.error(
                f"Failed to look up active run for pillar {pillar_id}: "
                f"{response['error']}"
            )
            return None
        if not response['data']:
            return None
        return _dict_to_pipeline_run(response['data'][0])
    except Exception as e:
        logger.error(f"Failed to look up active run for pillar {pillar_id}: {e}")
        return None


def sweep_interrupted_runs(trigger_sources: Optional[List[str]] = None) -> int:
    """Mark still-open runs as interrupted. Call once at application startup.

    A process that is killed cannot record its own death, so without this a crashed
    run stays 'running' forever and is indistinguishable from a live one. The status
    is 'interrupted' rather than 'failed' precisely because we do not know what
    happened — only that nobody is working on it any more.

    Args:
        trigger_sources: Restrict the sweep to runs this process could have owned.
            This matters because webui and the scheduler are SEPARATE containers
            sharing one database: an unscoped sweep on webui startup would declare a
            perfectly healthy scheduler run dead. webui passes its two UI sources,
            the scheduler passes its own. None sweeps everything, which is only
            correct when nothing else can be running.

    Returns:
        The number of rows swept, or 0 on failure (logged).

    """
    try:
        client = get_client()
        query = client.table('pipeline_runs')
        query.params['status'] = 'in.(pending,running)'
        if trigger_sources:
            joined = ','.join(trigger_sources)
            query.params['trigger_source'] = f'in.({joined})'
        response = query.update({
            'status': RunStatus.INTERRUPTED.value,
            'finished_at': datetime.now(timezone.utc).isoformat(),
            'error': 'Process exited before this run finished',
        })
        if response['error']:
            logger.error(f"Failed to sweep interrupted runs: {response['error']}")
            return 0
        count = len(response['data'] or [])
        if count:
            logger.warning(
                f"Swept {count} pipeline run(s) left open by a previous process"
            )
        return count
    except Exception as e:
        logger.error(f"Failed to sweep interrupted runs: {e}")
        return 0


def _update_pipeline_run(run_id: str, patch: Dict[str, Any]) -> bool:
    """PATCH one pipeline_runs row, swallowing and logging any failure.

    Progress bookkeeping must never take the pipeline down with it, so this returns
    a bool rather than raising.
    """
    try:
        client = get_client()
        response = (client.table('pipeline_runs')
                    .eq('id', run_id)
                    .update(patch))
        if response['error']:
            logger.error(f"Failed to update pipeline run {run_id}: {response['error']}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to update pipeline run {run_id}: {e}")
        return False
