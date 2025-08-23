"""
Comprehensive tests for Supabase DAO layer.
All Supabase calls are mocked for fast, reliable testing.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from uuid import uuid4

from nlp_pillars.schemas import PaperRef, PaperNote, Lesson, QuizCard, PillarID, DifficultyLevel, QuestionType
from nlp_pillars.db import (
    get_client, set_client,
    upsert_paper, mark_processed, insert_note, insert_lesson, insert_quiz_cards,
    get_recent_notes, queue_add_candidates, queue_pop_next,
    _paper_ref_to_dict, _dict_to_paper_ref, _paper_note_to_dict, _dict_to_paper_note,
    _lesson_to_dict, _dict_to_lesson, _quiz_card_to_dict, _dict_to_quiz_card
)


# Test fixtures
@pytest.fixture
def sample_paper_ref():
    """Sample paper reference for testing."""
    return PaperRef(
        id="test.12345",
        title="Test Paper: Advanced Techniques",
        authors=["Dr. Test", "Prof. Example"],
        venue="Test Conference",
        year=2023,
        url_pdf="https://example.com/test.pdf",
        abstract="This is a test paper abstract.",
        citation_count=42
    )


@pytest.fixture
def sample_paper_note():
    """Sample paper note for testing."""
    return PaperNote(
        paper_id="test.12345",
        pillar_id=PillarID.P2,
        problem="Limited effectiveness of traditional methods",
        method="Novel approach using advanced techniques",
        findings=["Achieved 95% accuracy", "Reduced processing time by 50%"],
        limitations=["Requires large datasets", "High computational cost"],
        future_work=["Explore real-time applications", "Optimize performance"],
        key_terms=["machine learning", "optimization", "algorithm"],
        related_papers=["related.123", "related.456"],
        confidence_score=0.9
    )


@pytest.fixture
def sample_lesson():
    """Sample lesson for testing."""
    return Lesson(
        paper_id="test.12345",
        pillar_id=PillarID.P2,
        tl_dr="Novel approach achieves significant improvements in accuracy and speed.",
        takeaways=[
            "Advanced techniques can improve accuracy significantly",
            "Optimization reduces processing time substantially",
            "Method scales well with dataset size"
        ],
        practice_ideas=[
            "Implement the algorithm in your own project",
            "Compare with traditional methods"
        ],
        connections=["related.123: Similar optimization approach"],
        difficulty=DifficultyLevel.MEDIUM,
        estimated_time=15
    )


@pytest.fixture
def sample_quiz_cards():
    """Sample quiz cards for testing."""
    return [
        QuizCard(
            paper_id="test.12345",
            pillar_id=PillarID.P2,
            question="What accuracy did the novel approach achieve?",
            answer="95% accuracy",
            difficulty=DifficultyLevel.EASY,
            question_type=QuestionType.FACTUAL,
            interval=1,
            repetitions=0,
            ease_factor=2.5,
            due_date=datetime(2023, 12, 1, 12, 0, 0)
        ),
        QuizCard(
            paper_id="test.12345",
            pillar_id=PillarID.P2,
            question="How does the novel approach improve upon traditional methods?",
            answer="It reduces processing time by 50% while maintaining high accuracy.",
            difficulty=DifficultyLevel.MEDIUM,
            question_type=QuestionType.CONCEPTUAL,
            interval=3,
            repetitions=1,
            ease_factor=2.6,
            due_date=datetime(2023, 12, 5, 12, 0, 0)
        )
    ]


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for testing."""
    mock_client = Mock()
    
    # Mock table method
    mock_table = Mock()
    mock_client.table.return_value = mock_table
    
    # Mock query methods
    mock_table.select.return_value = mock_table
    mock_table.insert.return_value = mock_table
    mock_table.upsert.return_value = mock_table
    mock_table.update.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.in_.return_value = mock_table
    mock_table.order.return_value = mock_table
    mock_table.limit.return_value = mock_table
    
    return mock_client


@pytest.fixture(autouse=True)
def setup_mock_client(mock_supabase_client):
    """Setup mock client for all tests."""
    set_client(mock_supabase_client)
    yield
    set_client(None)


class TestClientBootstrap:
    """Test client initialization and configuration."""
    
    @patch('nlp_pillars.db.create_client')
    @patch.dict('os.environ', {'SUPABASE_URL': 'test_url', 'SUPABASE_KEY': 'test_key'})
    def test_get_client_success(self, mock_create_client):
        """Test successful client creation with environment variables."""
        # Reset singleton
        set_client(None)
        
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        result = get_client()
        
        assert result == mock_client
        mock_create_client.assert_called_once_with('test_url', 'test_key')
    
    @patch.dict('os.environ', {}, clear=True)
    def test_get_client_missing_url(self):
        """Test client creation failure when SUPABASE_URL is missing."""
        # Reset singleton
        set_client(None)
        
        with pytest.raises(ValueError, match="SUPABASE_URL environment variable is required"):
            get_client()
    
    @patch.dict('os.environ', {'SUPABASE_URL': 'test_url'}, clear=True)
    def test_get_client_missing_key(self):
        """Test client creation failure when SUPABASE_KEY is missing."""
        # Reset singleton
        set_client(None)
        
        with pytest.raises(ValueError, match="SUPABASE_KEY environment variable is required"):
            get_client()


class TestSerializationHelpers:
    """Test Pydantic to dict conversions and vice versa."""
    
    def test_paper_ref_serialization(self, sample_paper_ref):
        """Test PaperRef to dict and back conversion."""
        pillar_id = PillarID.P2
        
        # Convert to dict
        paper_dict = _paper_ref_to_dict(pillar_id, sample_paper_ref)
        
        assert paper_dict['id'] == "test.12345"
        assert paper_dict['pillar_id'] == "P2"
        assert paper_dict['title'] == "Test Paper: Advanced Techniques"
        assert paper_dict['authors'] == ["Dr. Test", "Prof. Example"]
        assert paper_dict['venue'] == "Test Conference"
        assert paper_dict['year'] == 2023
        assert paper_dict['citation_count'] == 42
        
        # Convert back to PaperRef
        paper_ref = _dict_to_paper_ref(paper_dict)
        
        assert paper_ref.id == sample_paper_ref.id
        assert paper_ref.title == sample_paper_ref.title
        assert paper_ref.authors == sample_paper_ref.authors
        assert paper_ref.venue == sample_paper_ref.venue
        assert paper_ref.year == sample_paper_ref.year
        assert paper_ref.citation_count == sample_paper_ref.citation_count
    
    def test_paper_note_serialization(self, sample_paper_note):
        """Test PaperNote to dict and back conversion."""
        # Convert to dict
        note_dict = _paper_note_to_dict(sample_paper_note)
        
        assert note_dict['paper_id'] == "test.12345"
        assert note_dict['pillar_id'] == "P2"
        assert note_dict['problem'] == "Limited effectiveness of traditional methods"
        assert note_dict['findings'] == ["Achieved 95% accuracy", "Reduced processing time by 50%"]
        assert note_dict['key_terms'] == ["machine learning", "optimization", "algorithm"]
        
        # Convert back to PaperNote
        note = _dict_to_paper_note(note_dict)
        
        assert note.paper_id == sample_paper_note.paper_id
        assert note.pillar_id == sample_paper_note.pillar_id
        assert note.problem == sample_paper_note.problem
        assert note.findings == sample_paper_note.findings
        assert note.key_terms == sample_paper_note.key_terms
    
    def test_lesson_serialization(self, sample_lesson):
        """Test Lesson to dict and back conversion."""
        # Convert to dict
        lesson_dict = _lesson_to_dict(sample_lesson)
        
        assert lesson_dict['paper_id'] == "test.12345"
        assert lesson_dict['pillar_id'] == "P2"
        assert lesson_dict['tl_dr'] == "Novel approach achieves significant improvements in accuracy and speed."
        assert lesson_dict['takeaways'] == sample_lesson.takeaways
        assert lesson_dict['difficulty'] == 2  # DifficultyLevel.MEDIUM
        
        # Convert back to Lesson
        lesson = _dict_to_lesson(lesson_dict)
        
        assert lesson.paper_id == sample_lesson.paper_id
        assert lesson.pillar_id == sample_lesson.pillar_id
        assert lesson.tl_dr == sample_lesson.tl_dr
        assert lesson.takeaways == sample_lesson.takeaways
        assert lesson.difficulty == sample_lesson.difficulty
    
    def test_quiz_card_serialization(self, sample_quiz_cards):
        """Test QuizCard to dict and back conversion."""
        card = sample_quiz_cards[0]
        
        # Convert to dict
        card_dict = _quiz_card_to_dict(card)
        
        assert card_dict['paper_id'] == "test.12345"
        assert card_dict['pillar_id'] == "P2"
        assert card_dict['question'] == "What accuracy did the novel approach achieve?"
        assert card_dict['difficulty'] == 1  # DifficultyLevel.EASY
        assert card_dict['question_type'] == "factual"
        assert card_dict['interval'] == 1
        assert card_dict['ease_factor'] == 2.5
        
        # Convert back to QuizCard
        reconstructed_card = _dict_to_quiz_card(card_dict)
        
        assert reconstructed_card.paper_id == card.paper_id
        assert reconstructed_card.pillar_id == card.pillar_id
        assert reconstructed_card.question == card.question
        assert reconstructed_card.difficulty == card.difficulty
        assert reconstructed_card.question_type == card.question_type


class TestUpsertPaper:
    """Test paper upsert functionality."""
    
    def test_upsert_paper_success(self, mock_supabase_client, sample_paper_ref):
        """Test successful paper upsert."""
        # Mock successful upsert
        mock_result = Mock()
        mock_result.data = [{'id': 'test.12345'}]
        mock_supabase_client.table().upsert().execute.return_value = mock_result
        
        # Call upsert
        upsert_paper(PillarID.P2, sample_paper_ref)
        
        # Verify calls
        mock_supabase_client.table.assert_called_with('papers')
        
        # Verify upsert data includes pillar_id
        call_args = mock_supabase_client.table().upsert.call_args
        upsert_data = call_args[0][0]
        assert upsert_data['pillar_id'] == 'P2'
        assert upsert_data['id'] == 'test.12345'
        assert upsert_data['title'] == sample_paper_ref.title
    
    def test_upsert_paper_no_pillar_id(self, sample_paper_ref):
        """Test upsert failure when pillar_id is missing."""
        with pytest.raises(ValueError, match="pillar_id is required"):
            upsert_paper(None, sample_paper_ref)
    
    def test_upsert_paper_no_paper_id(self):
        """Test upsert failure when paper.id is missing."""
        paper = PaperRef(id="", title="Test", authors=[])
        
        with pytest.raises(ValueError, match="paper.id is required"):
            upsert_paper(PillarID.P1, paper)


class TestMarkProcessed:
    """Test mark processed functionality."""
    
    def test_mark_processed_success(self, mock_supabase_client):
        """Test successful marking of paper as processed."""
        # Mock successful update
        mock_result = Mock()
        mock_result.data = [{'id': 'test.12345', 'processed': True}]
        mock_supabase_client.table().update().eq().eq().execute.return_value = mock_result
        
        # Call mark_processed
        mark_processed(PillarID.P2, "test.12345")
        
        # Verify calls
        mock_supabase_client.table.assert_called_with('papers')
        
        # Verify the update was called with correct data
        update_call = mock_supabase_client.table().update.call_args
        update_data = update_call[0][0]
        assert update_data['processed'] is True
        assert 'processed_at' in update_data
    
    def test_mark_processed_no_match(self, mock_supabase_client):
        """Test marking processed when no paper matches pillar_id."""
        # Mock no results
        mock_result = Mock()
        mock_result.data = []
        mock_supabase_client.table().update().eq().eq().execute.return_value = mock_result
        
        # Should not raise error, just log warning
        mark_processed(PillarID.P2, "nonexistent.12345")
        
        # Verify update was called with data (second call has the data)
        update_calls = mock_supabase_client.table().update.call_args_list
        assert len(update_calls) == 2  # Empty call + data call
        assert update_calls[1][0][0]['processed'] is True


class TestInsertOperations:
    """Test insert operations for notes, lessons, and quiz cards."""
    
    def test_insert_note_success(self, mock_supabase_client, sample_paper_note):
        """Test successful note insertion."""
        # Mock successful insert
        mock_result = Mock()
        mock_result.data = [{'id': str(uuid4())}]
        mock_supabase_client.table().insert().execute.return_value = mock_result
        
        # Call insert_note
        insert_note(sample_paper_note)
        
        # Verify calls
        mock_supabase_client.table.assert_called_with('notes')
        
        # Verify insert data
        call_args = mock_supabase_client.table().insert.call_args
        insert_data = call_args[0][0]
        assert insert_data['pillar_id'] == 'P2'
        assert insert_data['paper_id'] == 'test.12345'
        assert insert_data['problem'] == sample_paper_note.problem
    
    def test_insert_lesson_success(self, mock_supabase_client, sample_lesson):
        """Test successful lesson insertion."""
        # Mock successful insert
        mock_result = Mock()
        mock_result.data = [{'id': str(uuid4())}]
        mock_supabase_client.table().insert().execute.return_value = mock_result
        
        # Call insert_lesson
        insert_lesson(sample_lesson)
        
        # Verify calls
        mock_supabase_client.table.assert_called_with('lessons')
        
        # Verify insert data
        call_args = mock_supabase_client.table().insert.call_args
        insert_data = call_args[0][0]
        assert insert_data['pillar_id'] == 'P2'
        assert insert_data['paper_id'] == 'test.12345'
        assert insert_data['tl_dr'] == sample_lesson.tl_dr
    
    def test_insert_quiz_cards_bulk(self, mock_supabase_client, sample_quiz_cards):
        """Test bulk insertion of quiz cards."""
        # Mock successful bulk insert
        mock_result = Mock()
        mock_result.data = [{'id': str(uuid4())} for _ in sample_quiz_cards]
        mock_supabase_client.table().insert().execute.return_value = mock_result
        
        # Call insert_quiz_cards
        insert_quiz_cards(sample_quiz_cards)
        
        # Verify calls
        mock_supabase_client.table.assert_called_with('quiz_cards')
        
        # Verify bulk insert data
        call_args = mock_supabase_client.table().insert.call_args
        insert_data = call_args[0][0]
        assert len(insert_data) == 2  # Two cards
        assert all(item['pillar_id'] == 'P2' for item in insert_data)
        assert all(item['paper_id'] == 'test.12345' for item in insert_data)
    
    def test_insert_quiz_cards_empty(self):
        """Test insert quiz cards with empty list."""
        # Should not raise error
        insert_quiz_cards([])


class TestGetRecentNotes:
    """Test get recent notes functionality."""
    
    def test_get_recent_notes_success(self, mock_supabase_client, sample_paper_note):
        """Test successful retrieval of recent notes."""
        # Mock database rows
        mock_rows = [
            {
                'paper_id': 'test.12345',
                'pillar_id': 'P2',
                'problem': 'Test problem 1',
                'method': 'Test method 1',
                'findings': ['Finding 1', 'Finding 2'],
                'limitations': ['Limitation 1'],
                'future_work': ['Future work 1'],
                'key_terms': ['term1', 'term2'],
                'related_papers': [],
                'confidence_score': 0.9,
                'created_at': '2023-12-01T12:00:00Z'
            },
            {
                'paper_id': 'test.67890',
                'pillar_id': 'P2',
                'problem': 'Test problem 2',
                'method': 'Test method 2',
                'findings': ['Finding 3'],
                'limitations': ['Limitation 2'],
                'future_work': [],
                'key_terms': ['term3'],
                'related_papers': ['related.123'],
                'confidence_score': 0.8,
                'created_at': '2023-12-02T12:00:00Z'
            }
        ]
        
        mock_result = Mock()
        mock_result.data = mock_rows
        mock_supabase_client.table().select().eq().order().limit().execute.return_value = mock_result
        
        # Call get_recent_notes
        notes = get_recent_notes(PillarID.P2, limit=2)
        
        # Verify results
        assert len(notes) == 2
        assert all(isinstance(note, PaperNote) for note in notes)
        assert notes[0].paper_id == 'test.12345'
        assert notes[1].paper_id == 'test.67890'
        
        # Verify table and basic calls were made
        mock_supabase_client.table.assert_called_with('notes')
        mock_supabase_client.table().select.assert_called_with('*')
        
        # Verify ordering and limit
        mock_supabase_client.table().select().eq().order.assert_called_with('created_at', desc=True)
        mock_supabase_client.table().select().eq().order().limit.assert_called_with(2)


class TestQueueOperations:
    """Test paper queue operations."""
    
    def test_queue_add_candidates_with_deduplication(self, mock_supabase_client, sample_paper_ref):
        """Test adding candidates with deduplication."""
        # Mock existing papers and queue
        existing_papers_result = Mock()
        existing_papers_result.data = [{'id': 'existing.123'}]
        
        existing_queue_result = Mock()
        existing_queue_result.data = [{'paper_id': 'queued.456'}]
        
        # Mock insert result
        insert_result = Mock()
        insert_result.data = [{'id': str(uuid4())}]
        
        # Setup mock calls - need different table() instances for different calls
        mock_table_papers = Mock()
        mock_table_queue = Mock()
        mock_table_insert = Mock()
        
        # Configure each table mock
        mock_table_papers.select().eq().execute.return_value = existing_papers_result
        mock_table_queue.select().eq().execute.return_value = existing_queue_result  
        mock_table_insert.insert().execute.return_value = insert_result
        
        # Configure table() to return different mocks based on call order
        mock_supabase_client.table.side_effect = [
            mock_table_papers,   # First call for papers table
            mock_table_queue,    # Second call for queue table
            mock_table_insert    # Third call for insert
        ]
        
        # Test papers: one new, one existing, one already queued
        test_papers = [
            sample_paper_ref,  # New paper
            PaperRef(id="existing.123", title="Existing Paper", authors=[]),  # Already in papers
            PaperRef(id="queued.456", title="Queued Paper", authors=[])  # Already in queue
        ]
        
        # Call queue_add_candidates
        inserted_count = queue_add_candidates(PillarID.P2, test_papers)
        
        # Should only insert the new paper
        assert inserted_count == 1
        
        # Verify insert was called with data (second call has the data)
        insert_calls = mock_table_insert.insert.call_args_list
        assert len(insert_calls) == 2  # Empty call + data call
        insert_data = insert_calls[1][0][0]
        assert len(insert_data) == 1
        assert insert_data[0]['paper_id'] == 'test.12345'
        assert insert_data[0]['pillar_id'] == 'P2'
    
    def test_queue_pop_next_with_papers_data(self, mock_supabase_client):
        """Test popping next papers from queue with full paper data."""
        # Mock queue result
        queue_rows = [
            {
                'id': str(uuid4()),
                'paper_id': 'test.12345',
                'title': 'Queue Title 1',
                'priority': 8,
                'added_at': '2023-12-01T12:00:00Z'
            },
            {
                'id': str(uuid4()),
                'paper_id': 'test.67890',
                'title': 'Queue Title 2',
                'priority': 5,
                'added_at': '2023-12-02T12:00:00Z'
            }
        ]
        queue_result = Mock()
        queue_result.data = queue_rows
        
        # Mock papers result (full data available)
        papers_rows = [
            {
                'id': 'test.12345',
                'title': 'Full Title 1',
                'authors': ['Author 1'],
                'venue': 'Venue 1',
                'year': 2023
            },
            {
                'id': 'test.67890',
                'title': 'Full Title 2',
                'authors': ['Author 2'],
                'venue': 'Venue 2',
                'year': 2023
            }
        ]
        papers_result = Mock()
        papers_result.data = papers_rows
        
        # Mock update result
        update_result = Mock()
        update_result.data = []
        
        # Setup mock calls with separate table instances  
        mock_table_queue = Mock()
        mock_table_papers = Mock()
        mock_table_update = Mock()
        
        mock_table_queue.select().eq().eq().order().order().limit().execute.return_value = queue_result
        mock_table_papers.select().in_().execute.return_value = papers_result
        mock_table_update.update().in_().execute.return_value = update_result
        
        mock_supabase_client.table.side_effect = [
            mock_table_queue,    # First call for queue
            mock_table_papers,   # Second call for papers  
            mock_table_update    # Third call for update
        ]
        
        # Call queue_pop_next
        paper_refs = queue_pop_next(PillarID.P2, limit=2)
        
        # Verify results use full paper data
        assert len(paper_refs) == 2
        assert paper_refs[0].id == 'test.12345'
        assert paper_refs[0].title == 'Full Title 1'  # From papers table
        assert paper_refs[0].authors == ['Author 1']
        assert paper_refs[1].id == 'test.67890'
        assert paper_refs[1].title == 'Full Title 2'
    
    def test_queue_pop_next_fallback_to_queue_data(self, mock_supabase_client):
        """Test popping next papers with fallback when papers table is missing data."""
        # Mock queue result
        queue_rows = [
            {
                'id': str(uuid4()),
                'paper_id': 'missing.123',
                'title': 'Queue Title Only',
                'priority': 8,
                'added_at': '2023-12-01T12:00:00Z'
            }
        ]
        queue_result = Mock()
        queue_result.data = queue_rows
        
        # Mock empty papers result (no data in papers table)
        papers_result = Mock()
        papers_result.data = []
        
        # Mock update result
        update_result = Mock()
        update_result.data = []
        
        # Setup mock calls with separate table instances
        mock_table_queue = Mock()
        mock_table_papers = Mock()
        mock_table_update = Mock()
        
        mock_table_queue.select().eq().eq().order().order().limit().execute.return_value = queue_result
        mock_table_papers.select().in_().execute.return_value = papers_result
        mock_table_update.update().in_().execute.return_value = update_result
        
        mock_supabase_client.table.side_effect = [
            mock_table_queue,    # First call for queue
            mock_table_papers,   # Second call for papers
            mock_table_update    # Third call for update
        ]
        
        # Call queue_pop_next
        paper_refs = queue_pop_next(PillarID.P2, limit=1)
        
        # Verify results use fallback data
        assert len(paper_refs) == 1
        assert paper_refs[0].id == 'missing.123'
        assert paper_refs[0].title == 'Queue Title Only'  # From queue table
        assert paper_refs[0].authors == []  # Fallback empty list


class TestPillarIsolation:
    """Test that all operations enforce pillar isolation."""
    
    def test_all_reads_filter_by_pillar(self, mock_supabase_client):
        """Test that all read operations include pillar_id filters."""
        # Test get_recent_notes
        mock_result = Mock()
        mock_result.data = []
        mock_supabase_client.table().select().eq().order().limit().execute.return_value = mock_result
        
        get_recent_notes(PillarID.P3, limit=5)
        
        # Verify pillar filter was applied
        eq_calls = mock_supabase_client.table().select().eq.call_args_list
        assert ('pillar_id', 'P3') in [call[0] for call in eq_calls]
    
    def test_all_writes_include_pillar_id(self, mock_supabase_client, sample_paper_ref, sample_paper_note):
        """Test that all write operations include pillar_id in data."""
        mock_result = Mock()
        mock_result.data = [{}]
        mock_supabase_client.table().upsert().execute.return_value = mock_result
        mock_supabase_client.table().insert().execute.return_value = mock_result
        
        # Test upsert_paper
        upsert_paper(PillarID.P4, sample_paper_ref)
        upsert_call = mock_supabase_client.table().upsert.call_args
        assert upsert_call[0][0]['pillar_id'] == 'P4'
        
        # Test insert_note
        insert_note(sample_paper_note)
        insert_call = mock_supabase_client.table().insert.call_args
        assert insert_call[0][0]['pillar_id'] == 'P2'
    
    def test_cross_pillar_access_prevented(self, mock_supabase_client):
        """Test that operations cannot access data from different pillars."""
        # Mock update with no results (simulating wrong pillar)
        mock_result = Mock()
        mock_result.data = []
        mock_supabase_client.table().update().eq().eq().execute.return_value = mock_result
        
        # Try to mark processed with wrong pillar - should not update anything
        mark_processed(PillarID.P1, "test.12345")
        
        # Verify update was called with data (second call has the data)
        update_calls = mock_supabase_client.table().update.call_args_list
        assert len(update_calls) == 2  # Empty call + data call
        assert update_calls[1][0][0]['processed'] is True


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_missing_pillar_id_errors(self):
        """Test that missing pillar_id raises ValueError."""
        paper_ref = PaperRef(id="test", title="Test", authors=[])
        
        with pytest.raises(ValueError, match="pillar_id is required"):
            upsert_paper(None, paper_ref)
        
        with pytest.raises(ValueError, match="pillar_id is required"):
            mark_processed(None, "test.123")
        
        with pytest.raises(ValueError, match="pillar_id is required"):
            get_recent_notes(None)
    
    def test_missing_required_fields_errors(self):
        """Test that missing required fields raise ValueError."""
        note = PaperNote(
            paper_id="",  # Missing paper_id
            pillar_id=PillarID.P1,
            problem="Test",
            method="Test",
            findings=[],
            limitations=[],
            future_work=[],
            key_terms=[]
        )
        
        with pytest.raises(ValueError, match="paper_id is required"):
            insert_note(note)
    
    @patch('nlp_pillars.db.logger')
    def test_database_error_logging(self, mock_logger, mock_supabase_client, sample_paper_ref):
        """Test that database errors are logged appropriately."""
        # Mock database error
        mock_supabase_client.table().upsert().execute.side_effect = Exception("Database connection failed")
        
        with pytest.raises(ValueError, match="Failed to upsert paper"):
            upsert_paper(PillarID.P1, sample_paper_ref)
        
        # Verify error was logged
        mock_logger.error.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
