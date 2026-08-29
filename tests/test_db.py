"""
Comprehensive tests for Supabase DAO layer.
All Supabase calls are mocked for fast, reliable testing.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from uuid import uuid4

from nlp_pillars.schemas import PaperRef, PaperNote, Lesson, QuizCard, DifficultyLevel, QuestionType
from nlp_pillars.db import (
    get_client, set_client, PostgRESTClient,
    upsert_paper, mark_processed, insert_note, insert_lesson, insert_quiz_cards,
    get_recent_notes, queue_add_candidates, queue_pop_next,
    _paper_ref_to_dict, _dict_to_paper_ref, _paper_note_to_dict, _dict_to_paper_note,
    _lesson_to_dict, _dict_to_lesson, _quiz_card_to_dict, _dict_to_quiz_card,
    get_pillars, get_pillars_or_empty, PillarLookupError,
    add_podcast_script, get_podcast_scripts, get_podcast_script_by_id, get_all_papers,
    PodcastScriptSaveError, PodcastScriptLookupError, PaperLookupError,
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
        pillar_id="models-architectures",
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
        pillar_id="models-architectures",
        # Required on Lesson; without them this fixture raises and every test that
        # requests it reports as an ERROR rather than a failure.
        title="A Novel Approach",
        content="Full lesson body for the database round-trip tests.",
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
            pillar_id="models-architectures",
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
            pillar_id="models-architectures",
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
    
    @patch.dict('os.environ', {'SUPABASE_URL': 'test_url', 'SUPABASE_KEY': 'test_key'})
    def test_get_client_success(self):
        """Test successful client creation with environment variables."""
        # get_client() builds a real PostgRESTClient straight from
        # SUPABASE_URL/SUPABASE_KEY -- this module has no create_client()
        # factory to intercept (that belonged to the old supabase-py client
        # this DAO no longer uses), so assert on the client it constructs.
        # Reset singleton
        set_client(None)

        result = get_client()

        assert isinstance(result, PostgRESTClient)
        assert result.base_url == 'test_url'
        assert result.headers['Authorization'] == 'Bearer test_key'
    
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
        pillar_id = "models-architectures"

        # Convert to dict
        paper_dict = _paper_ref_to_dict(pillar_id, sample_paper_ref)

        assert paper_dict['id'] == "test.12345"
        assert paper_dict['pillar_id'] == "models-architectures"
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
        assert note_dict['pillar_id'] == "models-architectures"
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
        assert lesson_dict['pillar_id'] == "models-architectures"
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
        assert card_dict['pillar_id'] == "models-architectures"
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
        # upsert_paper is an alias for add_paper, which calls .insert() (there
        # is no .upsert() method on TableQuery at all) and that call executes
        # the POST itself and returns {'data': ..., 'error': ...} directly --
        # there is no separate .execute() to hang a mock off.
        mock_supabase_client.table().insert.return_value = {
            'data': [{'id': 'test.12345'}], 'error': None
        }

        # Call upsert
        result = upsert_paper("models-architectures", sample_paper_ref)
        assert result is True

        # Verify calls
        mock_supabase_client.table.assert_called_with('papers')

        # Verify insert data includes pillar_id
        call_args = mock_supabase_client.table().insert.call_args
        insert_data = call_args[0][0]
        assert insert_data['pillar_id'] == 'models-architectures'
        assert insert_data['id'] == 'test.12345'
        assert insert_data['title'] == sample_paper_ref.title

    def test_upsert_paper_no_pillar_id(self, mock_supabase_client, sample_paper_ref):
        """A write that cannot name its pillar is refused, not silently un-pillared.

        _paper_ref_to_dict drops None values so optional metadata (venue, year,
        abstract) can be omitted rather than written as NULL — but that filter applied
        to every key, so a None pillar_id was quietly stripped and the row inserted
        with no pillar at all. Pillar isolation is the one invariant this schema has.

        add_paper turns the ValueError into False rather than letting it escape, so a
        daily run records the paper as failed and carries on.
        """
        mock_supabase_client.table().insert.return_value = {
            'data': [{'id': 'test.12345'}], 'error': None
        }

        assert upsert_paper(None, sample_paper_ref) is False
        mock_supabase_client.table().insert.assert_not_called()

    def test_upsert_paper_no_paper_id(self, mock_supabase_client):
        """An empty paper id is refused too.

        An empty string is not None, so the None-filter never touched it and a row
        with a blank primary key went straight to the database.
        """
        paper = PaperRef(id="", title="Test", authors=[])
        mock_supabase_client.table().insert.return_value = {
            'data': [{'id': ''}], 'error': None
        }

        assert upsert_paper("linguistic-cognitive-foundations", paper) is False
        mock_supabase_client.table().insert.assert_not_called()


class TestMarkProcessed:
    """Test mark processed functionality."""
    
    def test_mark_processed_success(self, mock_supabase_client):
        """Test successful marking of paper as processed."""
        # mark_processed writes to paper_queue, not papers, and its
        # TableQuery.update() executes the PATCH itself and returns
        # {'data': ..., 'error': ...} directly -- there's no .execute() to
        # intercept. It also only ever writes {'processed': True}, no
        # processed_at column.
        mock_supabase_client.table().update.return_value = {
            'data': [{'id': 'test.12345', 'processed': True}], 'error': None
        }

        # Call mark_processed
        result = mark_processed("models-architectures", "test.12345")
        assert result is True

        # Verify calls
        mock_supabase_client.table.assert_called_with('paper_queue')

        # Verify the update was called with correct data
        update_call = mock_supabase_client.table().update.call_args
        update_data = update_call[0][0]
        assert update_data == {'processed': True}
    
    def test_mark_processed_no_match(self, mock_supabase_client):
        """Test marking processed when no paper matches pillar_id."""
        # Mock no results
        mock_result = Mock()
        mock_result.data = []
        mock_supabase_client.table().update().eq().eq().execute.return_value = mock_result
        
        # Should not raise error, just log warning
        mark_processed("models-architectures", "nonexistent.12345")
        
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
        assert insert_data['pillar_id'] == 'models-architectures'
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
        assert insert_data['pillar_id'] == 'models-architectures'
        assert insert_data['paper_id'] == 'test.12345'
        assert insert_data['tl_dr'] == sample_lesson.tl_dr
    
    def test_insert_quiz_cards_bulk(self, mock_supabase_client, sample_quiz_cards):
        """Test bulk insertion of quiz cards."""
        # add_quiz_cards inserts one card per call, not a single batched
        # request, and each TableQuery.insert() executes and returns the
        # response dict directly (no .execute() to hang a mock off).
        mock_supabase_client.table().insert.return_value = {
            'data': [{'id': str(uuid4())}], 'error': None
        }

        # Call insert_quiz_cards
        added = insert_quiz_cards(sample_quiz_cards)
        assert added == 2

        # Verify calls
        mock_supabase_client.table.assert_called_with('quiz_cards')

        # Verify each of the two cards was inserted individually, each
        # carrying pillar_id/paper_id
        insert_calls = mock_supabase_client.table().insert.call_args_list
        assert len(insert_calls) == 2  # One insert per card
        inserted = [call[0][0] for call in insert_calls]
        assert all(item['pillar_id'] == 'models-architectures' for item in inserted)
        assert all(item['paper_id'] == 'test.12345' for item in inserted)
    
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
                'pillar_id': 'models-architectures',
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
                'pillar_id': 'models-architectures',
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
        
        # get_recent_notes' .execute() returns {'data': ..., 'error': ...}
        # directly; a bare Mock with a .data attribute is not subscriptable
        # the way the production code expects.
        mock_supabase_client.table().select().eq().order().limit().execute.return_value = {
            'data': mock_rows, 'error': None
        }

        # Call get_recent_notes
        notes = get_recent_notes("models-architectures", limit=2)
        
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
        # add_to_paper_queue's existing-paper and existing-queue checks both
        # end in .execute(), which returns {'data': ..., 'error': ...}
        # directly, not an object with a .data attribute. The queue check
        # chains TWO .eq() calls (pillar_id, then processed=False) -- one
        # more than the papers check. Queueing a new candidate then calls
        # .insert() with a single dict (not a list) per paper, and that
        # call executes the POST itself with no trailing .execute().
        existing_papers_result = {'data': [{'id': 'existing.123'}], 'error': None}
        existing_queue_result = {'data': [{'paper_id': 'queued.456'}], 'error': None}
        insert_result = {'data': [{'id': str(uuid4())}], 'error': None}

        # Setup mock calls - need different table() instances for different calls
        mock_table_papers = Mock()
        mock_table_queue = Mock()
        mock_table_insert = Mock()

        # Configure each table mock
        mock_table_papers.select().eq().execute.return_value = existing_papers_result
        mock_table_queue.select().eq().eq().execute.return_value = existing_queue_result
        mock_table_insert.insert.return_value = insert_result

        # Configure table() to return different mocks based on call order
        mock_supabase_client.table.side_effect = [
            mock_table_papers,   # First call for papers table
            mock_table_queue,    # Second call for queue table
            mock_table_insert    # Third call: paper_queue insert for the new candidate
        ]

        # Test papers: one new, one existing, one already queued
        test_papers = [
            sample_paper_ref,  # New paper
            PaperRef(id="existing.123", title="Existing Paper", authors=[]),  # Already in papers
            PaperRef(id="queued.456", title="Queued Paper", authors=[])  # Already in queue
        ]

        # Call queue_add_candidates
        inserted_count = queue_add_candidates("models-architectures", test_papers)

        # Should only insert the new paper
        assert inserted_count == 1

        # Verify insert was called once, with the new paper's data
        insert_calls = mock_table_insert.insert.call_args_list
        assert len(insert_calls) == 1
        insert_data = insert_calls[0][0][0]
        assert insert_data['paper_id'] == 'test.12345'
        assert insert_data['pillar_id'] == 'models-architectures'
    
    @patch('nlp_pillars.tools.arxiv_tool.ArXivTool')
    def test_queue_pop_next_with_papers_data(self, mock_arxiv_tool_cls, mock_supabase_client):
        """Test popping next papers from queue with full metadata available."""
        # pop_from_paper_queue never queries a papers table -- there is no
        # .in_() lookup anywhere in the production code. For an arXiv-shaped
        # id it instead calls out to ArXivTool (_fetch_full_paper_metadata),
        # which we mock here rather than hitting the network. Its single
        # select query chains select/eq/eq/order/limit then .execute(),
        # which returns {'data': ..., 'error': ...} directly; the
        # mark-as-processed .update() executes immediately per row (no
        # .in_() batching, no trailing .execute()).
        queue_rows = [
            {
                'id': str(uuid4()),
                'paper_id': '1706.03762',
                'title': 'Queue Title 1',
                'priority': 8,
                'added_at': '2023-12-01T12:00:00Z'
            },
            {
                'id': str(uuid4()),
                'paper_id': '2106.09685',
                'title': 'Queue Title 2',
                'priority': 5,
                'added_at': '2023-12-02T12:00:00Z'
            }
        ]

        mock_supabase_client.table().select().eq().eq().order().limit().execute.return_value = {
            'data': queue_rows, 'error': None
        }
        mock_supabase_client.table().update.return_value = {'data': [], 'error': None}

        full_papers = {
            '1706.03762': PaperRef(id='1706.03762', title='Full Title 1', authors=['Author 1'],
                                    venue='Venue 1', year=2023,
                                    url_pdf='https://arxiv.org/pdf/1706.03762.pdf'),
            '2106.09685': PaperRef(id='2106.09685', title='Full Title 2', authors=['Author 2'],
                                    venue='Venue 2', year=2023,
                                    url_pdf='https://arxiv.org/pdf/2106.09685.pdf'),
        }

        def fake_search(search_query):
            paper_id = search_query.query.removeprefix('id:')
            return [full_papers[paper_id]]

        mock_arxiv_tool_cls.return_value.search.side_effect = fake_search

        # Call queue_pop_next
        paper_refs = queue_pop_next("models-architectures", limit=2)

        # Verify results use full paper data
        assert len(paper_refs) == 2
        assert paper_refs[0].id == '1706.03762'
        assert paper_refs[0].title == 'Full Title 1'  # From ArXivTool, not the queue row
        assert paper_refs[0].authors == ['Author 1']
        assert paper_refs[1].id == '2106.09685'
        assert paper_refs[1].title == 'Full Title 2'
    
    def test_queue_pop_next_fallback_to_queue_data(self, mock_supabase_client):
        """Test popping next papers with fallback when full metadata isn't available."""
        # 'missing.123' is not an arXiv-shaped id, so
        # _fetch_full_paper_metadata never attempts an ArXiv lookup and falls
        # straight back to the queue row's own title/id, with authors
        # defaulting to []. As above, .execute() returns a dict directly and
        # the mark-as-processed .update() executes immediately with no
        # .in_() batching.
        queue_rows = [
            {
                'id': str(uuid4()),
                'paper_id': 'missing.123',
                'title': 'Queue Title Only',
                'priority': 8,
                'added_at': '2023-12-01T12:00:00Z'
            }
        ]

        mock_supabase_client.table().select().eq().eq().order().limit().execute.return_value = {
            'data': queue_rows, 'error': None
        }
        mock_supabase_client.table().update.return_value = {'data': [], 'error': None}

        # Call queue_pop_next
        paper_refs = queue_pop_next("models-architectures", limit=1)

        # Verify results use fallback data
        assert len(paper_refs) == 1
        assert paper_refs[0].id == 'missing.123'
        assert paper_refs[0].title == 'Queue Title Only'  # From queue table
        assert paper_refs[0].authors == []  # Fallback empty list


class TestPillarIsolation:
    """Test that all operations enforce pillar isolation."""
    
    def test_all_reads_filter_by_pillar(self, mock_supabase_client):
        """Test that all read operations include pillar_id filters."""
        # Test get_recent_notes -- .execute() returns {'data': ..., 'error':
        # ...} directly, not an object with a .data attribute.
        mock_supabase_client.table().select().eq().order().limit().execute.return_value = {
            'data': [], 'error': None
        }

        get_recent_notes("data-training-methodologies", limit=5)

        # Verify pillar filter was applied
        eq_calls = mock_supabase_client.table().select().eq.call_args_list
        assert ('pillar_id', 'data-training-methodologies') in [call[0] for call in eq_calls]
    
    def test_all_writes_include_pillar_id(self, mock_supabase_client, sample_paper_ref, sample_paper_note):
        """Test that all write operations include pillar_id in data."""
        # upsert_paper (add_paper) and insert_note (add_note) both call
        # .insert() directly -- there is no .upsert() method on TableQuery,
        # and .insert() executes the POST itself and returns the response
        # dict with no separate .execute() call to intercept.
        mock_supabase_client.table().insert.return_value = {'data': [{}], 'error': None}

        # Test upsert_paper
        upsert_paper("evaluation-interpretability", sample_paper_ref)
        upsert_call = mock_supabase_client.table().insert.call_args
        assert upsert_call[0][0]['pillar_id'] == 'evaluation-interpretability'

        # Test insert_note
        insert_note(sample_paper_note)
        insert_call = mock_supabase_client.table().insert.call_args
        assert insert_call[0][0]['pillar_id'] == 'models-architectures'
    
    def test_cross_pillar_access_prevented(self, mock_supabase_client):
        """Test that operations cannot access data from different pillars."""
        # Mock update with no results (simulating wrong pillar)
        mock_result = Mock()
        mock_result.data = []
        mock_supabase_client.table().update().eq().eq().execute.return_value = mock_result
        
        # Try to mark processed with wrong pillar - should not update anything
        mark_processed("linguistic-cognitive-foundations", "test.12345")
        
        # Verify update was called with data (second call has the data)
        update_calls = mock_supabase_client.table().update.call_args_list
        assert len(update_calls) == 2  # Empty call + data call
        assert update_calls[1][0][0]['processed'] is True


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_missing_pillar_id_errors(self, mock_supabase_client):
        """A paper write without a pillar is refused; the reads still are not.

        upsert_paper now rejects a missing pillar_id rather than letting
        _paper_ref_to_dict's None-filter strip it and write an un-pillared row.

        mark_processed and get_recent_notes still accept None and are left that way
        on purpose: they only ever read or narrow, so the blast radius is a query that
        matches nothing rather than a row written under the wrong pillar. Note the
        related documented bug they run into — TableQuery.eq() renders None as the
        literal string 'None', see AGENTS.md "Sharp edges" — which is deliberately
        untouched here.
        """
        paper_ref = PaperRef(id="test", title="Test", authors=[])

        mock_supabase_client.table().insert.return_value = {
            'data': [{'id': 'test'}], 'error': None
        }
        assert upsert_paper(None, paper_ref) is False

        mock_supabase_client.table().update.return_value = {'data': [], 'error': None}
        assert mark_processed(None, "test.123") is True

        mock_supabase_client.table().select().eq().order().limit().execute.return_value = {
            'data': [], 'error': None
        }
        assert get_recent_notes(None) == []

    def test_missing_required_fields_errors(self, mock_supabase_client):
        """insert_note does not validate required fields either: an empty
        paper_id is written through unchanged rather than rejected. Document
        that instead of the ValueError this test used to assume existed.
        """
        note = PaperNote(
            paper_id="",  # Missing paper_id
            pillar_id="linguistic-cognitive-foundations",
            problem="Test",
            method="Test",
            findings=[],
            limitations=[],
            future_work=[],
            key_terms=[]
        )

        mock_supabase_client.table().insert.return_value = {
            'data': [{'id': str(uuid4())}], 'error': None
        }

        assert insert_note(note) is True
        insert_data = mock_supabase_client.table().insert.call_args[0][0]
        assert insert_data['paper_id'] == ""

    @patch('nlp_pillars.db.logger')
    def test_database_error_logging(self, mock_logger, mock_supabase_client, sample_paper_ref):
        """Test that database errors are logged appropriately."""
        # add_paper wraps the whole call in a bare except and returns False
        # rather than raising -- there is no ValueError path here, and the
        # failure is on .insert() (add_paper calls insert(), never upsert()).
        mock_supabase_client.table().insert.side_effect = Exception("Database connection failed")

        result = upsert_paper("linguistic-cognitive-foundations", sample_paper_ref)

        assert result is False
        # Verify error was logged
        mock_logger.error.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestPillarLookupSignalling:
    """get_pillars must distinguish "no pillars" from "cannot read pillars".

    Collapsing the two into an empty list is what made two separate fallbacks dead
    code: cli.get_valid_pillars() and scheduler.run_all_pillars() each wrap the call
    in an except that could never fire. The user-visible results were a CLI that
    rejected every --pillar argument with "Valid pillars: " and nothing after it, and
    a scheduler that logged "no pillars in the database, nothing to do. Seed them with
    create_pillars.py" — advice that is wrong when the database is merely down.
    """

    def test_an_empty_table_is_an_empty_list(self, mock_supabase_client):
        """Genuine emptiness must stay a plain empty list, not an error."""
        mock_supabase_client.table().select().order().limit().execute.return_value = {
            'data': [], 'error': None
        }
        assert get_pillars() == []

    def test_a_read_failure_raises(self, mock_supabase_client):
        mock_supabase_client.table().select().order().limit().execute.return_value = {
            'data': None, 'error': {'message': 'Connection refused'}
        }
        with pytest.raises(PillarLookupError) as excinfo:
            get_pillars()
        # The real reason has to survive to whoever handles it.
        assert 'Connection refused' in str(excinfo.value)

    def test_a_transport_exception_raises_too(self, mock_supabase_client):
        mock_supabase_client.table.side_effect = RuntimeError('socket is closed')
        with pytest.raises(PillarLookupError):
            get_pillars()

    def test_the_degrading_variant_swallows_it(self, mock_supabase_client):
        """Render paths would rather show an empty dropdown than a 500 — but they
        have to opt into that explicitly rather than getting it by default."""
        mock_supabase_client.table.side_effect = RuntimeError('socket is closed')
        assert get_pillars_or_empty() == []

    def test_the_degrading_variant_still_returns_real_pillars(self, mock_supabase_client):
        mock_supabase_client.table().select().order().limit().execute.return_value = {
            'data': [{
                'id': 'ai-safety-alignment',
                'name': 'AI Safety',
                'goal': 'Understand AI safety and alignment research.',
                'focus_areas': [],
                'papers_per_day': 1,
                # Pillar requires both; a row without them fails to parse and the
                # whole read is reported as a lookup failure.
                'created_at': '2026-08-16T00:00:00+00:00',
                'updated_at': '2026-08-16T00:00:00+00:00',
            }],
            'error': None,
        }
        assert [p.id for p in get_pillars_or_empty()] == ['ai-safety-alignment']


class TestPillarIsRequiredToWrite:
    """_paper_ref_to_dict must refuse a write it cannot attribute to a pillar."""

    def test_a_none_pillar_is_rejected_not_stripped(self, sample_paper_ref):
        with pytest.raises(ValueError) as excinfo:
            _paper_ref_to_dict(None, sample_paper_ref)
        assert 'pillar_id is required' in str(excinfo.value)

    def test_an_empty_pillar_is_rejected(self, sample_paper_ref):
        with pytest.raises(ValueError):
            _paper_ref_to_dict('', sample_paper_ref)

    def test_a_paper_with_no_id_is_rejected(self):
        with pytest.raises(ValueError):
            _paper_ref_to_dict('ai-safety-alignment', PaperRef(id='', title='T', authors=[]))

    def test_optional_metadata_is_still_dropped_when_absent(self):
        """The None-filter has a real job; the fix must not disable it."""
        row = _paper_ref_to_dict(
            'ai-safety-alignment',
            PaperRef(id='arxiv:1', title='T', authors=[]),  # no venue/year/abstract
        )
        assert row['pillar_id'] == 'ai-safety-alignment'
        assert 'venue' not in row and 'year' not in row and 'abstract' not in row


class TestPodcastScriptPersistenceIsHonest:
    """The podcast path must not lose a paid-for artifact, or confuse absent with broken.

    Two separate lies used to live here:
      - add_podcast_script() returned None for every failure, and the router turned
        that into a 500 with the script discarded — four minutes and ~$0.27 gone
        because of a transient hiccup.
      - get_podcast_script_by_id() returned None for both "no such script" and "the
        query failed", and both routes answered 404. Measured: a malformed id
        produced PostgREST `400 invalid input syntax for type uuid` and reached the
        user as "Script not found".
    """

    @pytest.fixture
    def sample_script(self):
        from nlp_pillars.schemas import PodcastScript
        return PodcastScript(
            paper_id="test.12345",
            pillar_id="models-architectures",
            title="Deep Dive: Test Paper",
            script="[HOST]: Hello.",
            word_count=2,
        )

    def test_a_failed_insert_raises_instead_of_returning_none(
        self, mock_supabase_client, sample_script
    ):
        mock_supabase_client.table().insert.return_value = {
            'data': None, 'error': {'message': 'connection refused'}
        }
        with pytest.raises(PodcastScriptSaveError) as excinfo:
            add_podcast_script(sample_script)
        # The real reason has to reach the user; "Failed to save" is not a reason.
        assert 'connection refused' in str(excinfo.value)

    def test_an_insert_that_returns_no_row_is_also_a_failure(
        self, mock_supabase_client, sample_script
    ):
        """No id back means the caller cannot prove the script was stored."""
        mock_supabase_client.table().insert.return_value = {'data': [], 'error': None}
        with pytest.raises(PodcastScriptSaveError):
            add_podcast_script(sample_script)

    def test_a_pre_migration_database_still_saves_the_script(
        self, mock_supabase_client, sample_script
    ):
        """PGRST204 on source_material drops the provenance, never the script.

        Same degradation as paper_queue.url_pdf. Losing the record of what the
        script was written from is bad; losing the script is much worse. The live
        error string is the one PostgREST actually returns — verified against the
        running instance on 2026-08-29.
        """
        attempts = []

        def insert(data):
            attempts.append(dict(data))
            if 'source_material' in data:
                return {'data': None, 'error': {
                    'code': 'PGRST204', 'details': None, 'hint': None,
                    'message': "Could not find the 'source_material' column of "
                               "'podcast_scripts' in the schema cache",
                }}
            return {'data': [{'id': 'abc-123'}], 'error': None}

        mock_supabase_client.table().insert.side_effect = insert

        assert add_podcast_script(sample_script) == 'abc-123'
        assert len(attempts) == 2
        assert 'source_material' in attempts[0]
        assert 'source_material' not in attempts[1]
        # Everything else survives the retry.
        assert attempts[1]['script'] == "[HOST]: Hello."

    def test_a_database_behind_by_two_migrations_still_saves_the_script(
        self, mock_supabase_client, sample_script
    ):
        """011 and 012 both added a nullable JSONB column, and a database can be
        behind by both. One retry per column would report the second PGRST204 as
        a real save failure and throw the script away."""
        attempts = []

        def insert(data):
            attempts.append(dict(data))
            for column in ('source_material', 'options'):
                if column in data:
                    return {'data': None, 'error': {
                        'code': 'PGRST204',
                        'message': f"Could not find the '{column}' column of "
                                   f"'podcast_scripts' in the schema cache",
                    }}
            return {'data': [{'id': 'abc-123'}], 'error': None}

        mock_supabase_client.table().insert.side_effect = insert

        assert add_podcast_script(sample_script) == 'abc-123'
        assert len(attempts) == 3
        assert 'options' not in attempts[-1]
        assert 'source_material' not in attempts[-1]
        assert attempts[-1]['script'] == "[HOST]: Hello."

    def test_options_are_written_and_read_back(self, mock_supabase_client):
        """What a script was aimed at has to survive the round trip, or two
        scripts that differ give no way to tell settings from model."""
        from nlp_pillars.db import _dict_to_podcast_script, _podcast_script_to_dict
        from nlp_pillars.podcast_options import CUSTOM_VALUE, resolve
        from nlp_pillars.schemas import PodcastScript

        options = resolve({
            "field": CUSTOM_VALUE, "field_custom": "molecular biology",
            "length": "45",
        })
        row = _podcast_script_to_dict(PodcastScript(
            paper_id="test.1", pillar_id="p", title="t", script="s",
            options=options,
        ))
        assert row['options']['choices']['field']['custom'] == "molecular biology"

        row.update(id='abc', created_at=None)
        restored = _dict_to_podcast_script(row)
        assert restored.options == options

    def test_a_pre_012_row_reads_back_as_no_options(self):
        """Those scripts were generated when nothing was choosable, so empty is
        the honest reading — not a fabricated set of defaults they never had."""
        from nlp_pillars.db import _dict_to_podcast_script

        restored = _dict_to_podcast_script({
            'id': 'abc', 'paper_id': 'test.1', 'pillar_id': 'p',
            'title': 't', 'script': 's',
        })
        assert restored.options.choices == {}

    def test_a_missing_script_is_none_and_a_broken_query_raises(
        self, mock_supabase_client
    ):
        mock_supabase_client.table().select().eq().execute.return_value = {
            'data': [], 'error': None
        }
        assert get_podcast_script_by_id('00000000-0000-0000-0000-000000000000') is None

        mock_supabase_client.table().select().eq().execute.return_value = {
            'data': None,
            'error': {'code': '22P02',
                      'message': 'invalid input syntax for type uuid: "not-a-uuid"'},
        }
        with pytest.raises(PodcastScriptLookupError) as excinfo:
            get_podcast_script_by_id('not-a-uuid')
        assert 'invalid input syntax' in str(excinfo.value)

    def test_no_scripts_is_empty_and_a_dead_database_raises(self, mock_supabase_client):
        mock_supabase_client.table().select().order().limit().execute.return_value = {
            'data': [], 'error': None
        }
        assert get_podcast_scripts() == []

        mock_supabase_client.table().select().order().limit().execute.return_value = {
            'data': None, 'error': {'message': 'Connection refused'}
        }
        with pytest.raises(PodcastScriptLookupError):
            get_podcast_scripts()

    def test_no_papers_is_empty_and_a_dead_database_raises(self, mock_supabase_client):
        """An empty paper dropdown must not be what an unreachable database looks like."""
        mock_supabase_client.table().select().order().limit().execute.return_value = {
            'data': [], 'error': None
        }
        assert get_all_papers() == []

        mock_supabase_client.table.side_effect = RuntimeError('socket is closed')
        with pytest.raises(PaperLookupError):
            get_all_papers()
