"""
Comprehensive tests for the Typer-based CLI.
All orchestrator and database calls are mocked for fast, reliable testing.
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner

from nlp_pillars.cli import app
from nlp_pillars.schemas import PipelineResult, PaperNote, Lesson, QuizCard, DifficultyLevel, QuestionType


# `nlp_pillars.cli.get_valid_pillars()` dials the real database (falling back to
# `PILLAR_CONFIGS` only on an *exception* -- and `db.get_pillars()` itself swallows
# connection errors and returns `[]` rather than raising, so an unreachable database
# yields an empty valid-pillar list, not the static fallback). Every test that exercises
# `_validate_pillar` therefore patches `nlp_pillars.cli.get_valid_pillars` with this
# fixed list instead of letting it reach a real (or absent) database -- see AGENTS.md's
# "test suite runs in CI now" section for why that dial made these tests CI-flaky.
VALID_PILLAR_SLUGS = [
    "formal-linguistics-nlp",
    "neural-architectures-language",
    "llm-theory-practice",
    "computational-semantics",
    "model-interpretability",
    "ai-agents-tool-use",
    "ml-systems-production",
    "ai-safety-alignment",
]


# Test fixtures
@pytest.fixture
def cli_runner():
    """Typer CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_pipeline_result_success():
    """Mock successful pipeline result."""
    lesson = Lesson(
        paper_id="test.123",
        pillar_id="formal-linguistics-nlp",
        title="Test Lesson Title",
        content="Test lesson content with detailed explanation of the paper.",
        tl_dr="Test lesson summary",
        takeaways=["Key point 1", "Key point 2"],
        practice_ideas=["Practice 1"],
        connections=[],
        difficulty=DifficultyLevel.MEDIUM,
        estimated_time=15
    )

    quiz_cards = [
        QuizCard(
            paper_id="test.123",
            pillar_id="formal-linguistics-nlp",
            question="Test question?",
            answer="Test answer",
            difficulty=DifficultyLevel.EASY,
            question_type=QuestionType.FACTUAL
        ) for _ in range(5)
    ]

    return PipelineResult(
        pillar_id="formal-linguistics-nlp",
        papers_processed=["test.123"],
        lessons_created=[lesson],
        quizzes_generated=quiz_cards,
        podcasts_created=[],
        errors=[],
        total_time_seconds=10.5,
        success=True
    )


@pytest.fixture
def mock_pipeline_result_failure():
    """Mock failed pipeline result."""
    return PipelineResult(
        pillar_id="neural-architectures-language",
        papers_processed=[],
        lessons_created=[],
        quizzes_generated=[],
        podcasts_created=[],
        errors=[{"paper_id": "test.456", "step": "ingest", "message": "Failed to download PDF"}],
        total_time_seconds=5.2,
        success=False
    )


@pytest.fixture
def mock_recent_notes():
    """Mock recent paper notes."""
    return [
        PaperNote(
            paper_id="paper.123",
            pillar_id="formal-linguistics-nlp",
            problem="First test problem",
            method="Test method 1",
            findings=["Finding 1"],
            limitations=["Limitation 1"],
            future_work=["Future work 1"],
            key_terms=["term1"],
            confidence_score=0.9
        ),
        PaperNote(
            paper_id="paper.456",
            pillar_id="formal-linguistics-nlp",
            problem="Second test problem",
            method="Test method 2",
            findings=["Finding 2"],
            limitations=["Limitation 2"],
            future_work=["Future work 2"],
            key_terms=["term2"],
            confidence_score=0.8
        ),
        PaperNote(
            paper_id="paper.789",
            pillar_id="formal-linguistics-nlp",
            problem="Third test problem",
            method="Test method 3",
            findings=["Finding 3"],
            limitations=["Limitation 3"],
            future_work=["Future work 3"],
            key_terms=["term3"],
            confidence_score=0.7
        )
    ]


class TestCLIHelp:
    """Test CLI help and basic functionality."""
    
    def test_help_command(self, cli_runner):
        """Test that help command works and shows subcommands."""
        result = cli_runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "NLP Learning Workflow CLI" in result.stdout
        assert "run" in result.stdout
        assert "status" in result.stdout
        assert "review" in result.stdout
        assert "pillars" in result.stdout
    
    def test_pillars_command(self, cli_runner):
        """Test pillars list command."""
        with patch('nlp_pillars.cli.env_loaded_path', return_value=Path("/test/.env")):
            result = cli_runner.invoke(app, ["pillars"])
        
        assert result.exit_code == 0
        assert "Available Learning Pillars" in result.stdout
        # Check for slug-format pillar IDs (now dynamic from database)
        assert "linguistics" in result.stdout.lower()
        assert "architectures" in result.stdout.lower()
        assert "llm" in result.stdout.lower()
        assert "interpretability" in result.stdout.lower()
        assert "safety" in result.stdout.lower()
        # Check for key parts of the pillar names (may be wrapped/formatted)
        assert "Formal" in result.stdout
        assert "Linguistics" in result.stdout
        assert "Neural" in result.stdout
        assert "Architectures" in result.stdout
        assert "Loaded config from: /test/.env" in result.stdout


class TestRunCommand:
    """Test the run command functionality."""
    
    @patch('nlp_pillars.cli.get_valid_pillars')
    @patch('nlp_pillars.cli.Orchestrator')
    @patch('nlp_pillars.cli.env_loaded_path')
    def test_run_command_success(self, mock_env_path, mock_orchestrator_class, mock_get_pillars, cli_runner, mock_pipeline_result_success):
        """Test successful run command execution."""
        # Mock environment path
        mock_env_path.return_value = Path("/test/.env")

        # Mock the pillar lookup so this test does not dial a real database --
        # see the VALID_PILLAR_SLUGS comment above.
        mock_get_pillars.return_value = VALID_PILLAR_SLUGS

        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.run_daily.return_value = mock_pipeline_result_success
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Run command
        result = cli_runner.invoke(app, ["run", "--pillar", "formal-linguistics-nlp", "--papers", "2"])
        
        # Verify result
        assert result.exit_code == 0
        assert "Pipeline completed successfully!" in result.stdout
        assert "Papers processed: 1" in result.stdout
        assert "Lessons created: 1" in result.stdout
        assert "Quiz cards generated: 5" in result.stdout
        assert "Errors: 0" in result.stdout
        
        # Verify orchestrator was called correctly
        mock_orchestrator_class.assert_called_once_with(enable_quiz=True)
        mock_orchestrator.run_daily.assert_called_once_with("formal-linguistics-nlp", papers_limit=2)
    
    @patch('nlp_pillars.cli.get_valid_pillars')
    @patch('nlp_pillars.cli.Orchestrator')
    @patch('nlp_pillars.cli.env_loaded_path')
    def test_run_command_failure(self, mock_env_path, mock_orchestrator_class, mock_get_pillars, cli_runner, mock_pipeline_result_failure):
        """Test run command with pipeline failure."""
        # Mock environment path
        mock_env_path.return_value = None

        # Mock the pillar lookup so this test does not dial a real database --
        # see the VALID_PILLAR_SLUGS comment above.
        mock_get_pillars.return_value = VALID_PILLAR_SLUGS

        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.run_daily.return_value = mock_pipeline_result_failure
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Run command
        result = cli_runner.invoke(app, ["run", "--pillar", "neural-architectures-language"])
        
        # Verify result
        assert result.exit_code == 1
        assert "Pipeline failed or no papers processed" in result.stdout
        assert "Papers processed: 0" in result.stdout
        assert "Errors: 1" in result.stdout
        assert "No .env file found" in result.stdout
        
        # Verify orchestrator was called
        mock_orchestrator.run_daily.assert_called_once_with("neural-architectures-language", papers_limit=1)
    
    @patch('nlp_pillars.cli.get_valid_pillars')
    @patch('nlp_pillars.cli.Orchestrator')
    @patch('nlp_pillars.cli.env_loaded_path')
    def test_run_command_exception(self, mock_env_path, mock_orchestrator_class, mock_get_pillars, cli_runner):
        """Test run command with orchestrator exception."""
        # Mock environment path
        mock_env_path.return_value = None

        # Mock the pillar lookup so this test does not dial a real database --
        # see the VALID_PILLAR_SLUGS comment above.
        mock_get_pillars.return_value = VALID_PILLAR_SLUGS

        # Mock orchestrator to raise exception
        mock_orchestrator = Mock()
        mock_orchestrator.run_daily.side_effect = Exception("Database connection failed")
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Run command
        result = cli_runner.invoke(app, ["run", "--pillar", "llm-theory-practice", "--papers", "1"])
        
        # Verify result
        assert result.exit_code == 1
        assert "Pipeline error: Database connection failed" in result.stdout
    
    @patch('nlp_pillars.cli.get_valid_pillars')
    def test_run_command_invalid_pillar(self, mock_get_pillars, cli_runner):
        """Test run command with invalid pillar."""
        # Mock the pillar lookup so this test does not dial a real database --
        # see the VALID_PILLAR_SLUGS comment above. "P9" is invalid against this
        # list regardless, but a real DB dial is what made this test non-hermetic.
        mock_get_pillars.return_value = VALID_PILLAR_SLUGS

        result = cli_runner.invoke(app, ["run", "--pillar", "P9", "--papers", "1"])

        assert result.exit_code == 1
        assert "Invalid pillar 'P9'" in result.stdout
        assert "Valid pillars:" in result.stdout

    def test_run_command_default_papers(self, cli_runner):
        """Test run command with default papers value."""
        with patch('nlp_pillars.cli.get_valid_pillars', return_value=VALID_PILLAR_SLUGS), \
             patch('nlp_pillars.cli.Orchestrator') as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator.run_daily.return_value = Mock(success=True, papers_processed=[], lessons_created=[], quizzes_generated=[], errors=[], total_time_seconds=1.0)
            mock_orchestrator_class.return_value = mock_orchestrator

            result = cli_runner.invoke(app, ["run", "--pillar", "formal-linguistics-nlp"])

            # Should use default papers=1
            mock_orchestrator.run_daily.assert_called_once_with("formal-linguistics-nlp", papers_limit=1)


class TestStatusCommand:
    """Test the status command functionality."""

    @patch('nlp_pillars.cli.get_valid_pillars')
    @patch('nlp_pillars.cli.db')
    @patch('nlp_pillars.cli.env_loaded_path')
    def test_status_command_success(self, mock_env_path, mock_db, mock_get_pillars, cli_runner, mock_recent_notes):
        """Test successful status command execution."""
        mock_get_pillars.return_value = [
            "formal-linguistics-nlp",
            "neural-architectures-language",
            "llm-theory-practice",
            "computational-semantics",
            "model-interpretability",
            "ai-agents-tool-use",
            "ml-systems-production",
            "ai-safety-alignment"
        ]
        # Mock environment path
        mock_env_path.return_value = Path("/test/.env")
        
        # Mock database operations
        mock_db.get_recent_notes.return_value = mock_recent_notes
        mock_db.get_client.return_value = Mock()
        
        # Mock Supabase client for queue size
        mock_supabase_result = Mock()
        mock_supabase_result.count = 5
        mock_table = Mock()
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_supabase_result
        mock_db.get_client.return_value.table.return_value = mock_table
        
        # Run command
        result = cli_runner.invoke(app, ["status", "--pillar", "formal-linguistics-nlp"])
        
        # Verify result
        assert result.exit_code == 0
        assert "Status for formal-linguistics-nlp" in result.stdout
        assert "Recent Notes" in result.stdout
        assert "paper.123" in result.stdout
        assert "paper.456" in result.stdout
        assert "paper.789" in result.stdout
        
        # Verify database calls
        mock_db.get_recent_notes.assert_called_once_with("formal-linguistics-nlp", limit=3)
    
    @patch('nlp_pillars.cli.get_valid_pillars')
    @patch('nlp_pillars.cli.db')
    @patch('nlp_pillars.cli.env_loaded_path')
    def test_status_command_empty_lessons(self, mock_env_path, mock_db, mock_get_pillars, cli_runner):
        """Test status command with no recent lessons."""
        mock_get_pillars.return_value = [
            "formal-linguistics-nlp",
            "neural-architectures-language",
            "llm-theory-practice",
            "computational-semantics",
            "model-interpretability",
            "ai-agents-tool-use",
            "ml-systems-production",
            "ai-safety-alignment"
        ]
        # Mock environment path
        mock_env_path.return_value = None

        # Mock database operations
        mock_db.get_recent_notes.return_value = []
        mock_db.get_client.return_value = Mock()
        
        # Mock Supabase client for queue size
        mock_supabase_result = Mock()
        mock_supabase_result.count = 0
        mock_table = Mock()
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_supabase_result
        mock_db.get_client.return_value.table.return_value = mock_table
        
        # Run command
        result = cli_runner.invoke(app, ["status", "--pillar", "neural-architectures-language"])
        
        # Verify result
        assert result.exit_code == 0
        assert "No recent notes found" in result.stdout
    
    @patch('nlp_pillars.cli.get_valid_pillars')
    @patch('nlp_pillars.cli.db')
    @patch('nlp_pillars.cli.env_loaded_path')
    def test_status_command_db_error(self, mock_env_path, mock_db, mock_get_pillars, cli_runner):
        """Test status command with database error."""
        mock_get_pillars.return_value = [
            "formal-linguistics-nlp",
            "neural-architectures-language",
            "llm-theory-practice",
            "computational-semantics",
            "model-interpretability",
            "ai-agents-tool-use",
            "ml-systems-production",
            "ai-safety-alignment"
        ]
        # Mock environment path
        mock_env_path.return_value = None

        # Mock database error
        mock_db.get_recent_notes.side_effect = Exception("Database connection failed")
        
        # Run command
        result = cli_runner.invoke(app, ["status", "--pillar", "llm-theory-practice"])
        
        # Verify result
        assert result.exit_code == 1
        assert "Error fetching lessons: Database connection failed" in result.stdout
    
    @patch('nlp_pillars.cli.get_valid_pillars')
    def test_status_command_invalid_pillar(self, mock_get_pillars, cli_runner):
        """Test status command with invalid pillar."""
        # See the VALID_PILLAR_SLUGS comment above: without this the command
        # dials the real (possibly absent) database before failing validation.
        mock_get_pillars.return_value = VALID_PILLAR_SLUGS

        result = cli_runner.invoke(app, ["status", "--pillar", "P0"])

        assert result.exit_code == 1
        assert "Invalid pillar 'P0'" in result.stdout


class TestReviewCommand:
    """Test the review command functionality."""

    @patch('nlp_pillars.cli.get_valid_pillars')
    @patch('nlp_pillars.cli.db')
    @patch('nlp_pillars.cli.env_loaded_path')
    def test_review_command_success(self, mock_env_path, mock_db, mock_get_pillars, cli_runner):
        """Test successful review command execution."""
        mock_get_pillars.return_value = [
            "formal-linguistics-nlp",
            "neural-architectures-language",
            "llm-theory-practice",
            "computational-semantics",
            "model-interpretability",
            "ai-agents-tool-use",
            "ml-systems-production",
            "ai-safety-alignment"
        ]
        # Mock environment path
        mock_env_path.return_value = Path("/test/.env")

        # Mock Supabase client for due quiz cards
        mock_supabase_result = Mock()
        mock_supabase_result.count = 3
        mock_table = Mock()
        mock_table.select.return_value.eq.return_value.lte.return_value.execute.return_value = mock_supabase_result
        mock_db.get_client.return_value.table.return_value = mock_table
        
        # Run command
        result = cli_runner.invoke(app, ["review", "--pillar", "formal-linguistics-nlp"])
        
        # Verify result
        assert result.exit_code == 0
        assert "Review for formal-linguistics-nlp" in result.stdout
        assert "Due today: 3" in result.stdout
    
    @patch('nlp_pillars.cli.get_valid_pillars')
    @patch('nlp_pillars.cli.db')
    @patch('nlp_pillars.cli.env_loaded_path')
    def test_review_command_no_due_cards(self, mock_env_path, mock_db, mock_get_pillars, cli_runner):
        """Test review command with no due quiz cards."""
        mock_get_pillars.return_value = [
            "formal-linguistics-nlp",
            "neural-architectures-language",
            "llm-theory-practice",
            "computational-semantics",
            "model-interpretability",
            "ai-agents-tool-use",
            "ml-systems-production",
            "ai-safety-alignment"
        ]
        # Mock environment path
        mock_env_path.return_value = None

        # Mock Supabase client for due quiz cards
        mock_supabase_result = Mock()
        mock_supabase_result.count = 0
        mock_table = Mock()
        mock_table.select.return_value.eq.return_value.lte.return_value.execute.return_value = mock_supabase_result
        mock_db.get_client.return_value.table.return_value = mock_table
        
        # Run command
        result = cli_runner.invoke(app, ["review", "--pillar", "model-interpretability"])
        
        # Verify result
        assert result.exit_code == 0
        assert "Due today: 0" in result.stdout
    
    @patch('nlp_pillars.cli.get_valid_pillars')
    @patch('nlp_pillars.cli.db')
    @patch('nlp_pillars.cli.env_loaded_path')
    def test_review_command_db_error(self, mock_env_path, mock_db, mock_get_pillars, cli_runner):
        """Test review command with database error."""
        mock_get_pillars.return_value = [
            "formal-linguistics-nlp",
            "neural-architectures-language",
            "llm-theory-practice",
            "computational-semantics",
            "model-interpretability",
            "ai-agents-tool-use",
            "ml-systems-production",
            "ai-safety-alignment"
        ]
        # Mock environment path
        mock_env_path.return_value = None

        # Mock database error
        mock_db.get_client.side_effect = Exception("Database connection failed")
        
        # Run command
        result = cli_runner.invoke(app, ["review", "--pillar", "ai-safety-alignment"])
        
        # Verify result
        assert result.exit_code == 1
        assert "Error fetching due quiz cards: Database connection failed" in result.stdout
    
    @patch('nlp_pillars.cli.get_valid_pillars')
    def test_review_command_invalid_pillar(self, mock_get_pillars, cli_runner):
        """Test review command with invalid pillar."""
        # See the VALID_PILLAR_SLUGS comment above: without this the command
        # dials the real (possibly absent) database before failing validation.
        mock_get_pillars.return_value = VALID_PILLAR_SLUGS

        result = cli_runner.invoke(app, ["review", "--pillar", "PX"])

        assert result.exit_code == 1
        assert "Invalid pillar 'PX'" in result.stdout


class TestConfigLoading:
    """Test configuration loading and environment handling."""
    
    @patch('nlp_pillars.cli.env_loaded_path')
    def test_env_path_logging(self, mock_env_path, cli_runner):
        """Test that env path is logged when present."""
        mock_env_path.return_value = Path("/custom/path/.env")
        
        result = cli_runner.invoke(app, ["pillars"])
        
        assert result.exit_code == 0
        assert "Loaded config from: /custom/path/.env" in result.stdout
    
    @patch('nlp_pillars.cli.env_loaded_path')
    def test_no_env_file_logging(self, mock_env_path, cli_runner):
        """Test that missing env file is logged appropriately."""
        mock_env_path.return_value = None
        
        result = cli_runner.invoke(app, ["pillars"])
        
        assert result.exit_code == 0
        assert "No .env file found, using environment variables" in result.stdout


class TestPillarValidation:
    """Test pillar validation across commands."""
    
    def test_all_valid_pillars(self, cli_runner):
        """Test that all valid pillars are accepted."""
        for pillar in VALID_PILLAR_SLUGS:
            # Test with run command (mock orchestrator to avoid actual execution).
            # get_valid_pillars is mocked too -- see the VALID_PILLAR_SLUGS comment
            # above -- so this never dials a real database.
            with patch('nlp_pillars.cli.get_valid_pillars', return_value=VALID_PILLAR_SLUGS), \
                 patch('nlp_pillars.cli.Orchestrator') as mock_orchestrator_class:
                mock_orchestrator = Mock()
                mock_orchestrator.run_daily.return_value = Mock(success=True, papers_processed=[], lessons_created=[], quizzes_generated=[], errors=[], total_time_seconds=1.0)
                mock_orchestrator_class.return_value = mock_orchestrator

                result = cli_runner.invoke(app, ["run", "--pillar", pillar, "--papers", "1"])

                # Should not fail on pillar validation
                assert "Invalid pillar" not in result.stdout

    def test_invalid_pillars(self, cli_runner):
        """Test that invalid pillars are rejected."""
        invalid_pillars = ["P0", "P6", "P10", "X1", "invalid"]

        for pillar in invalid_pillars:
            # See the VALID_PILLAR_SLUGS comment above.
            with patch('nlp_pillars.cli.get_valid_pillars', return_value=VALID_PILLAR_SLUGS):
                result = cli_runner.invoke(app, ["run", "--pillar", pillar])

            assert result.exit_code == 1
            assert f"Invalid pillar '{pillar}'" in result.stdout
            # Now checks against dynamic pillar list from database
            assert "Valid pillars:" in result.stdout


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_run_command_missing_pillar(self, cli_runner):
        """Test run command without pillar argument."""
        result = cli_runner.invoke(app, ["run", "--papers", "1"])
        
        assert result.exit_code == 2  # Typer validation error
    
    def test_run_command_invalid_papers_count(self, cli_runner):
        """Test run command with invalid papers count."""
        result = cli_runner.invoke(app, ["run", "--pillar", "formal-linguistics-nlp", "--papers", "0"])
        
        assert result.exit_code == 2  # Typer validation error
    
    @patch('nlp_pillars.cli.logging.basicConfig')
    def test_logging_setup(self, mock_logging_config, cli_runner):
        """Test that logging is configured correctly."""
        # get_valid_pillars is mocked too -- see the VALID_PILLAR_SLUGS comment
        # above -- so this never dials a real database.
        with patch('nlp_pillars.cli.get_valid_pillars', return_value=VALID_PILLAR_SLUGS), \
             patch('nlp_pillars.cli.Orchestrator') as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator.run_daily.return_value = Mock(success=True, papers_processed=[], lessons_created=[], quizzes_generated=[], errors=[], total_time_seconds=1.0)
            mock_orchestrator_class.return_value = mock_orchestrator

            result = cli_runner.invoke(app, ["run", "--pillar", "formal-linguistics-nlp"])
            
            # Verify logging was configured
            mock_logging_config.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
