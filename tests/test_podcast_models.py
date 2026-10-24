"""Model routing registry for podcast generation."""

from unittest.mock import patch

from nlp_pillars.podcast_models import (
    EXTRACTION_ROUTE,
    FACTCHECK_EXTRACT,
    FACTCHECK_REJUDGE,
    FACTCHECK_SCREEN,
    SYNTHESIS_ROUTE,
)


def test_default_extraction_is_deepseek_flash():
    assert EXTRACTION_ROUTE.provider == "deepseek"
    assert EXTRACTION_ROUTE.default_model == "deepseek-v4-flash"


def test_synthesis_stays_on_claude():
    assert SYNTHESIS_ROUTE.provider == "anthropic"
    assert SYNTHESIS_ROUTE.default_model == "claude-sonnet-5"


def test_factcheck_defaults_are_haiku_then_sonnet():
    assert FACTCHECK_EXTRACT.provider == "anthropic"
    assert FACTCHECK_SCREEN.provider == "anthropic"
    assert FACTCHECK_REJUDGE.provider == "anthropic"
    assert FACTCHECK_EXTRACT.default_model == "claude-haiku-5-5"
    assert FACTCHECK_SCREEN.default_model == "claude-haiku-5-5"
    assert FACTCHECK_REJUDGE.default_model == "claude-sonnet-5-5"


def test_env_override():
    with patch("nlp_pillars.podcast_models.get_settings") as mock_settings:
        mock_settings.return_value.podcast_extraction_model = "custom-extraction"
        mock_settings.return_value.podcast_synthesis_model = "custom-synthesis"
        mock_settings.return_value.factcheck_extract_model = "custom-extract"
        mock_settings.return_value.factcheck_screen_model = "custom-screen"
        mock_settings.return_value.factcheck_rejudge_model = "custom-rejudge"
        assert EXTRACTION_ROUTE.resolved_model() == "custom-extraction"
        assert SYNTHESIS_ROUTE.resolved_model() == "custom-synthesis"
        assert FACTCHECK_EXTRACT.resolved_model() == "custom-extract"
        assert FACTCHECK_SCREEN.resolved_model() == "custom-screen"
        assert FACTCHECK_REJUDGE.resolved_model() == "custom-rejudge"
