"""Model routing registry for podcast generation."""

from unittest.mock import patch

from nlp_pillars.podcast_models import EXTRACTION_ROUTE, SYNTHESIS_ROUTE


def test_default_extraction_is_deepseek_flash():
    assert EXTRACTION_ROUTE.provider == "deepseek"
    assert EXTRACTION_ROUTE.default_model == "deepseek-v4-flash"


def test_synthesis_stays_on_claude():
    assert SYNTHESIS_ROUTE.provider == "anthropic"
    assert "claude" in SYNTHESIS_ROUTE.default_model


def test_env_override():
    with patch("nlp_pillars.podcast_models.get_settings") as mock_settings:
        mock_settings.return_value.podcast_extraction_model = "custom-extraction"
        mock_settings.return_value.podcast_synthesis_model = "custom-synthesis"
        assert EXTRACTION_ROUTE.resolved_model() == "custom-extraction"
        assert SYNTHESIS_ROUTE.resolved_model() == "custom-synthesis"
