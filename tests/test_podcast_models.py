"""Model routing registry.

The defaults matter: generation runs on Claude Haiku 5.5, the podcast script
stays on Sonnet 5, and DeepSeek is reachable but not on the default path.
"""

from types import SimpleNamespace
from unittest.mock import patch

from nlp_pillars.llm_routes import (
    CONCEPTS_ROUTE,
    DISCOVER_ROUTE,
    PODCAST_EXTRACTION_ROUTE,
    PODCAST_SYNTHESIS_ROUTE,
    QUIZ_ROUTE,
    SUMMARIZE_ROUTE,
    SYNTHESIZE_ROUTE,
)
from nlp_pillars.podcast_models import (
    EXTRACTION_ROUTE,
    FACTCHECK_EXTRACT,
    FACTCHECK_REJUDGE,
    FACTCHECK_SCREEN,
    SYNTHESIS_ROUTE,
)


def _settings(**overrides):
    """A settings object with only the routing attributes set."""
    base = {
        "llm_summarize_model": None,
        "llm_synthesize_model": None,
        "llm_quiz_model": None,
        "llm_discover_model": None,
        "llm_concepts_model": None,
        "llm_podcast_extraction_model": None,
        "llm_podcast_synthesis_model": None,
        "llm_factcheck_extract_model": None,
        "llm_factcheck_screen_model": None,
        "llm_factcheck_rejudge_model": None,
        "podcast_extraction_model": None,
        "podcast_synthesis_model": None,
        "podcast_extraction_provider": None,
        "factcheck_extract_model": None,
        "factcheck_screen_model": None,
        "factcheck_rejudge_model": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_every_generation_stage_defaults_to_claude_haiku():
    for route in (
        SUMMARIZE_ROUTE,
        SYNTHESIZE_ROUTE,
        QUIZ_ROUTE,
        DISCOVER_ROUTE,
        CONCEPTS_ROUTE,
    ):
        assert route.provider == "anthropic", route.stage
        assert route.default_model == "claude-haiku-5-5", route.stage


def test_the_podcast_ground_pack_stays_on_deepseek_with_haiku_behind_it():
    """DeepSeek was topped up on 2026-10-25 and verified working."""
    with patch(
        "nlp_pillars.llm_routes.get_settings",
        return_value=_settings(),
    ):
        resolved = PODCAST_EXTRACTION_ROUTE.resolve()
    assert resolved.provider == "deepseek"
    # `deepseek-v4-flash` is a legacy alias; the model behind it is served as
    # `deepseek-flash` now, and that is the id this route sends.
    assert resolved.model == "deepseek-flash"
    assert resolved.fallback_model == "claude-haiku-5-5"
    assert resolved.secondary_fallback_model == "claude-sonnet-5"


def test_haiku_can_become_the_ground_pack_primary_by_config():
    with patch(
        "nlp_pillars.llm_routes.get_settings",
        return_value=_settings(llm_podcast_extraction_model="claude-haiku-5-5"),
    ):
        resolved = PODCAST_EXTRACTION_ROUTE.resolve()
    assert resolved.provider == "anthropic"
    assert resolved.model == "claude-haiku-5-5"


def test_the_lesson_can_move_to_sonnet_with_a_config_change():
    """The captain's call was Haiku for now; Sonnet is one env var away."""
    assert SYNTHESIZE_ROUTE.default_model == "claude-haiku-5-5"
    assert SYNTHESIZE_ROUTE.fallback_model == "claude-sonnet-5-5"
    with patch(
        "nlp_pillars.llm_routes.get_settings",
        return_value=_settings(llm_synthesize_model="claude-sonnet-5-5"),
    ):
        assert SYNTHESIZE_ROUTE.resolved_model() == "claude-sonnet-5-5"


def test_podcast_synthesis_stays_on_claude_sonnet():
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
    with patch(
        "nlp_pillars.llm_routes.get_settings",
        return_value=_settings(
            llm_summarize_model="custom-summarize",
            llm_synthesize_model="custom-synthesize",
            llm_quiz_model="custom-quiz",
            llm_discover_model="custom-discover",
            llm_concepts_model="custom-concepts",
            llm_podcast_extraction_model="custom-extraction",
            llm_podcast_synthesis_model="custom-synthesis",
            llm_factcheck_extract_model="custom-extract",
            llm_factcheck_screen_model="custom-screen",
            llm_factcheck_rejudge_model="custom-rejudge",
        ),
    ):
        assert SUMMARIZE_ROUTE.resolved_model() == "custom-summarize"
        assert SYNTHESIZE_ROUTE.resolved_model() == "custom-synthesize"
        assert QUIZ_ROUTE.resolved_model() == "custom-quiz"
        assert DISCOVER_ROUTE.resolved_model() == "custom-discover"
        assert CONCEPTS_ROUTE.resolved_model() == "custom-concepts"
        assert EXTRACTION_ROUTE.resolved_model() == "custom-extraction"
        assert SYNTHESIS_ROUTE.resolved_model() == "custom-synthesis"
        assert FACTCHECK_EXTRACT.resolved_model() == "custom-extract"
        assert FACTCHECK_SCREEN.resolved_model() == "custom-screen"
        assert FACTCHECK_REJUDGE.resolved_model() == "custom-rejudge"


def test_the_older_podcast_and_factcheck_names_still_work():
    """An existing .env keeps working after the registry was generalised."""
    with patch(
        "nlp_pillars.llm_routes.get_settings",
        return_value=_settings(
            podcast_extraction_model="custom-extraction",
            podcast_synthesis_model="custom-synthesis",
            factcheck_extract_model="custom-extract",
            factcheck_screen_model="custom-screen",
            factcheck_rejudge_model="custom-rejudge",
        ),
    ):
        assert EXTRACTION_ROUTE.resolved_model() == "custom-extraction"
        assert SYNTHESIS_ROUTE.resolved_model() == "custom-synthesis"
        assert FACTCHECK_EXTRACT.resolved_model() == "custom-extract"
        assert FACTCHECK_SCREEN.resolved_model() == "custom-screen"
        assert FACTCHECK_REJUDGE.resolved_model() == "custom-rejudge"


def test_deepseek_is_selectable_but_not_the_default():
    """The route's default provider is DeepSeek; the env var can still say so."""
    with patch(
        "nlp_pillars.llm_routes.get_settings",
        return_value=_settings(podcast_extraction_provider="deepseek"),
    ):
        resolved = PODCAST_EXTRACTION_ROUTE.resolve()
        assert resolved.provider == "deepseek"
        assert resolved.model == "deepseek-flash"


def test_a_deepseek_model_id_selects_the_deepseek_path():
    """A stale .env line naming a DeepSeek model must not reach Anthropic."""
    with patch(
        "nlp_pillars.llm_routes.get_settings",
        return_value=_settings(podcast_extraction_model="deepseek-v4-flash"),
    ):
        resolved = PODCAST_EXTRACTION_ROUTE.resolve()
        assert resolved.provider == "deepseek"
        assert resolved.model == "deepseek-flash"

def test_a_deepseek_model_id_under_an_explicit_anthropic_provider_falls_back():
    """Anthropic would 404 on a DeepSeek id, so the route uses its own."""
    with patch(
        "nlp_pillars.llm_routes.get_settings",
        return_value=_settings(
            podcast_extraction_provider="anthropic",
            podcast_extraction_model="deepseek-v4-flash",
        ),
    ):
        resolved = PODCAST_EXTRACTION_ROUTE.resolve()
        assert resolved.provider == "anthropic"
        assert resolved.model == "claude-haiku-5-5"
