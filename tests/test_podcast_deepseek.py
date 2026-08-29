"""DeepSeek routing for Ground Pack extraction (calls 1–4).

All provider calls are stubbed — no live API spend in CI.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from nlp_pillars.agents.podcast_agent import (
    GroundPackExtractionError,
    LLMCallResult,
    PodcastAgent,
    TEMPERATURE_ANALYSIS,
    TEMPERATURE_EXTRACTION,
    TEMPERATURE_SCRIPT,
)
from nlp_pillars.schemas import GroundPackCallRecord


def _deepseek_result(text: str = "deepseek extraction", **kwargs) -> LLMCallResult:
    return LLMCallResult(
        text=text,
        provider="deepseek",
        model="deepseek-v4-flash",
        input_tokens=kwargs.get("input_tokens", 100),
        output_tokens=kwargs.get("output_tokens", 50),
        finish_reason=kwargs.get("finish_reason", "stop"),
    )


def _claude_result(text: str = "claude extraction", **kwargs) -> LLMCallResult:
    return LLMCallResult(
        text=text,
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        input_tokens=kwargs.get("input_tokens", 200),
        output_tokens=kwargs.get("output_tokens", 80),
        finish_reason=kwargs.get("finish_reason", "end_turn"),
    )


@pytest.fixture
def mock_ground_pack():
    return {
        "facts_outline": "- Problem\n- Method",
        "core_concepts": "Concept 1",
        "metrics_datasets": "| Dataset | Score |",
        "limitations": "- Scope",
    }


@pytest.fixture
def agent():
    with patch("nlp_pillars.agents.podcast_agent.get_settings") as mock_settings:
        mock_settings.return_value.anthropic_api_key = "sk-ant-test"
        mock_settings.return_value.deepseek_api_key = "sk-deepseek-test"
        mock_settings.return_value.deepseek_base_url = "https://api.deepseek.com"
        mock_settings.return_value.podcast_extraction_model = "deepseek-v4-flash"
        mock_settings.return_value.podcast_synthesis_model = "claude-sonnet-4-5-20250929"
        yield PodcastAgent()


class TestExtractionRouting:
    @pytest.mark.asyncio
    async def test_four_extraction_calls_use_deepseek(self, agent):
        agent._call_deepseek = AsyncMock(side_effect=[
            _deepseek_result("facts"),
            _deepseek_result("concepts"),
            _deepseek_result("metrics"),
            _deepseek_result("limits"),
        ])
        agent._call_claude = AsyncMock()

        await agent._generate_facts_outline("PAPER")
        await agent._generate_core_concepts("PAPER")
        await agent._generate_metrics_datasets("PAPER")
        await agent._generate_limitations("PAPER")

        assert agent._call_deepseek.await_count == 4
        agent._call_claude.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesis_stays_on_claude(self, agent, mock_ground_pack):
        agent._call_claude = AsyncMock(return_value=_claude_result("[HOST]: script"))
        agent._call_deepseek = AsyncMock()

        script = await agent._generate_final_script(mock_ground_pack, "PAPER")

        assert "[HOST]: script" in script
        agent._call_claude.assert_awaited_once()
        agent._call_deepseek.assert_not_called()
        assert agent._call_claude.call_args.kwargs.get("max_tokens") == 16000
        assert agent._call_claude.call_args.kwargs.get("temperature") == TEMPERATURE_SCRIPT

    @pytest.mark.asyncio
    async def test_truncated_deepseek_falls_back_to_claude(self, agent):
        agent._call_deepseek = AsyncMock(
            return_value=_deepseek_result("", finish_reason="length"),
        )
        agent._call_claude = AsyncMock(return_value=_claude_result("recovered table"))

        text, record = await agent._generate_metrics_datasets("PAPER")

        assert text == "recovered table"
        assert record.fallback is True
        assert record.provider == "anthropic"
        assert "truncated" in (record.fallback_reason or "").lower() or "empty" in (record.fallback_reason or "").lower()
        agent._call_claude.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_truncation_after_fallback_raises(self, agent):
        agent._call_deepseek = AsyncMock(
            return_value=_deepseek_result("partial", finish_reason="length"),
        )
        agent._call_claude = AsyncMock(
            return_value=_claude_result("", finish_reason="max_tokens"),
        )

        with pytest.raises(GroundPackExtractionError, match="empty extraction"):
            await agent._generate_facts_outline("PAPER")

    @pytest.mark.asyncio
    async def test_deepseek_unreachable_falls_back_loudly(self, agent):
        request = httpx.Request("POST", "https://dead.example/chat/completions")
        agent._call_deepseek = AsyncMock(
            side_effect=httpx.ConnectError("connection refused", request=request),
        )
        agent._call_claude = AsyncMock(return_value=_claude_result("from claude"))

        text, record = await agent._run_extraction_call(
            "facts_outline", "sys", "user", TEMPERATURE_EXTRACTION,
        )

        assert text == "from claude"
        assert record.fallback is True
        assert "connection refused" in (record.fallback_reason or "")

    @pytest.mark.asyncio
    async def test_missing_deepseek_key_falls_back_without_silent_disable(self, agent):
        agent._deepseek_client = None
        agent._call_claude = AsyncMock(return_value=_claude_result("claude only"))

        text, record = await agent._run_extraction_call(
            "limitations", "sys", "user", TEMPERATURE_ANALYSIS,
        )

        assert text == "claude only"
        assert record.fallback is True
        assert "DEEPSEEK_API_KEY" in (record.fallback_reason or "")

    @pytest.mark.asyncio
    async def test_extraction_temperatures_reach_deepseek(self, agent):
        agent._call_deepseek = AsyncMock(return_value=_deepseek_result("ok"))
        agent._call_claude = AsyncMock()

        await agent._generate_facts_outline("PAPER")
        await agent._generate_core_concepts("PAPER")
        await agent._generate_metrics_datasets("PAPER")
        await agent._generate_limitations("PAPER")

        temps = [c.kwargs["temperature"] for c in agent._call_deepseek.call_args_list]
        assert temps == [
            TEMPERATURE_EXTRACTION,
            TEMPERATURE_ANALYSIS,
            TEMPERATURE_EXTRACTION,
            TEMPERATURE_ANALYSIS,
        ]

    @pytest.mark.asyncio
    async def test_tiny_max_tokens_surfaces_not_passes(self, agent):
        """Forced truncation via max_tokens=1 must not become Ground Pack content."""
        agent._call_deepseek = AsyncMock(
            return_value=_deepseek_result("x", finish_reason="length"),
        )
        agent._call_claude = AsyncMock(
            return_value=_claude_result("y", finish_reason="max_tokens"),
        )

        with pytest.raises(GroundPackExtractionError):
            await agent._run_extraction_call(
                "metrics_datasets", "sys", "user", TEMPERATURE_EXTRACTION, max_tokens=1,
            )

    @pytest.mark.asyncio
    async def test_generate_records_ground_pack_calls(self, agent, mock_ground_pack):
        from nlp_pillars.agents.podcast_agent import FullTextResult
        from nlp_pillars.schemas import PaperRef

        mock_paper = PaperRef(
            id="test-paper-123",
            title="Test Paper",
            authors=["Alice"],
            venue="Test",
            year=2024,
            abstract="Abstract.",
            citation_count=1,
        )
        mock_script_content = "[HOST]: Hello.\n"

        with patch("nlp_pillars.agents.podcast_agent.get_paper_by_id") as mock_get_paper, \
             patch("nlp_pillars.agents.podcast_agent.get_notes_by_paper_id") as mock_get_notes:

            mock_get_paper.return_value = mock_paper
            mock_get_notes.return_value = None

            agent._get_full_text = MagicMock(return_value=FullTextResult("Body text."))
            agent._call_deepseek = AsyncMock(side_effect=[
                _deepseek_result(mock_ground_pack["facts_outline"]),
                _deepseek_result(mock_ground_pack["core_concepts"]),
                _deepseek_result(mock_ground_pack["metrics_datasets"]),
                _deepseek_result(mock_ground_pack["limitations"]),
            ])
            agent._call_claude = AsyncMock(
                return_value=_claude_result(mock_script_content),
            )

            result = await agent.generate("test-paper-123", "models-architectures")

        assert set(result.ground_pack_calls) == {
            "facts_outline", "core_concepts", "metrics_datasets", "limitations",
        }
        for rec in result.ground_pack_calls.values():
            assert isinstance(rec, GroundPackCallRecord)
            assert rec.provider == "deepseek"
            assert rec.model == "deepseek-v4-flash"
            assert rec.fallback is False
        assert agent._call_deepseek.await_count == 4
        assert agent._call_claude.await_count == 1
