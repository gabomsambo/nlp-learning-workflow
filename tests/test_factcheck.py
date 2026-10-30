"""Fact-check engine. Hermetic: no database, network, or paid API call."""

import inspect
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from anthropic import Anthropic
from anthropic.resources.messages import Messages
from typer.testing import CliRunner

from nlp_pillars.cli import app
from nlp_pillars.db import (
    FactcheckSaveError,
    insert_podcast_fact_check,
    list_paper_chunks,
)
from nlp_pillars.factcheck.anchor import (
    choose_chunk_rows,
    evidence_for_quotes,
    locate_quote,
    place_span,
)
from nlp_pillars.factcheck.claude import Spend, parse_structured
from nlp_pillars.factcheck.cost import price_usage
from nlp_pillars.factcheck.engine import execute_factcheck, run_factcheck
from nlp_pillars.factcheck.errors import FactcheckBudgetError, FactcheckError
from nlp_pillars.factcheck.extract import DraftClaim
from nlp_pillars.factcheck.prompts import verification_instructions
from nlp_pillars.factcheck.schemas import (
    ClaimVerdict,
    ExtractedClaim,
    ExtractionBatch,
    VerdictBatch,
)
from nlp_pillars.factcheck.severity import assign_severity
from nlp_pillars.factcheck.verify import screen_claims
from nlp_pillars.schemas import PodcastScript

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (ROOT / "docs/migrations/022_podcast_fact_checks.sql").read_text()
SCHEMA = (ROOT / "schema.sql").read_text()
RUNNER = CliRunner()
SCRIPT_ID = "b4269725-1111-4111-8111-111111111111"
PAPER_MARKER = "UNIQUE_PAPER_TAIL_9f3c"

SCRIPT = "\n".join([
    "[HOST]: The model has 7B parameters.",
    "[HOST]: The paper never tests on real AML data.",
])
PAPER = (
    "The model has 7 billion parameters. "
    "We compare laundering motifs with real Italian bank data in section 4. "
    + PAPER_MARKER
)


def _usage(**overrides):
    base = dict(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        cache_creation=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _response(parsed, **usage_overrides):
    return SimpleNamespace(
        parsed_output=parsed,
        usage=_usage(**usage_overrides),
        stop_reason="end_turn",
    )


def _verdict(claim_id, verdict, quote, error_type="none"):
    return ClaimVerdict(
        id=claim_id,
        quotes=[quote] if quote else [],
        source_says="The paper's words.",
        rationale="The verdict follows this rationale.",
        verdict=verdict,
        error_type=error_type,
    )


def _client(parse):
    client = Mock(spec=Anthropic)
    messages = Mock(spec=Messages)
    client.messages = messages
    messages.parse.side_effect = parse
    return client, messages


def _ids(user: str) -> list[str]:
    return re.findall(r"(?m)^id: (\S+)$", user)


@pytest.mark.parametrize(
    ("verdict", "scope", "category", "error_type", "severity"),
    [
        ("contradicted", "paper", "number", "number_mismatch", "high"),
        ("contradicted", "paper", "limitation", "other", "high"),
        ("contradicted", "background", "result", "number_mismatch", "medium"),
        ("unsupported", "paper", "number", "invented_detail", "high"),
        ("unsupported", "paper", "result", "other", "high"),
        ("unsupported", "paper", "comparison", "other", "high"),
        ("unsupported", "paper", "method", "method_misdescribed", "high"),
        ("unsupported", "paper", "setup", "invented_detail", "high"),
        ("unsupported", "paper", "attribution", "wrong_attribution", "high"),
        ("unsupported", "paper", "implication", "invented_detail", "medium"),
        ("unsupported", "paper", "limitation", "invented_detail", "medium"),
        ("unsupported", "background", "attribution", "outside_knowledge", "low"),
        ("unsupported", "paper", "number", "outside_knowledge", "low"),
        ("unsupported", "paper", "background", "other", "low"),
        ("partially_supported", "paper", "result", "number_mismatch", "medium"),
        ("partially_supported", "paper", "method", "wrong_attribution", "medium"),
        ("partially_supported", "paper", "result", "overstatement", "medium"),
        ("partially_supported", "paper", "comparison", "overstatement", "medium"),
        ("partially_supported", "paper", "number", "overstatement", "medium"),
        ("partially_supported", "paper", "method", "overstatement", "low"),
        ("partially_supported", "paper", "result", "other", "low"),
        ("supported", "paper", "result", "none", "none"),
        ("not_checkable", "paper", "implication", "other", "none"),
    ],
)
def test_severity_table(verdict, scope, category, error_type, severity):
    assert assign_severity(verdict, scope, category, error_type) == severity


def test_verdict_schema_puts_rationale_before_verdict():
    props = list(ClaimVerdict.model_json_schema()["properties"])
    assert props == [
        "id", "quotes", "source_says", "rationale", "verdict", "error_type",
    ]
    assert props.index("rationale") < props.index("verdict")
    instructions = verification_instructions(rejudge=False)
    assert "THEN choose the verdict" in instructions
    assert "so the verdict follows from the evidence" in instructions


def test_absence_claims_are_checkable_and_a_contradiction_is_high():
    screen = verification_instructions(rejudge=False)
    rejudge = verification_instructions(rejudge=True)
    assert "Claims of ABSENCE" in screen
    assert "supported if the source indeed does not do or discuss it" in screen
    assert "contradicted if it does" in screen
    assert "Be careful in both directions" not in screen
    assert "Claims of ABSENCE" in rejudge
    assert "Be careful in both directions" in rejudge
    assert "when a table is ambiguous" in rejudge
    # An absence the paper actually covers is a contradiction about the paper.
    assert assign_severity(
        "contradicted", "paper", "limitation", "other",
    ) == "high"


def test_span_anchors_and_reanchors():
    line = "Stage 1 plus 3 gives  61.7 on the average."
    hit = place_span([line], 0, "gives 61.7", 0, 1)
    assert hit is not None
    assert line[hit.start:hit.end] == "gives  61.7"
    assert hit.text == "gives  61.7"

    lines = ["hello world", "gives 61.7 exactly"]
    moved = place_span(lines, 0, "gives 61.7", 0, 2)
    assert moved is not None
    assert moved.line_index == 1
    assert lines[1][moved.start:moved.end] == "gives 61.7"

    ambiguous = place_span(["gives 61.7", "gives 61.7"], 9, "gives 61.7", 0, 2)
    assert ambiguous is None
    missing = place_span([line], 0, "not in this line", 0, 1)
    assert missing is None


def test_quote_exact_fuzzy_and_missing():
    paper = "Hashing is memory efficient according to the authors."
    chunks = [
        {"id": "early", "chunk_index": 0, "text": "introduction only"},
        {"id": "late", "chunk_index": 1, "text": paper},
    ]
    exact = locate_quote(paper, chunks, "Hashing is memory efficient")
    assert exact is not None
    assert exact.match_score == 100
    assert paper[exact.char_start:exact.char_end] == "Hashing is memory efficient"
    assert exact.chunk_id == "late"
    assert exact.chunk_index == 1

    fuzzy = locate_quote(paper, chunks, "Hashing is memroy efficient according")
    assert fuzzy is not None
    assert fuzzy.match_score >= 90
    assert fuzzy.quote == paper[fuzzy.char_start:fuzzy.char_end]
    assert "memory" in fuzzy.quote

    assert locate_quote(paper, chunks, "this sentence is not in the paper at all") is None

    evidence, status = evidence_for_quotes(
        paper,
        chunks,
        ["this sentence is not in the paper at all"],
        verdict="unsupported",
    )
    assert evidence == []
    assert status == "not_located"
    _kept, opinion = evidence_for_quotes(paper, chunks, [], verdict="not_checkable")
    assert opinion == "not_required"


def test_choose_chunk_rows_prefers_the_current_version():
    rows = [
        {"id": "old", "chunk_index": 0, "text": "old", "chunker_version": "v0", "created_at": "2020"},
        {"id": "new", "chunk_index": 0, "text": "new", "chunker_version": "v1", "created_at": "2026"},
    ]
    chosen = choose_chunk_rows(rows, "v1")
    assert [row["id"] for row in chosen] == ["new"]
    fallback = choose_chunk_rows(rows, "missing")
    assert [row["id"] for row in fallback] == ["new"]


def test_parse_kwargs_match_the_real_messages_parse():
    signature = inspect.signature(Messages.parse)
    client, messages = _client(lambda **_kwargs: _response(ExtractionBatch(claims=[])))
    parse_structured(
        client,
        model="claude-haiku-5-5",
        system="instructions",
        user="hello",
        output_format=ExtractionBatch,
        step="extract",
        spend=Spend(None),
    )
    kwargs = messages.parse.call_args.kwargs
    for key in kwargs:
        assert key in signature.parameters, key
    for banned in ("temperature", "top_p", "top_k", "tools", "tool_choice"):
        assert banned not in kwargs
    assert kwargs["output_format"] is ExtractionBatch
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["max_tokens"] == 16000


def test_prices_match_the_measured_rates():
    # Under 100k tokens Haiku cache reads are $0.01/MTok (0.1 x $0.10).
    short_read = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 10_000,
        "cache_creation_input_tokens": 0,
    }
    assert price_usage("claude-haiku-5-5", short_read)[0] == pytest.approx(
        10_000 / 1_000_000 * 0.01,
    )
    short_write = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 10_000,
    }
    assert price_usage("claude-haiku-5-5", short_write)[0] == pytest.approx(
        10_000 / 1_000_000 * 0.125,
    )
    hour = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 10_000,
        },
    }
    assert price_usage("claude-haiku-5-5", hour)[0] == pytest.approx(
        10_000 / 1_000_000 * 0.20,
    )
    # A prompt over 100k tokens uses Haiku's long-context rates for the lot.
    long_read = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 1_000_000,
        "cache_creation_input_tokens": 0,
    }
    assert price_usage("claude-haiku-5-5", long_read)[0] == pytest.approx(0.05)
    long_prompt = {
        "input_tokens": 100_001,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    assert price_usage("claude-haiku-5-5", long_prompt)[0] == pytest.approx(
        100_001 / 1_000_000 * 0.50, abs=1e-6,
    )
    sonnet_out = {
        "input_tokens": 0,
        "output_tokens": 1_000,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    assert price_usage("claude-sonnet-5-5", sonnet_out)[0] == pytest.approx(0.01)
    with pytest.raises(FactcheckError):
        price_usage("not-a-priced-model", sonnet_out)


def _pipeline_parse(**kwargs):
    fmt = kwargs["output_format"]
    user = kwargs["messages"][0]["content"]
    system = kwargs["system"]
    if fmt is ExtractionBatch:
        assert system[-1]["cache_control"] == {"type": "ephemeral"}
        assert "L0: " in system[-1]["text"]
        return _response(ExtractionBatch(claims=[
            ExtractedClaim(
                line=0,
                span="7B parameters",
                claim="The model has 7B parameters.",
                scope="paper",
                category="number",
                verify_marker=False,
            ),
            ExtractedClaim(
                line=1,
                span="never tests on real AML data",
                claim="The paper never tests on real AML data.",
                scope="paper",
                category="limitation",
                verify_marker=False,
            ),
        ]))
    assert fmt is VerdictBatch
    assert isinstance(system, list)
    assert system[-1]["cache_control"] == {"type": "ephemeral"}
    assert PAPER_MARKER in system[-1]["text"]
    assert PAPER_MARKER not in user
    assert "SOURCE\n" + PAPER == system[-1]["text"]
    ids = _ids(user)
    rejudge = "Be careful in both directions" in system[0]["text"]
    if rejudge:
        assert "Claims of ABSENCE" in system[0]["text"]
        assert "7B parameters" not in user
        assert "never tests" in user
        assert ids == ["0"]
        return _response(_verdict_batch([
            _verdict("0", "contradicted", "real Italian bank data", "other"),
        ]), output_tokens=20)
    assert "Be careful in both directions" not in system[0]["text"]
    assert ids == ["0", "1"]
    return _response(VerdictBatch(results=[
        _verdict("0", "supported", "7 billion parameters"),
        _verdict("1", "contradicted", "real Italian bank data", "other"),
    ]))


def _verdict_batch(items):
    return VerdictBatch(results=items)


def test_run_checks_the_whole_paper_and_records_cost_per_call():
    client, messages = _client(_pipeline_parse)
    chunks = [
        {"id": "c0", "chunk_index": 0, "text": "The model has 7 billion parameters."},
        {
            "id": "c1",
            "chunk_index": 1,
            "text": "We compare laundering motifs with real Italian bank data in section 4.",
        },
    ]
    result = run_factcheck(
        SCRIPT,
        PAPER,
        chunks,
        client,
        script_id=SCRIPT_ID,
        paper_id="arxiv:2306.16424",
        models={
            "extract": "claude-haiku-5-5",
            "screen": "claude-haiku-5-5",
            "rejudge": "claude-sonnet-5-5",
        },
    )
    steps = [call["step"] for call in result.calls]
    assert steps == ["extract", "screen", "rejudge"]
    assert result.cost_usd == pytest.approx(sum(call["cost_usd"] for call in result.calls))
    assert all("cost_usd" in call for call in result.calls)
    assert result.counts == {
        "claims": 2,
        "supported": 1,
        "partial": 0,
        "contradicted": 1,
        "unsupported": 0,
        "not_checkable": 0,
        "high": 1,
        "medium": 0,
        "low": 0,
    }
    flagged = result.claims[1]
    assert flagged["verdict"] == "contradicted"
    assert flagged["severity"] == "high"
    assert flagged["evidence_status"] == "located"
    assert flagged["evidence"][0]["chunk_id"] == "c1"
    assert flagged["evidence"][0]["quote"] in PAPER
    assert [step["step"] for step in flagged["judge_trail"]] == ["screen", "rejudge"]
    supported = result.claims[0]
    assert supported["judge_trail"] == [{
        "step": "screen",
        "model": "claude-haiku-5-5",
        "verdict": "supported",
    }]
    assert supported["severity"] == "none"
    assert SCRIPT not in str(result.header())
    assert messages.parse.call_count == 3


def test_screen_batches_fifteen_claims():
    claims = [
        DraftClaim(
            line_index=0, span_start=0, span_end=1, span_text="x",
            claim=f"claim {i}", scope="paper", category="result",
            verify_marker=False,
        )
        for i in range(16)
    ]
    seen = []

    def parse(**kwargs):
        user = kwargs["messages"][0]["content"]
        ids = _ids(user)
        seen.append(ids)
        assert PAPER_MARKER in kwargs["system"][-1]["text"]
        return _response(VerdictBatch(results=[
            _verdict(claim_id, "supported", "") for claim_id in ids
        ]))

    client, _messages = _client(parse)
    screen_claims(
        client, claims, paper_text="paper " + PAPER_MARKER,
        model="claude-haiku-5-5", spend=Spend(None),
    )
    assert seen == [[str(i) for i in range(15)], ["15"]]
    assert all(claim.verdict == "supported" for claim in claims)


def test_extract_windows_and_drops_a_span_that_is_not_in_the_line():
    lines = ["[HOST]: a fact stays here." for _ in range(31)]
    script = "\n".join(lines)

    def parse(**kwargs):
        user = kwargs["messages"][0]["content"]
        if "L0 through L29" in user:
            return _response(ExtractionBatch(claims=[
                ExtractedClaim(
                    line=0,
                    span="not present",
                    claim="A claim whose span is missing.",
                    scope="paper",
                    category="method",
                    verify_marker=False,
                ),
            ]))
        assert "L30 through L30" in user
        return _response(ExtractionBatch(claims=[]))

    client, messages = _client(parse)
    result = run_factcheck(
        script,
        "paper body that is long enough to be a source.",
        [],
        client,
        script_id=SCRIPT_ID,
        paper_id="arxiv:1",
        models={
            "extract": "claude-haiku-5-5",
            "screen": "claude-haiku-5-5",
            "rejudge": "claude-sonnet-5-5",
        },
    )
    assert messages.parse.call_count == 2
    assert result.claims == []
    assert result.counts["claims"] == 0


def test_budget_stops_before_another_call_and_saves_nothing():
    def parse(**kwargs):
        return _response(ExtractionBatch(claims=[]), output_tokens=1_000_000)

    client, messages = _client(parse)
    with pytest.raises(FactcheckBudgetError) as exc:
        run_factcheck(
            SCRIPT,
            PAPER,
            [],
            client,
            script_id=SCRIPT_ID,
            paper_id="arxiv:1",
            models={
                "extract": "claude-haiku-5-5",
                "screen": "claude-haiku-5-5",
                "rejudge": "claude-sonnet-5-5",
            },
            budget_usd=0.30,
        )
    assert messages.parse.call_count == 1
    assert exc.value.cost_usd > 0.30
    assert exc.value.calls[0]["cost_usd"] == exc.value.cost_usd


def _script():
    return PodcastScript(
        id=SCRIPT_ID,
        paper_id="arxiv:2306.16424",
        pillar_id="neural-architectures-language",
        title="AML",
        script=SCRIPT,
    )


def test_failure_leaves_the_script_untouched_and_unsaved():
    script = _script()
    original = script.script
    inserted = []
    client, _messages = _client(lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("down")))

    with (
        patch("nlp_pillars.factcheck.engine.db.get_podcast_script_by_id", return_value=script),
        patch("nlp_pillars.factcheck.engine.db.get_stored_paper_text", return_value=PAPER),
        patch("nlp_pillars.factcheck.engine.db.list_paper_chunks", return_value=[]),
        patch("nlp_pillars.factcheck.engine.db.insert_podcast_fact_check", side_effect=inserted.append),
        patch("nlp_pillars.factcheck.engine.make_client") as maker,
    ):
        with pytest.raises(FactcheckError):
            execute_factcheck(SCRIPT_ID, client=client)

    assert inserted == []
    assert script.script == original
    maker.assert_not_called()
    package = "\n".join(
        path.read_text() for path in (ROOT / "nlp_pillars/factcheck").rglob("*.py")
    )
    assert "update_podcast" not in package
    assert ".table(" not in package
    assert "UPDATE " not in package


def test_save_failure_does_not_change_the_script():
    script = _script()
    original = script.script
    client, _messages = _client(_pipeline_parse)
    with (
        patch("nlp_pillars.factcheck.engine.db.get_podcast_script_by_id", return_value=script),
        patch("nlp_pillars.factcheck.engine.db.get_stored_paper_text", return_value=PAPER),
        patch("nlp_pillars.factcheck.engine.db.list_paper_chunks", return_value=[]),
        patch(
            "nlp_pillars.factcheck.engine.db.insert_podcast_fact_check",
            side_effect=FactcheckSaveError("disk full"),
        ),
    ):
        with pytest.raises(FactcheckError) as exc:
            execute_factcheck(SCRIPT_ID, client=client)
    assert script.script == original
    assert "not saved" in str(exc.value)
    assert exc.value.cost_usd > 0


def test_stages_run_extract_through_persist_in_order():
    script = _script()
    events = []
    client, messages = _client(_pipeline_parse)

    def on_stage(name, status, detail=None):
        events.append((name, status, detail))

    with (
        patch("nlp_pillars.factcheck.engine.db.get_podcast_script_by_id", return_value=script),
        patch("nlp_pillars.factcheck.engine.db.get_stored_paper_text", return_value=PAPER),
        patch("nlp_pillars.factcheck.engine.db.list_paper_chunks", return_value=[]),
        patch(
            "nlp_pillars.factcheck.engine.db.insert_podcast_fact_check",
            return_value="fc-1",
        ) as insert,
        patch("nlp_pillars.factcheck.engine.make_client") as maker,
        patch("nlp_pillars.llm_routes.get_settings") as settings,
    ):
        settings.return_value.factcheck_extract_model = "claude-haiku-5-5"
        settings.return_value.factcheck_screen_model = "claude-haiku-5-5"
        settings.return_value.factcheck_rejudge_model = "claude-sonnet-5-5"
        result = execute_factcheck(SCRIPT_ID, client=client, on_stage=on_stage)

    completed = [name for name, status, _detail in events if status == "completed"]
    assert completed == [
        "factcheck_extract",
        "factcheck_screen",
        "factcheck_rejudge",
        "factcheck_anchor",
        "factcheck_persist",
    ]
    assert ("factcheck_rejudge", "completed", "1 re-judged") in events
    assert result.fact_check_id == "fc-1"
    assert script.script == SCRIPT
    insert.assert_called_once()
    maker.assert_not_called()
    assert messages.parse.called


def test_cancel_before_extract_does_not_call_the_model():
    import threading

    from nlp_pillars.factcheck.errors import FactcheckCancelled

    script = _script()
    client, messages = _client(_pipeline_parse)
    cancel = threading.Event()
    cancel.set()
    events = []
    with (
        patch("nlp_pillars.factcheck.engine.db.get_podcast_script_by_id", return_value=script),
        patch("nlp_pillars.factcheck.engine.db.get_stored_paper_text", return_value=PAPER),
        patch("nlp_pillars.factcheck.engine.db.list_paper_chunks", return_value=[]),
        patch("nlp_pillars.factcheck.engine.db.insert_podcast_fact_check") as insert,
        patch("nlp_pillars.llm_routes.get_settings") as settings,
    ):
        settings.return_value.factcheck_extract_model = "claude-haiku-5-5"
        settings.return_value.factcheck_screen_model = "claude-haiku-5-5"
        settings.return_value.factcheck_rejudge_model = "claude-sonnet-5-5"
        with pytest.raises(FactcheckCancelled):
            execute_factcheck(
                SCRIPT_ID,
                client=client,
                on_stage=lambda *args: events.append(args),
                cancel=cancel,
            )
    assert events == []
    messages.parse.assert_not_called()
    insert.assert_not_called()
    assert script.script == SCRIPT


def test_cli_dry_run_does_not_call_a_model_or_write():
    script = _script()
    with (
        patch("nlp_pillars.factcheck.engine.db.get_podcast_script_by_id", return_value=script),
        patch("nlp_pillars.factcheck.engine.db.get_stored_paper_text", return_value=PAPER),
        patch("nlp_pillars.factcheck.engine.db.list_paper_chunks", return_value=[{"id": "c", "chunk_index": 0, "text": "t"}]),
        patch("nlp_pillars.factcheck.engine.db.insert_podcast_fact_check") as insert,
        patch("nlp_pillars.factcheck.engine.make_client") as maker,
        patch("nlp_pillars.llm_routes.get_settings") as settings,
    ):
        settings.return_value.factcheck_extract_model = "claude-haiku-5-5"
        settings.return_value.factcheck_screen_model = "claude-haiku-5-5"
        settings.return_value.factcheck_rejudge_model = "claude-sonnet-5-5"
        result = RUNNER.invoke(app, ["factcheck", "run", "--script", SCRIPT_ID, "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Nothing written" in result.output
    assert "claude-haiku-5-5" in result.output
    assert "claude-sonnet-5-5" in result.output
    assert str(len(PAPER)) in result.output
    insert.assert_not_called()
    maker.assert_not_called()


def test_migration_and_schema_agree_and_do_not_touch_scripts():
    assert "CREATE TABLE IF NOT EXISTS podcast_fact_checks" in MIGRATION
    assert "CREATE TABLE IF NOT EXISTS podcast_claims" in MIGRATION
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON podcast_fact_checks TO web_anon;" in MIGRATION
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON podcast_claims TO web_anon;" in MIGRATION
    assert "GRANT EXECUTE ON FUNCTION insert_podcast_fact_check(JSONB, JSONB)" in MIGRATION
    assert MIGRATION.rstrip().endswith("NOTIFY pgrst, 'reload schema';")
    assert "UPDATE podcast_scripts" not in MIGRATION
    assert "DELETE FROM podcast_scripts" not in MIGRATION
    assert "CREATE TABLE podcast_fact_checks" in SCHEMA
    assert "CREATE TABLE podcast_claims" in SCHEMA
    assert "insert_podcast_fact_check" in SCHEMA
    assert "GRANT EXECUTE ON FUNCTION insert_podcast_fact_check(JSONB, JSONB)" in SCHEMA


def test_insert_uses_the_rpc_and_names_the_migration():
    client = Mock()
    client.rpc.return_value = {
        "data": None,
        "error": {
            "code": "PGRST202",
            "message": "Could not find the function public.insert_podcast_fact_check",
        },
    }
    with patch("nlp_pillars.db.get_client", return_value=client):
        with pytest.raises(FactcheckSaveError) as exc:
            insert_podcast_fact_check({"status": "succeeded"}, [])
    assert "docs/migrations/022_podcast_fact_checks.sql" in str(exc.value)
    assert client.rpc.call_args.args[0] == "insert_podcast_fact_check"


class _ChunkQuery:
    def __init__(self, rows):
        self.rows = rows
        self._offset = 0

    def table(self, name):
        assert name == "paper_chunks"
        return self

    def select(self, _columns):
        return self

    def eq(self, _column, _value):
        return self

    def order(self, _column):
        return self

    def limit(self, _count):
        return self

    def offset(self, count):
        self._offset = count
        return self

    def execute(self):
        if self._offset:
            return {"data": [], "error": None}
        return {"data": self.rows, "error": None}


def test_list_paper_chunks_keeps_one_version():
    rows = [
        {"id": "old", "chunk_index": 0, "text": "old", "chunker_version": "v0", "created_at": "2020"},
        {"id": "new", "chunk_index": 0, "text": "new", "chunker_version": "v1", "created_at": "2026"},
    ]
    with (
        patch("nlp_pillars.db.get_client", return_value=_ChunkQuery(rows)),
        patch("nlp_pillars.paper_text.chunker_version", return_value="v1"),
    ):
        chosen = list_paper_chunks("arxiv:1")
    assert chosen == [{"id": "new", "chunk_index": 0, "text": "new"}]
