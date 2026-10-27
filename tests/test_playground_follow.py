"""Playground follow-along: the alignment pass, its store, migration 029, the run kind, the CLI.

Hermetic. The episode and notebook are synthetic, written for these tests;
nothing reaches a database, the network or a model. The model is a spec'd
mock of the Anthropic client, so the call goes through the real
``llm_client.structured`` (fallback, pricing) with canned answers.
"""

import re
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from anthropic import Anthropic
from anthropic.resources.messages import Messages
from typer.testing import CliRunner

from nlp_pillars import cli, db
from nlp_pillars.playground import FOLLOW_MIGRATION, follow, links, store
from nlp_pillars.playground.follow import AlignDraft, SpanDraft
from nlp_pillars.schemas import PLAYGROUND_ALIGN_STAGES, RunStatus
from webui.services import run_service

ROOT = Path(__file__).resolve().parents[1]
MIGRATION_SQL = (ROOT / FOLLOW_MIGRATION).read_text()
SCHEMA = (ROOT / "schema.sql").read_text()
SCRIPT_ID = "0b1c2d3e-0000-4000-8000-000000000001"
PG_ID = "6a1f6c1e-1111-4111-8111-111111111111"
LINK_ID = "7c2e0000-2222-4222-8222-222222222222"

SCRIPT = "\n".join([
    "[HOST]: Welcome back. Today we study a toy regression notebook.",  # 0
    "",                                                                   # 1
    "[HOST]: First it reads the points file into a dataframe.",           # 2
    "[HOST]: Then head shows the first five rows.",                       # 3
    "[PAUSE]",                                                            # 4
    "[HOST]: Why would an interviewer care about that?",                  # 5
    "[HOST]: Next it fits a straight line with least squares.",           # 6
    "[MUSIC]",                                                            # 7
    "[HOST]: That's the show.",                                           # 8
])
SPOKEN = [0, 2, 3, 5, 6, 8]

NOTEBOOK = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {},
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": "# Toy regression"},
        {"cell_type": "code", "metadata": {}, "execution_count": 1, "outputs": [],
         "source": "import pandas as pd\ndf = pd.read_csv('points.csv')\ndf.head()"},
        {"cell_type": "code", "metadata": {}, "execution_count": 2, "outputs": [],
         "source": "import numpy as np\nslope, icept = np.polyfit(df.x, df.y, 1)"},
        {"cell_type": "code", "metadata": {}, "execution_count": 3, "outputs": [],
         "source": "\n".join(f"line_{i} = {i}" for i in range(60))},
    ],
}
STEPS = [
    {"id": "step-1", "title": "Load the points", "goal": "Read the data.",
     "cells": [{"cell_index": 1, "explanation": "Reads the points and shows five rows."}]},
    {"id": "step-2", "title": "Fit a line", "goal": "Fit and check.",
     "cells": [
         {"cell_index": 2, "explanation": "Fits a least-squares line."},
         {"cell_index": 3, "explanation": "Sets sixty throwaway names."},
     ]},
]


def _script(**over):
    fields = {"id": SCRIPT_ID, "paper_id": "arxiv:study-guide", "pillar_id": "p",
              "title": "Deep dive", "script": SCRIPT}
    fields.update(over)
    return SimpleNamespace(**fields)


def _playground(**over):
    row = {"id": PG_ID, "paper_id": "arxiv:notebook-paper", "title": "Toy regression",
           "notebook": NOTEBOOK, "steps": STEPS, "status": "ready"}
    row.update(over)
    return row


def _link(**over):
    row = {"id": LINK_ID, "script_id": SCRIPT_ID, "playground_id": PG_ID,
           "status": "pending", "mapping": []}
    row.update(over)
    return row


def _parsed(value, *, input_tokens=4000, output_tokens=300, stop="end_turn"):
    return SimpleNamespace(
        parsed_output=value,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                              cache_read_input_tokens=0, cache_creation_input_tokens=0),
        stop_reason=stop,
    )


def _client(*answers):
    messages = Mock(spec=Messages)
    messages.parse.side_effect = list(answers)
    client = Mock(spec=Anthropic)
    client.messages = messages
    return client


def _draft(*spans):
    return AlignDraft(spans=[SpanDraft(line_start=a, line_end=b, cell_index=c, step_id=s)
                             for a, b, c, s in spans])


GOOD = _draft((2, 3, 1, "step-1"), (6, 6, 2, "step-2"))


def _run(client, *, playground=None, link=None, cancel=None):
    stages, saved, failures = [], {}, []
    with patch.object(follow.db, "get_podcast_script_by_id", return_value=_script()), \
         patch.object(store, "get", return_value=playground or _playground()), \
         patch.object(links, "ensure_link", return_value=link or _link()), \
         patch.object(links, "save_mapping", side_effect=lambda lid, **kw: saved.update(kw, link_id=lid)), \
         patch.object(links, "_patch", side_effect=lambda lid, fields, action: failures.append(fields)):
        try:
            result = follow.run_align_job(
                SCRIPT_ID, PG_ID, lambda n, s, d=None: stages.append((n, s, d)), cancel,
                client=client,
            )
        except Exception as exc:  # noqa: BLE001
            result = exc
    return result, stages, saved, failures


# -- inputs --------------------------------------------------------------------------


def test_spoken_lines_use_the_timeline_numbering_and_skip_cues_and_blanks():
    lines = follow.spoken_lines(SCRIPT)
    assert [i for i, _ in lines] == SPOKEN
    assert lines[1] == (2, "First it reads the points file into a dataframe.")
    assert follow.spoken_lines("") == []


def test_prompt_shows_numbered_lines_steps_cells_and_truncated_code():
    sources = {1: NOTEBOOK["cells"][1]["source"], 2: NOTEBOOK["cells"][2]["source"],
               3: NOTEBOOK["cells"][3]["source"]}
    prompt = follow.build_prompt(episode_title="Deep dive", playground_title="Toy regression",
                                 lines=follow.spoken_lines(SCRIPT), steps=STEPS, sources=sources)
    assert "L6: Next it fits a straight line with least squares." in prompt
    assert "[MUSIC]" not in prompt and "L4:" not in prompt
    assert "STEP step-2: Fit a line" in prompt
    assert "--- CELL 2 ---" in prompt and "Explanation: Fits a least-squares line." in prompt
    assert "np.polyfit" in prompt
    assert "line_29 = 29" in prompt and "line_30 = 30" not in prompt
    assert "[code truncated]" in prompt


def test_prompt_in_docs_is_the_prompt_in_code():
    assert follow.SYSTEM_PROMPT in (ROOT / "docs/PROMPTS.md").read_text()


def test_prompt_states_the_no_span_rule():
    assert "get no span" in follow.SYSTEM_PROMPT
    assert "Never invent one" in follow.SYSTEM_PROMPT
    assert "do not overlap" in follow.SYSTEM_PROMPT


# -- validation ------------------------------------------------------------------------

CELLS = follow.cell_steps(STEPS)


def test_a_good_answer_validates_completely_in_script_order():
    unordered = _draft((6, 6, 2, "step-2"), (2, 3, 1, "step-1"))
    out = follow.validate_spans(unordered, spoken=SPOKEN, cells=CELLS)
    assert out.problems == []
    assert out.spans == [
        {"line_start": 2, "line_end": 3, "cell_index": 1, "step_id": "step-1"},
        {"line_start": 6, "line_end": 6, "cell_index": 2, "step_id": "step-2"},
    ]


def test_overlapping_spans_keep_the_first_and_say_so():
    out = follow.validate_spans(
        _draft((2, 5, 1, "step-1"), (3, 6, 2, "step-2"), (8, 8, 3, "step-2")),
        spoken=SPOKEN, cells=CELLS,
    )
    assert [s["line_start"] for s in out.spans] == [2, 8]
    assert out.problems == ["Span 2 (lines 3-6) overlaps lines 2-5; dropped."]


def test_unknown_cells_wrong_steps_and_silent_lines_are_dropped():
    out = follow.validate_spans(
        _draft(
            (2, 3, 0, "step-1"),     # the markdown cell: not in the guide
            (2, 3, 42, "step-1"),    # no such cell
            (6, 6, 2, "step-1"),     # cell 2 is in step-2
            (4, 4, 1, "step-1"),     # line 4 is a [PAUSE]
            (6, 3, 2, "step-2"),     # ends before it starts
            (99, 99, 2, "step-2"),   # past the end
        ),
        spoken=SPOKEN, cells=CELLS,
    )
    assert out.spans == []
    assert len(out.problems) == 6
    assert "cell 0, which is not in the study guide" in out.problems[0]
    assert "cell 42" in out.problems[1]
    assert "cell 2 is in step-2" in out.problems[2]
    assert "spoken lines" in out.problems[3]
    assert "ends before it starts" in out.problems[4]


def test_a_malformed_answer_raises():
    for parsed in (None, {"spans": []}, "L2-3: cell 1"):
        with pytest.raises(follow.PlaygroundAlignError) as exc:
            follow.validate_spans(parsed, spoken=SPOKEN, cells=CELLS)
        assert "malformed" in str(exc.value)


# -- the job, with a mocked model ------------------------------------------------------


def test_a_good_pass_is_saved_ready_with_its_cost_in_the_stage_detail():
    client = _client(_parsed(GOOD))
    result, stages, saved, failures = _run(client)
    assert result["status"] == "ready" and result["spans"] == 2
    assert result["lines_covered"] == 3 and result["lines_total"] == 6
    assert result["model"] == "claude-haiku-5-5"
    assert saved["link_id"] == LINK_ID
    assert saved["mapping"] == follow.validate_spans(GOOD, spoken=SPOKEN, cells=CELLS).spans
    assert saved["error"] is None
    assert saved["script_sha256"] == follow.sha256_text(SCRIPT)
    assert failures == []
    final = [s for s in stages if s[1] in ("completed", "failed", "skipped")]
    assert [s[0] for s in final] == [s.value for s in PLAYGROUND_ALIGN_STAGES]
    align = [d for n, s, d in stages if n == "playground_align" and s == "completed"][0]
    assert re.search(r"\$\d+\.\d{6}", align)  # what the daily spend cap adds up
    kwargs = client.messages.parse.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-5-5"
    assert kwargs["output_format"] is AlignDraft
    assert "temperature" not in kwargs and "tool_choice" not in kwargs
    assert kwargs["thinking"] == {"type": "disabled"}
    assert "L6: Next it fits" in kwargs["messages"][0]["content"]


def test_an_overlapping_answer_is_saved_without_the_overlap_and_names_it():
    draft = _draft((2, 5, 1, "step-1"), (3, 6, 2, "step-2"))
    result, stages, saved, _ = _run(_client(_parsed(draft)))
    assert result["spans"] == 1 and result["dropped"] == 1
    assert [s["cell_index"] for s in saved["mapping"]] == [1]
    assert "overlaps lines 2-5" in saved["error"]


def test_an_answer_naming_only_unknown_cells_fails_and_records_why():
    draft = _draft((2, 3, 42, "step-1"))
    result, stages, saved, failures = _run(_client(_parsed(draft)))
    assert isinstance(result, follow.PlaygroundAlignError)
    assert "cell 42" in str(result)
    assert saved == {}
    assert failures and failures[0]["status"] == "failed" and "cell 42" in failures[0]["error"]
    failed = [d for n, s, d in stages if n == "playground_align" and s == "failed"][0]
    assert "$" in failed  # the call was paid for either way
    assert ("playground_align_persist", "skipped") in [(n, s) for n, s, _ in stages]


def test_a_malformed_answer_falls_back_then_fails_visibly():
    # No parsed output from Haiku, then none from the Sonnet fallback.
    client = _client(_parsed(None), _parsed(None))
    result, stages, saved, failures = _run(client)
    assert isinstance(result, follow.PlaygroundAlignError)
    assert "alignment call failed" in str(result)
    assert client.messages.parse.call_count == 2
    assert client.messages.parse.call_args_list[1].kwargs["model"] == "claude-sonnet-5-5"
    assert saved == {}
    assert failures[0]["status"] == "failed"


def test_an_empty_answer_is_a_valid_empty_mapping():
    result, _, saved, _ = _run(_client(_parsed(_draft())))
    assert result["spans"] == 0
    assert saved["mapping"] == []
    assert "No line of this episode" in saved["error"]


def test_a_failed_retry_keeps_the_mapping_already_there():
    ready = _link(status="ready", mapping=[{"line_start": 2, "line_end": 3,
                                            "cell_index": 1, "step_id": "step-1"}])
    result, _, saved, failures = _run(_client(RuntimeError("overloaded"), RuntimeError("overloaded")), link=ready)
    assert isinstance(result, follow.PlaygroundAlignError)
    assert failures == [{"error": "The last attempt failed: " + str(result)}]


def test_a_playground_without_a_study_guide_is_refused_before_any_call():
    client = _client()
    result, stages, saved, _ = _run(client, playground=_playground(steps=[], status="imported"))
    assert isinstance(result, follow.PlaygroundAlignError)
    assert "no study guide" in str(result)
    client.messages.parse.assert_not_called()
    assert ("playground_align_prepare", "failed") in [(n, s) for n, s, _ in stages]


def test_cancel_stops_before_the_paid_call():
    cancel = threading.Event()
    cancel.set()
    client = _client()
    result, _, saved, _ = _run(client, cancel=cancel)
    assert isinstance(result, follow.RunCancelledError)
    client.messages.parse.assert_not_called()


def test_an_unknown_episode_or_playground_is_a_missing_target():
    with patch.object(follow.db, "get_podcast_script_by_id", return_value=None):
        with pytest.raises(follow.AlignTargetMissing):
            follow.prepare(SCRIPT_ID, PG_ID)
    with patch.object(follow.db, "get_podcast_script_by_id", return_value=_script()), \
         patch.object(store, "get", return_value=None):
        with pytest.raises(follow.AlignTargetMissing):
            follow.prepare(SCRIPT_ID, PG_ID)


# -- the store and the missing-migration message ----------------------------------------


def _client_returning(error):
    client = MagicMock()
    query = client.table.return_value
    for name in ("select", "eq", "limit", "order"):
        getattr(query, name).return_value = query
    query.params = {}
    query.execute.return_value = {"data": None, "error": error}
    return client


def test_a_missing_links_table_names_migration_029():
    error = {"code": "PGRST205",
             "message": "Could not find the table 'public.playground_links' in the schema cache"}
    with patch("nlp_pillars.db.get_client", return_value=_client_returning(error)):
        with pytest.raises(links.LinkTableMissing) as exc:
            links.links_for_script(SCRIPT_ID)
    assert "docs/migrations/029_playground_links.sql" in str(exc.value)
    assert isinstance(exc.value, store.PlaygroundStoreError)


def test_other_store_failures_do_not_blame_the_migration():
    with patch("nlp_pillars.db.get_client", return_value=_client_returning({"message": "connection refused"})):
        with pytest.raises(store.PlaygroundStoreError) as exc:
            links.get_link(SCRIPT_ID, PG_ID)
    assert not isinstance(exc.value, links.LinkTableMissing)
    assert "029" not in str(exc.value)


def test_the_pass_refuses_before_the_model_when_029_is_missing():
    client = _client()
    missing = links.LinkTableMissing("Follow-along is not set up. Apply docs/migrations/029_playground_links.sql by hand.")
    with patch.object(follow.db, "get_podcast_script_by_id", return_value=_script()), \
         patch.object(store, "get", return_value=_playground()), \
         patch.object(links, "ensure_link", side_effect=missing):
        with pytest.raises(links.LinkTableMissing):
            follow.run_align_job(SCRIPT_ID, PG_ID, client=client)
    client.messages.parse.assert_not_called()


def test_malformed_ids_have_no_links_and_are_not_queried():
    with patch("nlp_pillars.db.get_client") as get_client:
        assert links.links_for_script("not-a-uuid") == []
        assert links.get_link("x", PG_ID) is None
    get_client.assert_not_called()


def test_paper_ids_are_quoted_in_the_title_lookup():
    client = MagicMock()
    query = client.table.return_value
    query.select.return_value = query
    query.params = {}
    query.execute.return_value = {"data": [{"id": "arxiv:1,2", "title": "T"}], "error": None}
    with patch("nlp_pillars.db.get_client", return_value=client):
        assert links.paper_titles(["arxiv:1,2", "", None]) == {"arxiv:1,2": "T"}
    assert query.params["id"] == 'in.("arxiv:1,2")'


# -- run kind, dispatch and migration ----------------------------------------------------


def test_dispatch_seeds_the_three_stages_and_stores_both_ids():
    scheduler = MagicMock()
    with patch.object(run_service.db, "create_pipeline_run", return_value=MagicMock(id="run-a")) as create:
        run_id = run_service.dispatch_run(
            scheduler, {}, pillar_id="p", trigger_source="ui_playground",
            kind=run_service.KIND_PLAYGROUND_ALIGN,
            script_id=SCRIPT_ID, playground_id=PG_ID,
            initial_result={"script_id": SCRIPT_ID, "playground_id": PG_ID},
        )
    assert run_id == "run-a"
    assert create.call_args.args[2] == "playground_align"
    assert create.call_args.args[3] == [s.value for s in PLAYGROUND_ALIGN_STAGES]
    assert create.call_args.kwargs["result"] == {"script_id": SCRIPT_ID, "playground_id": PG_ID}
    job = scheduler.add_job.call_args.kwargs["kwargs"]
    assert job["script_id"] == SCRIPT_ID and job["playground_id"] == PG_ID


def test_execute_run_records_a_stored_mapping_as_succeeded():
    payload = {"script_id": SCRIPT_ID, "playground_id": PG_ID, "status": "ready", "spans": 2}
    with patch.object(run_service.db, "start_pipeline_run"), \
         patch.object(run_service.db, "finish_pipeline_run") as finish, \
         patch.object(run_service.playground_follow, "run_align_job", return_value=payload), \
         patch.object(run_service.cloud_publish_service, "notify") as notify:
        run_service.execute_run("run-1", run_service.KIND_PLAYGROUND_ALIGN, "p", threading.Event(), {},
                                script_id=SCRIPT_ID, playground_id=PG_ID)
    assert finish.call_args.args[1] == RunStatus.SUCCEEDED.value
    assert finish.call_args.kwargs["result"] == payload
    notify.assert_not_called()  # the cloud reader does not carry playgrounds yet


def test_execute_run_records_a_failed_pass_as_failed():
    with patch.object(run_service.db, "start_pipeline_run"), \
         patch.object(run_service.db, "finish_pipeline_run") as finish, \
         patch.object(run_service.playground_follow, "run_align_job",
                      side_effect=follow.PlaygroundAlignError("nothing usable")):
        with pytest.raises(follow.PlaygroundAlignError):
            run_service.execute_run("run-1", run_service.KIND_PLAYGROUND_ALIGN, "p", threading.Event(), {},
                                    script_id=SCRIPT_ID, playground_id=PG_ID)
    assert finish.call_args.args[1] == RunStatus.FAILED.value
    assert finish.call_args.kwargs["error"] == "nothing usable"


def _statement(pattern):
    match = re.search(pattern, MIGRATION_SQL, re.S | re.I)
    assert match, pattern
    return match.group(0)


def test_migration_keeps_every_kind_and_adds_playground_align():
    kind = _statement(r"ADD CONSTRAINT pipeline_runs_kind_check.*?;")
    for value in ("run_daily", "process_selected", "discover", "podcast_audio", "upload",
                  "podcast_script", "podcast_factcheck", "playground_explain", "playground_align"):
        assert f"'{value}'" in kind
        assert f"'{value}'" in SCHEMA
    assert "pipeline_runs_trigger_source_check" not in MIGRATION_SQL.split("-- The trigger source")[1].split(";")[0]
    assert "CREATE TABLE IF NOT EXISTS playground_links" in MIGRATION_SQL
    assert "CREATE TABLE playground_links" in SCHEMA
    assert "UNIQUE (script_id, playground_id)" in MIGRATION_SQL
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON playground_links TO web_anon" in MIGRATION_SQL
    assert MIGRATION_SQL.rstrip().endswith("NOTIFY pgrst, 'reload schema';")
    exempt = SCHEMA.split("CREATE UNIQUE INDEX pipeline_runs_one_active_per_pillar", 1)[1].split(";", 1)[0]
    for value in ("'upload'", "'podcast_script'", "'podcast_factcheck'", "'playground_explain'",
                  "'playground_align'"):
        assert value in exempt


def test_migration_and_schema_agree_on_the_table_columns():
    def columns(sql, head):
        body = sql.split(head, 1)[1].split("\n);", 1)[0]
        return [line.split()[0] for line in body.splitlines() if re.match(r"\s{4}[a-z_0-9]+\s", line)]
    expected = ["id", "script_id", "playground_id", "mapping", "model", "status", "error",
                "script_sha256", "created_at", "updated_at"]
    assert columns(MIGRATION_SQL, "CREATE TABLE IF NOT EXISTS playground_links (") == expected
    assert columns(SCHEMA, "CREATE TABLE playground_links (") == expected


def _runs_table(index_pattern):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE pipeline_runs (id INTEGER PRIMARY KEY, pillar_id TEXT,"
                 " kind TEXT, status TEXT, result TEXT)")
    conn.execute(_statement(index_pattern))
    return conn


def _insert(conn, kind, status="running", result=None):
    conn.execute("INSERT INTO pipeline_runs (pillar_id, kind, status, result) VALUES ('p', ?, ?, ?)",
                 (kind, status, result))


def test_an_alignment_may_overlap_discovery_on_the_same_pillar():
    conn = _runs_table(r"CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_per_pillar[^;]*;")
    _insert(conn, "discover")
    _insert(conn, "playground_align")
    _insert(conn, "playground_explain")
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, "discover")


def test_one_active_alignment_per_episode_and_playground():
    conn = _runs_table(r"CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_align_per_link[^;]*;")
    pair = '{"script_id": "s1", "playground_id": "a"}'
    _insert(conn, "playground_align", result=pair)
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, "playground_align", "pending", pair)
    _insert(conn, "playground_align", "failed", pair)
    _insert(conn, "playground_align", result='{"script_id": "s1", "playground_id": "b"}')
    _insert(conn, "playground_align", result='{"script_id": "s2", "playground_id": "a"}')


def test_route_and_setting_default_to_the_cheap_haiku():
    from nlp_pillars.config import Settings
    from nlp_pillars.llm_routes import PLAYGROUND_ALIGN_ROUTE, route_for

    assert route_for("playground_align") is PLAYGROUND_ALIGN_ROUTE
    assert PLAYGROUND_ALIGN_ROUTE.default_model == "claude-haiku-5-5"
    assert PLAYGROUND_ALIGN_ROUTE.model_attrs == ("llm_playground_align_model",)
    assert PLAYGROUND_ALIGN_ROUTE.thinking == "disabled"
    assert "llm_playground_align_model" in Settings.model_fields


# -- CLI ---------------------------------------------------------------------------------


def test_cli_link_prints_what_it_matched_and_what_it_cost():
    payload = {"script_id": SCRIPT_ID, "playground_id": PG_ID, "spans": 2, "lines_covered": 3,
               "lines_total": 6, "cost_usd": 0.00042, "model": "claude-haiku-5-5", "error": None}
    with patch.object(follow, "run_align_job", return_value=payload) as job:
        result = CliRunner().invoke(cli.app, ["playground", "link", SCRIPT_ID, PG_ID])
    assert result.exit_code == 0, result.output
    assert job.call_args.args[:2] == (SCRIPT_ID, PG_ID)
    assert "2 spans over 3 of 6 spoken lines" in result.output
    assert "$0.000420" in result.output


@pytest.mark.parametrize("error", [
    links.LinkTableMissing("Follow-along is not set up. Apply docs/migrations/029_playground_links.sql by hand."),
    follow.PlaygroundAlignError("This playground has no study guide yet."),
])
def test_cli_link_refuses_with_the_reason(error):
    with patch.object(follow, "run_align_job", side_effect=error):
        result = CliRunner().invoke(cli.app, ["playground", "link", SCRIPT_ID, PG_ID])
    assert result.exit_code == 1
    assert "Not linked:" in result.output
    assert str(error).split(".")[0] in result.output


def test_db_module_still_owns_the_script_read():
    # follow.prepare reads the episode through nlp_pillars.db, the one client.
    assert follow.db is db
