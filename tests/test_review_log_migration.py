"""Migration 025 and schema.sql agree, and the migration is safe to hand-apply twice."""

import re
from pathlib import Path

from nlp_pillars.db import TableQuery, _dict_to_review_log, _review_log_to_dict
from nlp_pillars.schemas import FSRSRating, ReviewLog

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (ROOT / "docs" / "migrations" / "025_review_log_origin.sql").read_text()
SCHEMA = (ROOT / "schema.sql").read_text()


def _review_logs_block():
    return re.search(r"CREATE TABLE review_logs \((.*?)\n\);", SCHEMA, re.S).group(1)


def test_migration_is_idempotent_granted_and_reloads_postgrest():
    assert "ADD COLUMN IF NOT EXISTS device_id TEXT" in MIGRATION
    assert "ADD COLUMN IF NOT EXISTS origin TEXT NOT NULL DEFAULT 'desktop'" in MIGRATION
    assert "CREATE INDEX IF NOT EXISTS idx_review_logs_card_fold" in MIGRATION
    assert "IF NOT EXISTS (" in MIGRATION  # the CHECK is added only once
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON review_logs TO web_anon" in MIGRATION
    assert "NOTIFY pgrst, 'reload schema'" in MIGRATION
    assert "docker exec -i nlp_postgres psql" in MIGRATION


def test_schema_sql_carries_what_the_migration_adds():
    block = _review_logs_block()
    assert "device_id TEXT" in block
    assert "origin TEXT NOT NULL DEFAULT 'desktop' CHECK (origin IN ('desktop', 'cloud'))" in block
    assert "idx_review_logs_card_fold ON review_logs(card_id, review_timestamp, id)" in SCHEMA


def test_the_check_in_migration_and_schema_name_the_same_origins():
    assert "origin IN ('desktop', 'cloud')" in MIGRATION
    assert "origin IN ('desktop', 'cloud')" in SCHEMA


def _log(**extra):
    return ReviewLog(
        card_id="c", pillar_id="p", paper_id="x", rating=FSRSRating.GOOD,
        difficulty=1.0, stability=2.0, **extra,
    )


def test_a_desktop_review_row_omits_the_new_columns_so_an_unmigrated_database_takes_it():
    row = _review_log_to_dict(_log(id="00000000-0000-4000-8000-000000000001"))
    assert row["id"] == "00000000-0000-4000-8000-000000000001"
    assert "device_id" not in row and "origin" not in row


def test_a_log_without_an_id_lets_the_database_pick_one():
    assert "id" not in _review_log_to_dict(_log())


def test_origin_and_device_round_trip_when_set():
    row = _review_log_to_dict(_log(device_id="iphone", origin="cloud"))
    assert row["device_id"] == "iphone" and row["origin"] == "cloud"
    back = _dict_to_review_log({**row, "review_timestamp": "2026-03-10T09:00:00+00:00"})
    assert back.device_id == "iphone" and back.origin == "cloud"


def test_in_filter_renders_a_postgrest_in_list():
    query = TableQuery(None, "http://x", "review_logs").in_("card_id", ["a", "b"])
    assert query.params["card_id"] == "in.(a,b)"
