"""db.set_claim_dismissal: one RPC, a clean error, and the migration named."""

from unittest.mock import MagicMock, patch

import pytest

from nlp_pillars import db

CLAIM = "c1a1a1a1-1111-4111-8111-111111111111"


def test_dismiss_calls_the_rpc_with_the_note():
    client = MagicMock()
    client.rpc.return_value = {"data": {"id": CLAIM, "dismissed_note": "ok"}, "error": None}
    with patch("nlp_pillars.db.get_client", return_value=client):
        row = db.set_claim_dismissal(CLAIM, True, "ok")
    assert row["dismissed_note"] == "ok"
    name, params = client.rpc.call_args.args
    assert name == "set_podcast_claim_dismissal"
    assert params == {"p_claim_id": CLAIM, "p_dismissed": True, "p_note": "ok"}


def test_unknown_claim_is_none_not_an_error():
    client = MagicMock()
    client.rpc.return_value = {"data": None, "error": None}
    with patch("nlp_pillars.db.get_client", return_value=client):
        assert db.set_claim_dismissal(CLAIM, False) is None


def test_missing_function_names_migration_031():
    client = MagicMock()
    client.rpc.return_value = {
        "data": None,
        "error": {
            "code": "PGRST202",
            "message": "Could not find the function public.set_podcast_claim_dismissal in the schema cache",
        },
    }
    with patch("nlp_pillars.db.get_client", return_value=client):
        with pytest.raises(db.ClaimDismissalError, match="033_claim_dismissals.sql"):
            db.set_claim_dismissal(CLAIM, True, "x")


def test_a_bad_id_never_reaches_the_database():
    with patch("nlp_pillars.db.get_client") as get_client:
        with pytest.raises(db.ClaimDismissalError):
            db.set_claim_dismissal("nope", True, "x")
    get_client.assert_not_called()


def test_migration_and_schema_agree_on_the_objects():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    migration = (root / "docs/migrations/033_claim_dismissals.sql").read_text()
    schema = (root / "schema.sql").read_text()
    for needle in (
        "CREATE TABLE podcast_claim_reviews",
        "CREATE OR REPLACE FUNCTION set_podcast_claim_dismissal(",
        "GRANT EXECUTE ON FUNCTION set_podcast_claim_dismissal(UUID, BOOLEAN, TEXT) TO web_anon;",
    ):
        assert needle.replace("CREATE TABLE ", "CREATE TABLE ") in schema
    assert "CREATE TABLE IF NOT EXISTS podcast_claim_reviews" in migration
    assert "dismissal needs a note" in migration and "dismissal needs a note" in schema
