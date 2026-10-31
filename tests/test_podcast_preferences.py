"""The Studio remembers its last confirmed voice in server-side storage."""

from unittest.mock import Mock, patch

import pytest

from nlp_pillars import db


def _client_with_table():
    client = Mock()
    table = client.table.return_value
    table.select.return_value = table
    table.eq.return_value = table
    table.limit.return_value = table
    return client, table


def test_reads_the_last_voice_from_the_singleton_setting_row():
    client, table = _client_with_table()
    table.execute.return_value = {
        "data": [{"last_voice_path": "Podcasts/voices/host.wav"}],
        "error": None,
    }

    with patch.object(db, "get_client", return_value=client):
        assert db.get_last_podcast_voice_path() == "Podcasts/voices/host.wav"

    client.table.assert_called_once_with("podcast_settings")
    table.select.assert_called_once_with("last_voice_path")
    table.eq.assert_called_once_with("id", "true")


def test_updates_the_singleton_setting_after_voice_confirmation():
    client, table = _client_with_table()
    table.update.return_value = {"data": [{"id": True}], "error": None}

    with patch.object(db, "get_client", return_value=client):
        db.set_last_podcast_voice_path("Podcasts/voices/host.wav")

    payload = table.update.call_args.args[0]
    assert payload["last_voice_path"] == "Podcasts/voices/host.wav"
    assert payload["updated_at"]


def test_missing_singleton_row_is_not_reported_as_a_saved_preference():
    client, table = _client_with_table()
    table.update.return_value = {"data": [], "error": None}

    with patch.object(db, "get_client", return_value=client):
        with pytest.raises(db.PodcastPreferenceError, match="singleton row is missing"):
            db.set_last_podcast_voice_path("host.wav")
