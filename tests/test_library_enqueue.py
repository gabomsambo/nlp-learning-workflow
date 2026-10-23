"""Enqueue of a library paper onto paper_queue.

Search hits are already in `papers`. `add_to_paper_queue` skips those, which is
the discovery dedupe. An explicit Enqueue still has to land on the same table
the auto-run pops.
"""

from unittest.mock import Mock

import pytest

from nlp_pillars.db import LibraryEnqueueError, enqueue_library_paper


def _client(papers_data, queue_data, insert_result=None):
    papers = Mock()
    papers.select.return_value = papers
    papers.eq.return_value = papers
    papers.execute.return_value = {"data": papers_data, "error": None}

    queued = Mock()
    queued.select.return_value = queued
    queued.eq.return_value = queued
    queued.execute.return_value = {"data": queue_data, "error": None}

    inserted = Mock()
    inserted.insert.return_value = insert_result or {"data": [{"id": "row"}], "error": None}

    client = Mock()
    client.table.side_effect = [papers, queued, inserted]
    return client, inserted


def test_library_paper_is_written_to_the_queue(monkeypatch):
    client, inserted = _client(
        [{"id": "file:abc", "title": "Attention", "url_pdf": "https://example.com/a.pdf"}],
        [],
    )
    monkeypatch.setattr("nlp_pillars.db.get_client", lambda: client)

    result = enqueue_library_paper("ai-agents-tool-use", "file:abc")

    assert result == {"status": "queued", "paper_id": "file:abc", "title": "Attention"}
    payload = inserted.insert.call_args[0][0]
    assert payload["pillar_id"] == "ai-agents-tool-use"
    assert payload["paper_id"] == "file:abc"
    assert payload["processed"] is False
    assert payload["url_pdf"] == "https://example.com/a.pdf"
    assert payload["source"] == "url"


def test_second_enqueue_does_not_insert(monkeypatch):
    client, inserted = _client(
        [{"id": "file:abc", "title": "Attention", "url_pdf": "https://example.com/a.pdf"}],
        [{"paper_id": "file:abc"}],
    )
    monkeypatch.setattr("nlp_pillars.db.get_client", lambda: client)

    result = enqueue_library_paper("ai-agents-tool-use", "file:abc")

    assert result["status"] == "already_queued"
    inserted.insert.assert_not_called()


def test_missing_paper_is_not_found(monkeypatch):
    client, _inserted = _client([], [])
    monkeypatch.setattr("nlp_pillars.db.get_client", lambda: client)

    with pytest.raises(LibraryEnqueueError) as caught:
        enqueue_library_paper("ai-agents-tool-use", "file:missing")
    assert caught.value.code == "not_found"


def test_paper_without_a_pdf_is_refused(monkeypatch):
    client, inserted = _client(
        [{"id": "file:abc", "title": "No file", "url_pdf": None}],
        [],
    )
    monkeypatch.setattr("nlp_pillars.db.get_client", lambda: client)

    with pytest.raises(LibraryEnqueueError) as caught:
        enqueue_library_paper("ai-agents-tool-use", "file:abc")
    assert caught.value.code == "unresolvable"
    inserted.insert.assert_not_called()
