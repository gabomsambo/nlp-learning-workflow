"""Listen page view helpers: links, grouping, and which pane opens."""

from datetime import datetime, timezone

from nlp_pillars.schemas import AudioMetadata, PodcastScript

from webui.listen_view import (
    audio_href,
    build_episode,
    format_created,
    format_duration,
    group_episodes,
    listen_href,
    paper_page_href,
    view_is_studio,
)


def test_paper_links_encode_ids_that_contain_a_colon():
    assert paper_page_href("arxiv:1706.03762") == "/papers/arxiv%3A1706.03762"
    assert paper_page_href("") is None
    assert paper_page_href(None) is None


def test_audio_href_rejects_path_escape():
    assert audio_href("episode.mp3") == "/api/podcast/audio/episode.mp3"
    assert audio_href("../secret.mp3") is None
    assert audio_href("a/b.mp3") is None
    assert audio_href("a\\b.mp3") is None
    assert audio_href(None) is None


def test_duration_and_created_labels():
    assert format_duration(125) == "2:05"
    assert format_duration(0) == ""
    assert format_duration(None) == ""
    assert format_duration(3661) == "1:01:01"
    created = datetime(2026, 10, 23, 22, 1, tzinfo=timezone.utc)
    assert format_created(created) == "23 Oct 2026"
    assert format_created("2026-01-05T00:00:00+00:00") == "5 Jan 2026"
    assert format_created(None) == ""


def test_studio_opens_for_a_paper_or_a_run_and_not_for_a_script_alone():
    assert view_is_studio({}) is False
    assert view_is_studio({"script": "abc"}) is False
    assert view_is_studio({"studio": "1"}) is True
    assert view_is_studio({"paper": "arxiv:1"}) is True
    assert view_is_studio({"run": "run-1"}) is True
    assert listen_href() == "/podcast/"
    assert listen_href(studio=True, paper="arxiv:1706.03762") == (
        "/podcast/?studio=1&paper=arxiv%3A1706.03762"
    )


def test_episodes_group_by_pillar_with_the_newest_group_first():
    older = _script(
        "11111111-1111-1111-1111-111111111111",
        "Older",
        "p-old",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
        file_name="old.mp3",
        duration=90,
    )
    newer = _script(
        "22222222-2222-2222-2222-222222222222",
        "Newer",
        "p-new",
        datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    same_pillar = _script(
        "33333333-3333-3333-3333-333333333333",
        "Also new",
        "p-new",
        datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    rows = [
        build_episode(older, {"arxiv:1": "Attention"}, {"p-old": "Old pillar", "p-new": "New pillar"}),
        build_episode(newer, {"arxiv:1": "Attention"}, {"p-old": "Old pillar", "p-new": "New pillar"}),
        build_episode(same_pillar, {"arxiv:1": "Attention"}, {"p-old": "Old pillar", "p-new": "New pillar"}),
    ]
    groups = group_episodes(rows)
    assert [group["pillar_name"] for group in groups] == ["New pillar", "Old pillar"]
    assert [ep["title"] for ep in groups[0]["episodes"]] == ["Newer", "Also new"]
    assert groups[1]["episodes"][0]["paper_href"] == "/papers/arxiv%3A1"
    assert groups[1]["episodes"][0]["audio_href"] == "/api/podcast/audio/old.mp3"
    assert groups[1]["episodes"][0]["duration_label"] == "1:30"
    assert groups[0]["episodes"][0]["audio_href"] is None


def _script(script_id, title, pillar_id, created, file_name=None, duration=None):
    return PodcastScript(
        id=script_id,
        paper_id="arxiv:1",
        pillar_id=pillar_id,
        title=title,
        script="[HOST]: Hello",
        word_count=10,
        created_at=created,
        audio_metadata=AudioMetadata(file_name=file_name, duration_seconds=duration),
    )
