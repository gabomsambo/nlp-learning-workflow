"""Semantic Scholar key rejection. No network and no .env key."""

import logging
import time

import httpx
import pytest

from nlp_pillars.tools.semantic_scholar_tool import (
    KEY_REJECTED_LOG,
    SemanticScholarError,
    SemanticScholarTool,
)


def _tool(api_key="rejected-key-value"):
    tool = SemanticScholarTool(api_key=api_key)
    tool.last_request_time = time.time()
    return tool


def _response(code, body):
    request = httpx.Request(
        "GET", "https://api.semanticscholar.org/graph/v1/paper/x"
    )
    return httpx.Response(code, json=body, request=request)


def _citing(code_key):
    def get(url, params=None, headers=None):
        code_key.append(None if headers is None else headers.get("x-api-key"))
        if headers and headers.get("x-api-key"):
            return _response(403, {"message": "Forbidden"})
        return _response(
            200,
            {
                "data": [
                    {
                        "citingPaper": {
                            "paperId": "abc",
                            "externalIds": {"ArXiv": "1706.03762"},
                            "title": "Attention",
                            "authors": [{"name": "Vaswani"}],
                            "year": 2017,
                        },
                        "isInfluential": False,
                    }
                ]
            },
        )

    return get


def test_403_retries_once_without_the_key_and_logs_once(caplog):
    tool = _tool()
    seen = []
    tool.client.get = _citing(seen)
    with caplog.at_level(logging.ERROR):
        rows = tool.get_citations("1706.03762", limit=1)
        rows_again = tool.get_citations("1706.03762", limit=1)
    assert seen == ["rejected-key-value", None, None]
    assert rows[0][0].id == "1706.03762"
    assert rows_again[0][0].id == "1706.03762"
    assert tool.key_rejected is True
    assert "x-api-key" not in tool.headers
    assert "rejected-key-value" not in caplog.text
    assert caplog.text.count("SEMANTIC_SCHOLAR_API_KEY") == 1
    assert "403" in caplog.text
    assert "not changed" in KEY_REJECTED_LOG
    assert KEY_REJECTED_LOG in caplog.text


def test_403_after_the_keyless_retry_is_a_failure_not_zero_citations():
    tool = _tool()
    seen = []

    def get(url, params=None, headers=None):
        seen.append(None if headers is None else headers.get("x-api-key"))
        return _response(403, {"message": "Forbidden"})

    tool.client.get = get
    with pytest.raises(SemanticScholarError, match="403"):
        tool.get_citations("1706.03762", limit=1)
    assert seen == ["rejected-key-value", None]


def test_404_is_still_zero_citations():
    tool = _tool(api_key="present-key")
    tool.client.get = lambda url, params=None, headers=None: _response(
        404, {"message": "not found"}
    )
    assert tool.get_citations("1706.03762", limit=1) == []
