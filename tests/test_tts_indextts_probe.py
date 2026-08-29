"""Tests for IndexTTS Gradio contract probing."""

from nlp_pillars.tts.indextts_client import GEN_SINGLE_PARAMS, IndexTtsClient


def test_gen_single_params_match_contract():
    assert len(GEN_SINGLE_PARAMS) == 24
    assert GEN_SINGLE_PARAMS[0] == "emo_control_method"
    assert GEN_SINGLE_PARAMS[2] == "text"
    assert GEN_SINGLE_PARAMS[-1] == "param_23"


def test_not_running_when_connection_fails(tmp_path):
    client = IndexTtsClient(
        "http://127.0.0.1:59999",
        tmp_path,
        timeout_seconds=0.2,
    )
    info = client.check_status()
    assert info.status.value == "not_running"
    assert "not running" in info.message.lower()


def test_wrong_service_when_dots_endpoint_present(tmp_path, monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "named_endpoints": {
                    "/run_synthesis": {"parameters": []},
                }
            }

    monkeypatch.setattr(
        "nlp_pillars.tts.indextts_client.requests.get",
        lambda *args, **kwargs: FakeResponse(),
    )
    client = IndexTtsClient("http://127.0.0.1:7860", tmp_path)
    info = client.check_status()
    assert info.status.value == "wrong_service"
    assert "Dots" in info.message or "/run_synthesis" in info.message


def test_ready_when_contract_matches(tmp_path, monkeypatch):
    params = [{"parameter_name": name} for name in GEN_SINGLE_PARAMS]

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"named_endpoints": {"/gen_single": {"parameters": params}}}

    monkeypatch.setattr(
        "nlp_pillars.tts.indextts_client.requests.get",
        lambda *args, **kwargs: FakeResponse(),
    )
    client = IndexTtsClient("http://127.0.0.1:7861", tmp_path)
    info = client.check_status()
    assert info.status.value == "ready"


def test_normalize_output_path_from_gradio_update_dict():
    raw = {
        "visible": True,
        "value": "/tmp/spk_123.wav",
        "__type__": "update",
    }
    assert IndexTtsClient._normalize_output_path(raw) == "/tmp/spk_123.wav"
