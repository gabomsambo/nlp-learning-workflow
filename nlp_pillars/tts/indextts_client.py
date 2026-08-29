"""IndexTTS Gradio client — liveness probe and /gen_single synthesis."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Callable, List, Optional
from urllib.parse import urlparse

import requests

from nlp_pillars.tts.engine import (
    SynthesisProgress,
    TtsEngineStatus,
    TtsStatusInfo,
)

logger = logging.getLogger(__name__)

# Exact /gen_single parameter list from the installed IndexTTS WebUI (Gradio 5.45).
GEN_SINGLE_PARAMS: List[str] = [
    "emo_control_method",
    "prompt",
    "text",
    "emo_ref_path",
    "emo_weight",
    "vec1",
    "vec2",
    "vec3",
    "vec4",
    "vec5",
    "vec6",
    "vec7",
    "vec8",
    "emo_text",
    "emo_random",
    "max_text_tokens_per_segment",
    "param_16",
    "param_17",
    "param_18",
    "param_19",
    "param_20",
    "param_21",
    "param_22",
    "param_23",
]

DEFAULT_START_COMMAND = (
    "cd /home/gabo/index-tts && uv run webui.py --host 0.0.0.0 --port 7861"
)

_PROGRESS_RE = re.compile(r"speech synthesis\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


class IndexTtsClient:
    """Call the captain-run IndexTTS Gradio service."""

    def __init__(
        self,
        base_url: str,
        download_dir: Path,
        *,
        start_command: str = DEFAULT_START_COMMAND,
        timeout_seconds: float = 3.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.download_dir = Path(download_dir)
        self.start_command = start_command
        self.timeout_seconds = timeout_seconds
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def check_status(self) -> TtsStatusInfo:
        """Two-part contract probe — not a bare port check."""
        info_url = f"{self.base_url}/gradio_api/info"
        try:
            response = requests.get(info_url, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException:
            port = urlparse(self.base_url).port or 7861
            return TtsStatusInfo(
                status=TtsEngineStatus.NOT_RUNNING,
                message=(
                    f"IndexTTS is not running on port {port}. "
                    "Start it on the host with the command below."
                ),
                start_command=self.start_command,
                base_url=self.base_url,
            )

        try:
            payload = response.json()
        except ValueError:
            return TtsStatusInfo(
                status=TtsEngineStatus.WRONG_SERVICE,
                message=(
                    f"Something on {self.base_url} answered but did not return "
                    "Gradio API JSON."
                ),
                start_command=self.start_command,
                base_url=self.base_url,
            )

        endpoints = payload.get("named_endpoints") or {}
        if "/gen_single" not in endpoints:
            wrong = "/run_synthesis" in endpoints
            if wrong:
                message = (
                    f"Port {urlparse(self.base_url).port or '?'} is occupied by "
                    "Dots TTS (/run_synthesis), not IndexTTS (/gen_single). "
                    "Start IndexTTS on a different port or stop Dots first."
                )
            else:
                message = (
                    f"A Gradio app on {self.base_url} is running but it is not "
                    "IndexTTS — /gen_single is missing."
                )
            return TtsStatusInfo(
                status=TtsEngineStatus.WRONG_SERVICE,
                message=message,
                start_command=self.start_command,
                base_url=self.base_url,
            )

        endpoint = endpoints["/gen_single"]
        actual = [p["parameter_name"] for p in endpoint.get("parameters", [])]
        if actual != GEN_SINGLE_PARAMS:
            return TtsStatusInfo(
                status=TtsEngineStatus.CONTRACT_MISMATCH,
                message=(
                    "IndexTTS is running but /gen_single has unexpected parameters. "
                    "The WebUI may have been upgraded — integration needs updating."
                ),
                start_command=self.start_command,
                base_url=self.base_url,
            )

        return TtsStatusInfo(
            status=TtsEngineStatus.READY,
            message="IndexTTS is ready.",
            start_command=self.start_command,
            base_url=self.base_url,
        )

    def synthesize(
        self,
        text: str,
        voice_path: str,
        *,
        on_progress: Optional[Callable[[SynthesisProgress], None]] = None,
        poll_interval: float = 0.5,
        job_timeout: float = 3600.0,
    ) -> str:
        """Submit one synthesis job and return the downloaded WAV path."""
        from gradio_client import Client, handle_file

        client = Client(self.base_url, download_files=str(self.download_dir))
        job = client.submit(
            "Same as the voice reference",
            handle_file(voice_path),
            text,
            None,
            0.65,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            "",
            False,
            120,
            True,
            0.8,
            30,
            0.8,
            0.0,
            3,
            10.0,
            1500,
            api_name="/gen_single",
        )

        deadline = time.monotonic() + job_timeout
        last_progress: Optional[SynthesisProgress] = None
        while not job.done():
            if time.monotonic() > deadline:
                raise TimeoutError("IndexTTS synthesis timed out")
            status = job.status()
            progress = self._parse_progress(status.progress_data)
            if progress and (
                last_progress is None
                or progress.current != last_progress.current
                or progress.total != last_progress.total
            ):
                last_progress = progress
                if on_progress:
                    on_progress(progress)
            time.sleep(poll_interval)

        result = job.result()
        if not result:
            raise RuntimeError("IndexTTS returned no audio file")
        return str(result)

    @staticmethod
    def _parse_progress(progress_data) -> Optional[SynthesisProgress]:
        if not progress_data:
            return None
        for entry in progress_data:
            desc = getattr(entry, "desc", None) or (
                entry.get("desc") if isinstance(entry, dict) else None
            )
            if not desc:
                continue
            match = _PROGRESS_RE.search(str(desc))
            if match:
                return SynthesisProgress(
                    current=int(match.group(1)),
                    total=int(match.group(2)),
                    description=str(desc),
                )
        return None
