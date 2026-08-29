"""TTS engine boundary — IndexTTS for v1, room for a second backend later."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Protocol


class TtsEngineStatus(str, Enum):
    """Distinct user-facing states for the IndexTTS service."""

    READY = "ready"
    NOT_RUNNING = "not_running"
    WRONG_SERVICE = "wrong_service"
    CONTRACT_MISMATCH = "contract_mismatch"


@dataclass(frozen=True)
class TtsStatusInfo:
    status: TtsEngineStatus
    message: str
    start_command: str
    base_url: str


@dataclass(frozen=True)
class SynthesisProgress:
    """Progress parsed from Gradio Job.status().progress_data."""

    current: int
    total: int
    description: str


class TtsEngine(Protocol):
    """Minimal interface a host TTS backend must implement."""

    def check_status(self) -> TtsStatusInfo:
        ...

    def synthesize(
        self,
        text: str,
        voice_path: str,
        *,
        on_progress: Optional[Callable[[SynthesisProgress], None]] = None,
    ) -> str:
        """Return path to a mono 22.05 kHz WAV file."""
