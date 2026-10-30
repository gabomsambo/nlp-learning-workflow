"""The evidence-ledger podcast writer.

``PODCAST_WRITER_MODE=evidence`` writes the script from a ledger of facts the
paper states, each backed by a quote that code checked against the paper text,
and every ``[HOST]`` line ends with the ledger ids it relies on. Design and
measurements: ``docs/PODCAST_EVIDENCE_WRITER.md``.

* :mod:`.tags` parses and strips the ``[[F12,T3]]`` citation tags.
* :mod:`.ledger` builds and verifies the ledger.
* :mod:`.prompts` holds the versioned prompts.
"""

WRITER_MODES = ("free", "evidence")
DEFAULT_WRITER_MODE = "free"


def resolve_writer_mode(value=None) -> str:
    """``free`` or ``evidence``. ``None`` reads ``PODCAST_WRITER_MODE``.

    Raises:
        ValueError: anything else. A typo must not quietly select the writer
            nobody meant to run.
    """
    if value is None:
        from ..config import get_settings

        value = getattr(get_settings(), "podcast_writer_mode", DEFAULT_WRITER_MODE)
    if not isinstance(value, str):
        value = DEFAULT_WRITER_MODE
    mode = value.strip().lower() or DEFAULT_WRITER_MODE
    if mode not in WRITER_MODES:
        raise ValueError(
            f"PODCAST_WRITER_MODE must be one of {', '.join(WRITER_MODES)}, got {value!r}"
        )
    return mode
