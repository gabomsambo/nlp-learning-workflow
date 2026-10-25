"""Where the review sync has got to: ``cloud_sync_state`` (migration 027).

Two numbers, both counted in the cloud's event ``seq``:

* ``reviews_seq``: every event up to here is in ``review_logs`` and its card is
  re-derived. ``cloud export`` publishes it as ``through_seq``.
* ``reviews_pending``: the highest seq a pull is applying. Written before the
  cards are touched and set equal to ``reviews_seq`` afterwards, so a gap between
  the two means a pull stopped half-way and the cards may hold some events the
  cursor does not claim yet.

``APPLY_LOCK`` keeps a pull's apply step and an export's read of the cards and the
cursor from interleaving inside one process. The two numbers cover what the lock
cannot: a crash, and a second process.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

MIGRATION = "docs/migrations/027_cloud_sync_state.sql"
SEQ_KEY = "reviews_seq"
PENDING_KEY = "reviews_pending"

# Held while a pull applies a page of events and while an export reads the cards.
APPLY_LOCK = threading.RLock()


class SyncStateError(Exception):
    """The cursor could not be read or written."""


class SyncStateMissing(SyncStateError):
    """``cloud_sync_state`` does not exist: migration 027 has not been applied."""


@dataclass(frozen=True)
class ReviewCursor:
    seq: int = 0
    pending: int = 0

    @property
    def unfinished(self) -> bool:
        """A pull started applying events past ``seq`` and has not finished."""
        return self.pending > self.seq


def _missing(text: str) -> bool:
    lowered = text.lower()
    return "cloud_sync_state" in lowered and any(
        token in lowered for token in ("pgrst205", "pgrst204", "does not exist", "schema cache", "404")
    )


class PostgrestSyncState:
    """``cloud_sync_state`` over the project's PostgREST client."""

    def __init__(self, client: Optional[Any] = None) -> None:
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from .. import db

            self._client = db.get_client()
        return self._client

    @staticmethod
    def _fail(action: str, error: Any) -> SyncStateError:
        text = str(error)
        if _missing(text):
            return SyncStateMissing(f"{action}: cloud_sync_state is missing. Apply {MIGRATION} by hand.")
        return SyncStateError(f"{action}: {text}")

    def read(self) -> ReviewCursor:
        try:
            response = self.client.table("cloud_sync_state").select("key,value").order("key").limit(10).execute()
        except Exception as exc:  # noqa: BLE001
            raise self._fail("Could not read the review cursor", exc) from exc
        if response["error"]:
            raise self._fail("Could not read the review cursor", response["error"])
        values = {row["key"]: int(row["value"]) for row in response["data"] or []}
        return ReviewCursor(seq=values.get(SEQ_KEY, 0), pending=values.get(PENDING_KEY, 0))

    def write(self, key: str, value: int) -> None:
        client = self.client
        row = {
            "key": key, "value": int(value),
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        try:
            response = client.client.post(
                f"{client.base_url}/cloud_sync_state",
                params={"on_conflict": "key"},
                json=[row],
                headers={**client.headers, "Prefer": "resolution=merge-duplicates,return=minimal"},
            )
        except Exception as exc:  # noqa: BLE001
            raise self._fail(f"Could not save {key}", exc) from exc
        if response.status_code >= 400:
            raise self._fail(f"Could not save {key}", f"HTTP {response.status_code} {response.text[:200]}")
