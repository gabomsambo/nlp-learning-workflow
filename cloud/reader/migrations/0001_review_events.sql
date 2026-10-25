-- D1 migration 0001: the phone's review events.
--
-- This table is the only thing the reader can write. Each row is one card review,
-- made on a device and immutable once stored. The computer pulls the rows into its
-- own review log (nlp_pillars/cloud/reviews.py) and the log, not this table, is
-- what card schedules are derived from; this is a mailbox, not a source of truth.
--
--   seq        arrival order, never reused (an ignored duplicate can leave a gap).
--              The computer's cursor and the phone's "everything after the schedule
--              snapshot" both count in seq.
--   event_id   the review's identity (a UUID the device made). UNIQUE, so sending an
--              event twice stores it once.
--   reviewed_at / received_at   ISO-8601 UTC with milliseconds. The first is the
--              device's clock, the second the Worker's.
--
-- No free text: every column is an id, a number or a timestamp, and the Worker
-- (src/reviews.js) checks each against a fixed shape before it gets here.
--
--   wrangler d1 migrations apply nlp-reader-reviews --remote

CREATE TABLE IF NOT EXISTS review_events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE CHECK (length(event_id) = 36),
  card_id TEXT NOT NULL CHECK (length(card_id) = 36),
  rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 4),
  reviewed_at TEXT NOT NULL,
  device_id TEXT NOT NULL CHECK (length(device_id) BETWEEN 8 AND 64),
  response_time_ms INTEGER CHECK (response_time_ms IS NULL OR response_time_ms BETWEEN 0 AND 3600000),
  session_id TEXT CHECK (session_id IS NULL OR length(session_id) BETWEEN 8 AND 64),
  received_at TEXT NOT NULL
);
