/*
 * Review events: the one thing the phone writes.
 *
 *   POST /api/reviews           the signed-in reader sends events (batch)
 *   GET  /api/reviews?since=N   events after seq N, so a second device can catch up
 *   GET  /ingest/reviews?after=N  the computer pulls them (service token, ingest.js)
 *
 * An event is `{event_id, card_id, rating, reviewed_at, device_id}` plus an optional
 * `response_time_ms` and `session_id`. Every field has a fixed shape, no key is
 * accepted that is not listed, and there is no free text, so nothing stored here
 * can carry an instruction anywhere: the computer applies an event with a pure
 * function and never hands it to a model, a shell or a query string.
 *
 * Each event in a POST gets its own answer, so one bad event cannot hold up the
 * rest and the phone knows which ones to stop retrying:
 *   accepted   stored now, with its `seq`
 *   duplicate  already stored with the same content (a retry); `seq` as stored
 *   rejected   will never be stored, with a short `reason`
 *
 * The only binding used is REVIEWS (D1). No request leaves the Worker.
 */

export const MAX_BATCH = 50;
export const MAX_BODY_BYTES = 32 * 1024;
export const DEFAULT_PAGE = 200;
export const MAX_PAGE = 500;
// The same window `nlp_pillars/fsrs_fold.py` applies: a clock that is wrong by more
// than this is not a review time.
export const MAX_FUTURE_MS = 5 * 60 * 1000;
export const MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;

const ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const SLUG = /^[A-Za-z0-9_-]{8,64}$/;
const TIMESTAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;
const EVENT_KEYS = new Set(['event_id', 'card_id', 'rating', 'reviewed_at', 'device_id', 'response_time_ms', 'session_id']);
const MAX_RESPONSE_MS = 3600000;

export class ReviewsError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function reply(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
  });
}

const isoMs = (ms) => new Date(ms).toISOString();

/**
 * `{event}` when `raw` is a well-formed review at `now`, else `{reason}`.
 * The returned event holds exactly the stored fields, nothing the client added.
 */
export function validateEvent(raw, now) {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) return { reason: 'not an object' };
  for (const key of Object.keys(raw)) {
    if (!EVENT_KEYS.has(key)) return { reason: `unknown field ${key.slice(0, 24)}` };
  }
  if (typeof raw.event_id !== 'string' || !ID.test(raw.event_id)) return { reason: 'event_id must be a lowercase UUID' };
  if (typeof raw.card_id !== 'string' || !ID.test(raw.card_id)) return { reason: 'card_id must be a lowercase UUID' };
  if (!Number.isInteger(raw.rating) || raw.rating < 1 || raw.rating > 4) return { reason: 'rating must be 1 to 4' };
  if (typeof raw.device_id !== 'string' || !SLUG.test(raw.device_id)) return { reason: 'device_id has the wrong shape' };

  if (typeof raw.reviewed_at !== 'string' || !TIMESTAMP.test(raw.reviewed_at)) {
    return { reason: 'reviewed_at must be an ISO UTC time with milliseconds' };
  }
  const at = Date.parse(raw.reviewed_at);
  if (Number.isNaN(at) || isoMs(at) !== raw.reviewed_at) return { reason: 'reviewed_at is not a real time' };
  if (at > now + MAX_FUTURE_MS) return { reason: 'reviewed_at is in the future' };
  if (at < now - MAX_AGE_MS) return { reason: 'reviewed_at is more than 30 days old' };

  let responseTime = null;
  if (raw.response_time_ms !== undefined && raw.response_time_ms !== null) {
    if (!Number.isInteger(raw.response_time_ms) || raw.response_time_ms < 0 || raw.response_time_ms > MAX_RESPONSE_MS) {
      return { reason: 'response_time_ms must be a whole number of milliseconds up to an hour' };
    }
    responseTime = raw.response_time_ms;
  }
  let session = null;
  if (raw.session_id !== undefined && raw.session_id !== null) {
    if (typeof raw.session_id !== 'string' || !SLUG.test(raw.session_id)) return { reason: 'session_id has the wrong shape' };
    session = raw.session_id;
  }
  return {
    event: {
      event_id: raw.event_id,
      card_id: raw.card_id,
      rating: raw.rating,
      reviewed_at: raw.reviewed_at,
      device_id: raw.device_id,
      response_time_ms: responseTime,
      session_id: session,
    },
  };
}

const sameEvent = (stored, event) => stored.card_id === event.card_id
  && stored.rating === event.rating
  && stored.reviewed_at === event.reviewed_at
  && stored.device_id === event.device_id;

const INSERT = 'INSERT OR IGNORE INTO review_events '
  + '(event_id, card_id, rating, reviewed_at, device_id, response_time_ms, session_id, received_at) '
  + 'VALUES (?, ?, ?, ?, ?, ?, ?, ?)';

function placeholders(n) {
  return Array.from({ length: n }, () => '?').join(', ');
}

function database(env) {
  if (!env.REVIEWS) throw new ReviewsError(503, 'reviews are not configured');
  return env.REVIEWS;
}

async function readBody(request) {
  const type = (request.headers.get('content-type') || '').split(';')[0].trim().toLowerCase();
  if (type !== 'application/json') throw new ReviewsError(415, 'send application/json');
  const declared = Number(request.headers.get('content-length'));
  if (Number.isFinite(declared) && declared > MAX_BODY_BYTES) throw new ReviewsError(413, 'request is too large');
  const text = await request.text();
  if (text.length > MAX_BODY_BYTES) throw new ReviewsError(413, 'request is too large');
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new ReviewsError(400, 'body is not JSON');
  }
}

/** A cross-site page must not be able to post reviews with the reader's session cookie. */
function assertSameOrigin(request) {
  const site = request.headers.get('sec-fetch-site');
  if (site && site !== 'same-origin') throw new ReviewsError(403, 'same-origin requests only');
  const origin = request.headers.get('origin');
  if (origin && origin !== new URL(request.url).origin) throw new ReviewsError(403, 'same-origin requests only');
}

/** POST /api/reviews */
async function post(request, env, now) {
  assertSameOrigin(request);
  const db = database(env);
  const body = await readBody(request);
  if (body === null || typeof body !== 'object' || Array.isArray(body)
    || Object.keys(body).length !== 1 || !Array.isArray(body.events)) {
    throw new ReviewsError(400, 'send {"events": [...]}');
  }
  if (body.events.length === 0) throw new ReviewsError(400, 'no events');
  if (body.events.length > MAX_BATCH) throw new ReviewsError(413, `at most ${MAX_BATCH} events per request`);

  const results = new Array(body.events.length);
  const toStore = new Map(); // event_id -> {event, index}
  body.events.forEach((raw, index) => {
    const checked = validateEvent(raw, now);
    if (!checked.event) {
      const id = raw && typeof raw.event_id === 'string' && ID.test(raw.event_id) ? raw.event_id : null;
      results[index] = { event_id: id, status: 'rejected', reason: checked.reason };
      return;
    }
    const seen = toStore.get(checked.event.event_id);
    if (seen) {
      results[index] = sameEvent(seen.event, checked.event)
        ? { event_id: checked.event.event_id, status: 'duplicate' }
        : { event_id: checked.event.event_id, status: 'rejected', reason: 'event_id reused with different content' };
      return;
    }
    toStore.set(checked.event.event_id, { event: checked.event, index });
  });

  if (toStore.size > 0) {
    const received = isoMs(now);
    const ids = [...toStore.keys()];
    // One transaction: D1 runs a batch atomically.
    const written = await db.batch(ids.map((id) => {
      const e = toStore.get(id).event;
      return db.prepare(INSERT).bind(e.event_id, e.card_id, e.rating, e.reviewed_at, e.device_id, e.response_time_ms, e.session_id, received);
    }));
    const stored = await db.prepare(
      `SELECT seq, event_id, card_id, rating, reviewed_at, device_id FROM review_events WHERE event_id IN (${placeholders(ids.length)})`,
    ).bind(...ids).all();
    const byId = new Map(stored.results.map((row) => [row.event_id, row]));
    ids.forEach((id, i) => {
      const { event, index } = toStore.get(id);
      const row = byId.get(id);
      if (!row) {
        results[index] = { event_id: id, status: 'rejected', reason: 'not stored' };
      } else if (written[i].meta.changes > 0) {
        results[index] = { event_id: id, status: 'accepted', seq: row.seq };
      } else if (sameEvent(row, event)) {
        results[index] = { event_id: id, status: 'duplicate', seq: row.seq };
      } else {
        results[index] = { event_id: id, status: 'rejected', reason: 'event_id reused with different content' };
      }
    });
  }

  const tally = (status) => results.filter((r) => r.status === status).length;
  return reply(200, {
    results,
    accepted: tally('accepted'),
    duplicate: tally('duplicate'),
    rejected: tally('rejected'),
    through_seq: await tip(db),
  });
}

async function tip(db) {
  const row = await db.prepare('SELECT COALESCE(MAX(seq), 0) AS seq FROM review_events').first();
  return row ? row.seq : 0;
}

function page(url, key) {
  const raw = url.searchParams.get(key);
  let after = 0;
  if (raw !== null) {
    if (!/^\d{1,15}$/.test(raw)) throw new ReviewsError(400, `${key} must be a whole number`);
    after = Number(raw);
  }
  const rawLimit = url.searchParams.get('limit');
  let limit = DEFAULT_PAGE;
  if (rawLimit !== null) {
    if (!/^\d{1,4}$/.test(rawLimit) || Number(rawLimit) < 1) throw new ReviewsError(400, 'limit must be a whole number from 1');
    limit = Math.min(Number(rawLimit), MAX_PAGE);
  }
  return { after, limit };
}

/** GET /api/reviews?since=N for devices. Only what a device needs to fold a card. */
async function list(request, env) {
  const db = database(env);
  const { after, limit } = page(new URL(request.url), 'since');
  const rows = await db.prepare(
    'SELECT seq, event_id, card_id, rating, reviewed_at, device_id FROM review_events WHERE seq > ? ORDER BY seq LIMIT ?',
  ).bind(after, limit).all();
  const through = await tip(db);
  const events = rows.results;
  const last = events.length ? events[events.length - 1].seq : after;
  return reply(200, { events, through_seq: through, more: last < through });
}

/** GET /ingest/reviews?after=N for the computer: every field, and when the Worker got it. */
export async function pull(request, env) {
  if (request.method !== 'GET') return reply(405, { error: 'method not allowed' });
  try {
    const db = database(env);
    const { after, limit } = page(new URL(request.url), 'after');
    const rows = await db.prepare(
      'SELECT seq, event_id, card_id, rating, reviewed_at, device_id, response_time_ms, session_id, received_at '
      + 'FROM review_events WHERE seq > ? ORDER BY seq LIMIT ?',
    ).bind(after, limit).all();
    const through = await tip(db);
    const events = rows.results;
    const last = events.length ? events[events.length - 1].seq : after;
    return reply(200, { events, through_seq: through, more: last < through });
  } catch (error) {
    return failed(error);
  }
}

function failed(error) {
  if (error instanceof ReviewsError) return reply(error.status, { error: error.message });
  console.error(`reviews failed: ${error && error.message}`);
  return reply(500, { error: 'reviews failed' });
}

/**
 * `/api/reviews` for a signed-in reader. `identity` is already verified; a
 * service token may read the site but cannot write reviews.
 */
export async function handleReviews(request, env, identity, now) {
  try {
    if (request.method === 'POST') {
      if (identity.kind !== 'user') throw new ReviewsError(403, 'forbidden');
      return await post(request, env, now);
    }
    if (request.method === 'GET' || request.method === 'HEAD') return await list(request, env);
    return reply(405, { error: 'method not allowed' });
  } catch (error) {
    return failed(error);
  }
}
