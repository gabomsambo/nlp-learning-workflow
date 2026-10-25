import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { MAX_AGE_MS, MAX_BATCH, MAX_FUTURE_MS, validateEvent } from '../src/reviews.js';
import { createHandler } from '../src/worker.js';
import {
  HAS_SQLITE, INGEST_AUD, KID, NOW_MS, PUBLISHER, fakeD1, fakeStore, makeIngestEnv, makeKeyPair, serviceClaims,
  signJwt, sqliteD1, userClaims,
} from './helpers.mjs';

const pair = await makeKeyPair();
const handle = createHandler({ keyFor: async (kid) => (kid === KID ? pair.jwk : null), now: () => NOW_MS });
const MIGRATION = readFileSync(new URL('../migrations/0001_review_events.sql', import.meta.url), 'utf8');

const CARD = '11111111-2222-4333-8444-555555555555';
const OTHER_CARD = '99999999-2222-4333-8444-555555555555';
const PHONE = 'phone-1234abcd';
const IPAD = 'ipad-5678efgh';
const at = (offsetMs = 0) => new Date(NOW_MS + offsetMs).toISOString();
let counter = 0;
const eventId = () => `00000000-0000-4000-8000-${String((counter += 1)).padStart(12, '0')}`;
const event = (over = {}) => ({
  event_id: eventId(), card_id: CARD, rating: 3, reviewed_at: at(-60000), device_id: PHONE, ...over,
});

const userToken = () => signJwt(pair.privateKey, userClaims());
const publisherToken = () => signJwt(pair.privateKey, serviceClaims({ aud: [INGEST_AUD], common_name: PUBLISHER }));
const env = (REVIEWS = fakeD1(), extra = {}) => makeIngestEnv({ REVIEWS, DOCS: fakeStore(), ...extra });

async function send(path, { method = 'POST', body, token, environment, headers = {} } = {}) {
  const init = { method, headers: { ...headers } };
  if (body !== undefined) {
    init.body = typeof body === 'string' ? body : JSON.stringify(body);
    if (!('content-type' in init.headers)) init.headers['content-type'] = 'application/json';
  }
  if (token !== undefined) init.headers['cf-access-jwt-assertion'] = token;
  return handle(new Request(`https://read.example.test${path}`, init), environment);
}

const post = async (events, environment, over = {}) => send('/api/reviews', {
  body: { events }, token: await userToken(), environment, ...over,
});

// ---------------------------------------------------------------------------
// The shape of an event
// ---------------------------------------------------------------------------

test('a well-formed event is accepted and holds only the stored fields', () => {
  const checked = validateEvent({ ...event(), response_time_ms: 4200, session_id: 'session-abcdef12' }, NOW_MS);
  assert.deepEqual(Object.keys(checked.event).sort(), [
    'card_id', 'device_id', 'event_id', 'rating', 'response_time_ms', 'reviewed_at', 'session_id',
  ]);
});

test('every field is held to a fixed shape', () => {
  const bad = {
    'a field that is not listed': { note: 'hello' },
    'an uppercase id': { event_id: 'AAAAAAAA-0000-4000-8000-00000000000B' },
    'an id that is not a UUID': { card_id: 'drop table' },
    'rating 0': { rating: 0 },
    'rating 5': { rating: 5 },
    'a rating sent as text': { rating: '3' },
    'a fractional rating': { rating: 2.5 },
    'a timestamp without milliseconds': { reviewed_at: '2026-10-25T11:59:00Z' },
    'a timestamp with an offset': { reviewed_at: '2026-10-25T11:59:00.000+02:00' },
    'a date that does not exist': { reviewed_at: '2026-02-30T11:59:00.000Z' },
    'a review from the future': { reviewed_at: at(MAX_FUTURE_MS + 1000) },
    'a review older than 30 days': { reviewed_at: at(-MAX_AGE_MS - 1000) },
    'a device id with spaces': { device_id: 'my phone 123' },
    'a device id that is too short': { device_id: 'abc' },
    'a negative response time': { response_time_ms: -1 },
    'a response time over an hour': { response_time_ms: 3600001 },
    'a response time with decimals': { response_time_ms: 1.5 },
    'a session id with markup': { session_id: '<script>alert(1)</script>' },
  };
  for (const [name, over] of Object.entries(bad)) {
    const checked = validateEvent({ ...event(), ...over }, NOW_MS);
    assert.ok(checked.reason, `${name} should be rejected`);
    assert.equal(checked.event, undefined, name);
  }
  for (const raw of [null, 'x', 7, [], undefined]) assert.ok(validateEvent(raw, NOW_MS).reason);
});

test('the edges of the time window are inside it', () => {
  assert.ok(validateEvent(event({ reviewed_at: at(MAX_FUTURE_MS) }), NOW_MS).event);
  assert.ok(validateEvent(event({ reviewed_at: at(-MAX_AGE_MS) }), NOW_MS).event);
});

// ---------------------------------------------------------------------------
// POST /api/reviews
// ---------------------------------------------------------------------------

test('an accepted event is stored with a seq, and sending it again stores nothing', async () => {
  const db = fakeD1();
  const e = event();
  const first = await (await post([e], env(db))).json();
  assert.equal(first.accepted, 1);
  assert.deepEqual(first.results[0], { event_id: e.event_id, status: 'accepted', seq: 1 });
  assert.equal(first.through_seq, 1);

  const again = await (await post([e], env(db))).json();
  assert.equal(again.accepted, 0);
  assert.equal(again.duplicate, 1);
  assert.deepEqual(again.results[0], { event_id: e.event_id, status: 'duplicate', seq: 1 });
  assert.equal(db.rows.length, 1);
});

test('the same event twice in one request is stored once', async () => {
  const db = fakeD1();
  const e = event();
  const body = await (await post([e, { ...e }], env(db))).json();
  assert.deepEqual(body.results.map((r) => r.status), ['accepted', 'duplicate']);
  assert.equal(db.rows.length, 1);
});

test('an event id sent again with different content is rejected, and the first one stands', async () => {
  const db = fakeD1();
  const e = event({ rating: 3 });
  await post([e], env(db));
  const body = await (await post([{ ...e, rating: 1 }], env(db))).json();
  assert.equal(body.results[0].status, 'rejected');
  assert.match(body.results[0].reason, /different content/);
  assert.equal(db.rows[0].rating, 3);
  const within = await (await post([event({ event_id: 'aaaaaaaa-0000-4000-8000-000000000001' }), { ...event({ event_id: 'aaaaaaaa-0000-4000-8000-000000000001' }), rating: 2 }], env(db))).json();
  assert.ok(within.results.some((r) => r.status === 'rejected'));
});

test('one bad event does not hold up the others, and each is told which', async () => {
  const db = fakeD1();
  const good = event();
  const body = await (await post([good, event({ rating: 9 }), { nonsense: true }], env(db))).json();
  assert.deepEqual(body.results.map((r) => r.status), ['accepted', 'rejected', 'rejected']);
  assert.equal(body.rejected, 2);
  assert.equal(db.rows.length, 1);
  assert.equal(body.results[2].event_id, null);
});

test('events keep arriving in seq order across requests and devices', async () => {
  const db = fakeD1();
  await post([event({ device_id: PHONE }), event({ device_id: PHONE, card_id: OTHER_CARD })], env(db));
  await post([event({ device_id: IPAD })], env(db));
  assert.deepEqual(db.rows.map((r) => [r.seq, r.device_id]), [[1, PHONE], [2, PHONE], [3, IPAD]]);
});

test('the request itself is held to a shape: type, size, count, keys', async () => {
  const environment = env();
  const token = await userToken();
  assert.equal((await send('/api/reviews', { body: '{}', token, environment, headers: { 'content-type': 'text/plain' } })).status, 415);
  assert.equal((await send('/api/reviews', { body: '{nope', token, environment })).status, 400);
  assert.equal((await send('/api/reviews', { body: { events: [] }, token, environment })).status, 400);
  assert.equal((await send('/api/reviews', { body: [event()], token, environment })).status, 400);
  assert.equal((await send('/api/reviews', { body: { events: [event()], extra: 1 }, token, environment })).status, 400);
  assert.equal((await send('/api/reviews', { body: { events: 'x' }, token, environment })).status, 400);
  const many = Array.from({ length: MAX_BATCH + 1 }, () => event());
  assert.equal((await send('/api/reviews', { body: { events: many }, token, environment })).status, 413);
  const huge = { events: [{ ...event(), session_id: 'x'.repeat(64) }], pad: 'y'.repeat(40000) };
  assert.equal((await send('/api/reviews', { body: huge, token, environment })).status, 413);
  assert.equal(environment.REVIEWS.rows.length, 0);
});

test('a page on another site cannot post reviews with the reader session', async () => {
  const db = fakeD1();
  const token = await userToken();
  const crossSite = await send('/api/reviews', {
    body: { events: [event()] }, token, environment: env(db), headers: { 'sec-fetch-site': 'cross-site' },
  });
  assert.equal(crossSite.status, 403);
  const foreignOrigin = await send('/api/reviews', {
    body: { events: [event()] }, token, environment: env(db), headers: { origin: 'https://evil.example' },
  });
  assert.equal(foreignOrigin.status, 403);
  const sameOrigin = await send('/api/reviews', {
    body: { events: [event()] }, token, environment: env(db),
    headers: { origin: 'https://read.example.test', 'sec-fetch-site': 'same-origin' },
  });
  assert.equal(sameOrigin.status, 200);
  assert.equal(db.rows.length, 1);
});

test('nobody without a reader session can post, and a service token cannot', async () => {
  const db = fakeD1();
  assert.equal((await send('/api/reviews', { body: { events: [event()] }, environment: env(db) })).status, 401);
  const stranger = await signJwt(pair.privateKey, userClaims({ email: 'stranger@example.com' }));
  assert.equal((await send('/api/reviews', { body: { events: [event()] }, token: stranger, environment: env(db) })).status, 403);
  // A service token that may read the site (the deployment check) still cannot write.
  const service = await signJwt(pair.privateKey, serviceClaims());
  const readable = env(db, { ALLOWED_SERVICE_TOKENS: 'abc123.access' });
  assert.equal((await send('/api/reviews', { body: { events: [event()] }, token: service, environment: readable })).status, 403);
  assert.equal(db.rows.length, 0);
});

test('without the REVIEWS binding the answer is 503 and nothing else breaks', async () => {
  const environment = makeIngestEnv({ DOCS: fakeStore() });
  assert.equal((await post([event()], environment)).status, 503);
  assert.equal((await send('/api/manifest', { method: 'GET', token: await userToken(), environment })).status, 404);
});

test('other write routes stay closed', async () => {
  const token = await userToken();
  for (const path of ['/api/manifest', '/api/doc/library.json', '/', '/api/reviews/x']) {
    const response = await send(path, { body: {}, token, environment: env() });
    assert.equal(response.status, 405, path);
  }
  assert.equal((await send('/api/reviews', { method: 'DELETE', token, environment: env() })).status, 405);
  assert.equal((await send('/api/reviews', { method: 'PUT', body: {}, token, environment: env() })).status, 405);
});

test('a database failure answers 500 without its message', async () => {
  const broken = { prepare() { throw new Error('D1_ERROR: secret detail'); }, batch: async () => { throw new Error('boom'); } };
  const response = await post([event()], env(broken));
  assert.equal(response.status, 500);
  assert.equal(JSON.stringify(await response.json()).includes('secret'), false);
});

// ---------------------------------------------------------------------------
// GET /api/reviews
// ---------------------------------------------------------------------------

async function seeded(n) {
  const db = fakeD1();
  await post(Array.from({ length: n }, (_, i) => event({ reviewed_at: at(-1000 * (n - i)), device_id: i % 2 ? IPAD : PHONE })), env(db));
  return db;
}

test('a device reads what came after its snapshot, in pages', async () => {
  const db = await seeded(5);
  const token = await userToken();
  const page = await (await send('/api/reviews?since=1&limit=2', { method: 'GET', token, environment: env(db) })).json();
  assert.deepEqual(page.events.map((e) => e.seq), [2, 3]);
  assert.equal(page.through_seq, 5);
  assert.equal(page.more, true);
  const rest = await (await send('/api/reviews?since=3', { method: 'GET', token, environment: env(db) })).json();
  assert.deepEqual(rest.events.map((e) => e.seq), [4, 5]);
  assert.equal(rest.more, false);
  assert.deepEqual(Object.keys(rest.events[0]).sort(), ['card_id', 'device_id', 'event_id', 'rating', 'reviewed_at', 'seq']);
  const none = await (await send('/api/reviews?since=5', { method: 'GET', token, environment: env(db) })).json();
  assert.deepEqual(none, { events: [], through_seq: 5, more: false });
});

test('the paging parameters are checked', async () => {
  const token = await userToken();
  for (const query of ['since=-1', 'since=abc', 'since=1.5', 'limit=0', 'limit=x', 'since=99999999999999999']) {
    const response = await send(`/api/reviews?${query}`, { method: 'GET', token, environment: env() });
    assert.equal(response.status, 400, query);
  }
});

test('reading reviews needs a reader session', async () => {
  assert.equal((await send('/api/reviews', { method: 'GET', environment: env() })).status, 401);
});

// ---------------------------------------------------------------------------
// GET /ingest/reviews (the computer)
// ---------------------------------------------------------------------------

test('the computer pulls every field, after a cursor, with its service token', async () => {
  const db = await seeded(3);
  const environment = env(db);
  const body = await (await send('/ingest/reviews?after=1', { method: 'GET', token: await publisherToken(), environment })).json();
  assert.deepEqual(body.events.map((e) => e.seq), [2, 3]);
  assert.deepEqual(Object.keys(body.events[0]).sort(), [
    'card_id', 'device_id', 'event_id', 'rating', 'received_at', 'response_time_ms', 'reviewed_at', 'seq', 'session_id',
  ]);
  assert.equal(body.events[0].received_at, at());
  assert.equal(body.through_seq, 3);
  assert.equal(body.more, false);
});

test('no person can pull from /ingest/reviews, whatever they carry', async () => {
  const db = await seeded(1);
  const environment = env(db);
  assert.equal((await send('/ingest/reviews', { method: 'GET', environment })).status, 401);
  assert.equal((await send('/ingest/reviews', { method: 'GET', token: await userToken(), environment })).status, 401);
  const readerAudService = await signJwt(pair.privateKey, serviceClaims());
  assert.equal((await send('/ingest/reviews', { method: 'GET', token: readerAudService, environment })).status, 401);
  const other = await signJwt(pair.privateKey, serviceClaims({ aud: [INGEST_AUD], common_name: 'someone-else.access' }));
  assert.equal((await send('/ingest/reviews', { method: 'GET', token: other, environment })).status, 403);
  assert.equal((await send('/ingest/reviews', { method: 'POST', body: {}, token: await publisherToken(), environment })).status, 405);
});

// ---------------------------------------------------------------------------
// The same statements on real SQLite, with the migration file as written
// ---------------------------------------------------------------------------

test('the migration and the statements work on real SQLite', { skip: !HAS_SQLITE && 'node:sqlite is not available' }, async () => {
  const db = sqliteD1(MIGRATION);
  const e = event();
  const first = await (await post([e, event({ rating: 1 })], env(db))).json();
  assert.deepEqual(first.results.map((r) => r.status), ['accepted', 'accepted']);
  assert.equal(first.through_seq, 2);

  const retry = await (await post([e], env(db))).json();
  assert.deepEqual(retry.results[0], { event_id: e.event_id, status: 'duplicate', seq: 1 });
  const clash = await (await post([{ ...e, rating: 4 }], env(db))).json();
  assert.equal(clash.results[0].status, 'rejected');

  const pulled = await (await send('/ingest/reviews?after=0', { method: 'GET', token: await publisherToken(), environment: env(db) })).json();
  assert.deepEqual(pulled.events.map((x) => x.seq), [1, 2]);
  assert.equal(pulled.events[0].received_at, at());
  assert.equal(pulled.events[0].response_time_ms, null);

  // seq only grows: a number is never handed out twice, even after the newest row is deleted by
  // hand. (An ignored duplicate can burn a number, so there can be gaps; a cursor does not mind.)
  db.db.exec('DELETE FROM review_events WHERE seq = 2');
  const next = await (await post([event()], env(db))).json();
  assert.ok(next.results[0].seq > 2);
  // The CHECK constraints are a second wall behind the Worker's own checks.
  assert.throws(() => db.db.exec(
    `INSERT INTO review_events (event_id, card_id, rating, reviewed_at, device_id, received_at) VALUES ('${eventId()}', '${CARD}', 7, 'x', 'abcdefgh', 'y')`,
  ));
});
