// Reviewing on the phone and getting the answers home: the outbox, the sync calls, the review
// manager, and the real Worker with a D1 stand-in in the middle. Two devices, a lost response,
// a page that dies half-way, a snapshot that overtakes the events.
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { OutboxFullError, MAX_EVENTS, createOutbox, localStorageStorage, memoryStorage } from '../public/js/outbox.js';
import { BATCH, createSync } from '../public/js/sync.js';
import { createReviews, statusLine } from '../public/js/reviews.js';
import {
  Rating, STATE_NAME, cardFromSnapshot, foldEvents, newCard, describe,
} from '../public/js/fsrs.js';
import { NEW_PER_SESSION, buildQueue, dueCards, effectiveCards, nextDueAt, summarize } from '../public/js/schedule.js';
import { createHandler } from '../src/worker.js';
import {
  INGEST_AUD, KID, NOW_MS, PUBLISHER, fakeD1, fakeStore, makeIngestEnv, makeKeyPair, serviceClaims, signJwt, userClaims,
} from './helpers.mjs';

const MIN = 60 * 1000;
const HOUR = 60 * MIN;
const DAY = 24 * HOUR;
const CARD = '11111111-2222-4333-8444-555555555555';
const CARD_B = '99999999-2222-4333-8444-555555555555';

// --- a world: the real Worker, a D1 stand-in, and devices that talk to it -----------------

const pair = await makeKeyPair();
const handle = createHandler({ keyFor: async (kid) => (kid === KID ? pair.jwk : null), now: () => NOW_MS });

function world() {
  const env = makeIngestEnv({ REVIEWS: fakeD1(), DOCS: fakeStore() });
  const network = { online: true, loseNextResponse: 0, calls: [], auth: true };
  async function fetchFn(path, init = {}) {
    network.calls.push(`${init.method || 'GET'} ${path.split('?')[0]}`);
    if (!network.online) throw new TypeError('Failed to fetch');
    if (!network.auth) return { type: 'opaqueredirect', status: 0, ok: false };
    const token = await signJwt(pair.privateKey, userClaims());
    const headers = { ...(init.headers || {}), 'cf-access-jwt-assertion': token, 'sec-fetch-site': 'same-origin' };
    const response = await handle(new Request(`https://read.example.test${path}`, { ...init, headers }), env);
    if (network.loseNextResponse > 0) {
      network.loseNextResponse -= 1;
      throw new TypeError('The network connection was lost.'); // the Worker stored it; the phone never heard
    }
    return response;
  }
  return { env, network, fetchFn, rows: () => env.REVIEWS.rows };
}

function uuidSeq(prefix) {
  let n = 0;
  return () => `${prefix}-0000-4000-8000-${String((n += 1)).padStart(12, '0')}`;
}

function deviceWithIds(w, prefix, opts = {}) {
  const storage = opts.storage || memoryStorage();
  const clock = opts.clock || { t: NOW_MS - 2 * DAY };
  const ids = uuidSeq(prefix);
  // The first id is the device's; the rest are events'.
  const deviceId = `${prefix}-device-0001`;
  let first = true;
  const outbox = createOutbox(storage, { now: () => clock.t, newId: () => (first ? ((first = false), deviceId) : ids()) });
  const sync = createSync({ outbox, fetchFn: w.fetchFn });
  return { outbox, sync, storage, clock, deviceId };
}

const review = (d, card, rating, at) => {
  d.clock.t = at;
  return d.outbox.record({ card_id: card, rating });
};

/** What the computer would publish for a card after folding these events: a schedule.json entry. */
function snapshotEntry(cardId, events) {
  const card = foldEvents(newCard(), events);
  if (card.lastReview === null) return { card_id: cardId, paper_id: 'p', state: 'new', due: null, stability: 0, difficulty: 0, last_review: null, step: null };
  return {
    card_id: cardId, paper_id: 'p', state: STATE_NAME[card.state], due: new Date(card.due).toISOString(),
    stability: card.stability, difficulty: card.difficulty, step: card.step, last_review: new Date(card.lastReview).toISOString(),
  };
}

const T = (offsetMs) => NOW_MS - 3 * DAY + offsetMs;

// --- the outbox ----------------------------------------------------------------------------

test('a review is written down before anything is sent, and carries the device clock to the millisecond', () => {
  const clock = { t: Date.UTC(2026, 9, 25, 9, 30, 15, 123) };
  const outbox = createOutbox(memoryStorage(), { now: () => clock.t });
  const event = outbox.record({ card_id: CARD, rating: 3, response_time_ms: 4200, session_id: 'session-abcdef12' });
  assert.equal(event.reviewed_at, '2026-10-25T09:30:15.123Z');
  assert.match(event.event_id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  assert.equal(event.device_id, outbox.deviceId());
  assert.equal(outbox.waiting(), 1);
  assert.deepEqual(outbox.pending(), [event]);
});

test('reviews and the device id survive a reload; a half-written or foreign state is ignored', () => {
  const storage = memoryStorage();
  const first = createOutbox(storage);
  const event = first.record({ card_id: CARD, rating: 4 });
  const second = createOutbox(storage);
  assert.equal(second.deviceId(), first.deviceId());
  assert.deepEqual(second.pending(), [event]);

  for (const junk of ['text', 7, [], { v: 2 }, { v: 1, outbox: 'x' }, { v: 1, outbox: [{ event: { rating: 9 } }, null, 3] }]) {
    const outbox = createOutbox(memoryStorage(junk));
    assert.deepEqual(outbox.events(), [], JSON.stringify(junk));
    assert.ok(outbox.deviceId().length >= 8);
  }
});

test('storage that throws never loses a review in memory, and says it is not durable', () => {
  let writes = 0;
  const flaky = { load: () => null, save: () => { writes += 1; throw new Error('QuotaExceededError'); } };
  const outbox = createOutbox(flaky);
  outbox.record({ card_id: CARD, rating: 3 });
  assert.equal(outbox.waiting(), 1);
  assert.equal(outbox.durable(), false);
  assert.ok(writes >= 2);
  const unreadable = createOutbox({ load: () => { throw new Error('SecurityError'); }, save: () => {} });
  assert.equal(unreadable.waiting(), 0);
});

test('localStorage is one JSON value under one key', () => {
  const area = new Map();
  const shim = { getItem: (k) => (area.has(k) ? area.get(k) : null), setItem: (k, v) => area.set(k, v) };
  const storage = localStorageStorage('k', shim);
  assert.equal(storage.load(), null);
  createOutbox(storage).record({ card_id: CARD, rating: 2 });
  assert.equal(JSON.parse(area.get('k')).outbox.length, 1);
});

test('a runaway loop is refused rather than filling the phone', () => {
  const outbox = createOutbox(memoryStorage());
  for (let i = 0; i < MAX_EVENTS; i += 1) outbox.record({ card_id: CARD, rating: 3 });
  assert.throws(() => outbox.record({ card_id: CARD, rating: 3 }), OutboxFullError);
});

// --- sending -------------------------------------------------------------------------------

test('offline reviews wait, then go out once, and sending again sends nothing', async () => {
  const w = world();
  const d = deviceWithIds(w, 'aaaaaaaa');
  w.network.online = false;
  review(d, CARD, 3, T(0));
  review(d, CARD, 3, T(10 * MIN));
  assert.equal((await d.sync.flush()).status, 'offline');
  assert.equal(d.outbox.waiting(), 2);
  assert.equal(w.rows().length, 0);

  w.network.online = true;
  assert.deepEqual(await d.sync.flush(), { status: 'ok', sent: 2, rejected: 0 });
  assert.equal(d.outbox.waiting(), 0);
  assert.equal(w.rows().length, 2);
  const calls = w.network.calls.length;
  assert.deepEqual(await d.sync.flush(), { status: 'nothing', sent: 0, rejected: 0 });
  assert.equal(w.network.calls.length, calls, 'nothing to send, nothing sent');
});

test('a response that was lost is retried and the cloud stores the review once', async () => {
  const w = world();
  const d = deviceWithIds(w, 'bbbbbbbb');
  const event = review(d, CARD, 3, T(0));
  w.network.loseNextResponse = 1;
  assert.equal((await d.sync.flush()).status, 'offline');
  assert.equal(w.rows().length, 1, 'the Worker stored it');
  assert.equal(d.outbox.waiting(), 1, 'the phone does not know that');

  assert.equal((await d.sync.flush()).status, 'ok');
  assert.equal(w.rows().length, 1, 'still one');
  assert.equal(w.rows()[0].event_id, event.event_id);
  assert.equal(d.outbox.waiting(), 0);
});

test('a page that dies after the cloud stored the review but before the phone noted it', async () => {
  const w = world();
  const storage = memoryStorage();
  const d = deviceWithIds(w, 'cccccccc', { storage });
  review(d, CARD, 3, T(0));
  // The answer arrives but the page is gone before applyResults runs: model it as the save never landing.
  const before = JSON.parse(JSON.stringify(storage.load()));
  await d.sync.flush();
  storage.save(before);

  const reopened = createOutbox(storage);
  assert.equal(reopened.waiting(), 1);
  const sync = createSync({ outbox: reopened, fetchFn: w.fetchFn });
  assert.equal((await sync.flush()).status, 'ok');
  assert.equal(w.rows().length, 1);
  assert.equal(reopened.waiting(), 0);
});

test('a page that dies between rating and sending loses nothing', async () => {
  const w = world();
  const storage = memoryStorage();
  const d = deviceWithIds(w, 'dddddddd', { storage });
  review(d, CARD, 4, T(0));
  // ... the page is gone. A new one opens on the same storage.
  const reopened = createOutbox(storage);
  const sync = createSync({ outbox: reopened, fetchFn: w.fetchFn });
  await sync.flush();
  assert.equal(w.rows().length, 1);
  assert.equal(w.rows()[0].rating, 4);
});

test('more than a batch is sent in batches', async () => {
  const w = world();
  const d = deviceWithIds(w, 'eeeeeeee');
  for (let i = 0; i < BATCH + 7; i += 1) review(d, CARD, 3, T(i * MIN));
  const result = await d.sync.flush();
  assert.equal(result.sent, BATCH + 7);
  assert.equal(w.rows().length, BATCH + 7);
  assert.equal(w.network.calls.filter((c) => c.startsWith('POST')).length, 2);
});

test('two flushes at once are one', async () => {
  const w = world();
  const d = deviceWithIds(w, 'ffffffff');
  review(d, CARD, 3, T(0));
  const [a, b] = await Promise.all([d.sync.flush(), d.sync.flush()]);
  assert.equal(a, b);
  assert.equal(w.network.calls.filter((c) => c.startsWith('POST')).length, 1);
});

test('an ended session keeps every review and says to sign in again', async () => {
  const w = world();
  const d = deviceWithIds(w, 'abababab');
  review(d, CARD, 3, T(0));
  w.network.auth = false;
  assert.equal((await d.sync.flush()).status, 'auth');
  assert.equal(d.outbox.waiting(), 1);
  assert.match(statusLine({ waiting: 1, rejected: 0, durable: true, last: 'auth' }), /Sign in again/);
  w.network.auth = true;
  assert.equal((await d.sync.flush()).status, 'ok');
});

test('a review the cloud will never take is dropped and counted, the rest go through', async () => {
  const w = world();
  const d = deviceWithIds(w, 'cdcdcdcd');
  review(d, CARD, 3, NOW_MS - 40 * DAY); // older than the cloud accepts
  review(d, CARD, 3, T(0));
  const result = await d.sync.flush();
  assert.equal(result.rejected, 1);
  assert.equal(d.outbox.rejected(), 1);
  assert.equal(w.rows().length, 1);
  assert.match(statusLine({ waiting: 0, rejected: 1, durable: true, last: 'ok' }), /1 review was not accepted/);
  d.outbox.clearRejected();
  assert.equal(statusLine({ waiting: 0, rejected: 0, durable: true, last: 'ok' }), '');
});

test('an answer that settles nothing does not loop forever', async () => {
  const outbox = createOutbox(memoryStorage());
  outbox.record({ card_id: CARD, rating: 3 });
  let calls = 0;
  const sync = createSync({
    outbox,
    fetchFn: async () => { calls += 1; return new Response(JSON.stringify({ results: [] }), { status: 200 }); },
  });
  assert.equal((await sync.flush()).status, 'error');
  assert.equal(calls, 1);
  assert.equal(outbox.waiting(), 1);
});

test('server trouble is "offline": try later, lose nothing', async () => {
  const outbox = createOutbox(memoryStorage());
  outbox.record({ card_id: CARD, rating: 3 });
  for (const status of [500, 502, 503, 429]) {
    const sync = createSync({ outbox, fetchFn: async () => new Response('{}', { status }) });
    assert.equal((await sync.flush()).status, 'offline', String(status));
  }
  const refused = createSync({ outbox, fetchFn: async () => new Response('{}', { status: 400 }) });
  assert.equal((await refused.flush()).status, 'error');
  assert.equal(outbox.waiting(), 1);
});

// --- two devices ---------------------------------------------------------------------------

test('two devices that review the same card offline both count, and agree afterwards', async () => {
  const w = world();
  const phone = deviceWithIds(w, 'a1a1a1a1');
  const ipad = deviceWithIds(w, 'b2b2b2b2');
  review(phone, CARD, 3, T(0));
  review(ipad, CARD, 2, T(5 * MIN));
  review(phone, CARD, 3, T(10 * MIN));
  review(ipad, CARD, 3, T(1 * DAY));
  await ipad.sync.flush();   // the iPad's go first
  await phone.sync.flush();

  assert.equal(w.rows().length, 4);
  await phone.sync.pullPeers(0);
  await ipad.sync.pullPeers(0);

  const entry = { card_id: CARD, paper_id: 'p', state: 'new', due: null, stability: 0, difficulty: 0, last_review: null, step: null };
  const seen = (d) => effectiveCards([entry], [...d.outbox.peerEvents(), ...d.outbox.events()])[0].card;
  assert.deepEqual(describe(seen(phone)), describe(seen(ipad)), 'every device lands on the same card');

  // And it is the card the computer derives from the cloud's log, whatever order it arrived in.
  const pulled = await (await handle(
    new Request('https://read.example.test/ingest/reviews?after=0', {
      headers: { 'cf-access-jwt-assertion': await signJwt(pair.privateKey, serviceClaims({ aud: [INGEST_AUD], common_name: PUBLISHER })) },
    }),
    w.env,
  )).json();
  assert.equal(pulled.events.length, 4);
  assert.deepEqual(describe(foldEvents(newCard(), pulled.events)), describe(seen(phone)));
  assert.equal(describe(seen(phone)).state, 2, 'four reviews on, the card is in review');
});

test('a review heard about from another device counts the moment it is pulled', async () => {
  const w = world();
  const phone = deviceWithIds(w, 'c3c3c3c3');
  const ipad = deviceWithIds(w, 'd4d4d4d4');
  const entry = { card_id: CARD, paper_id: 'p', state: 'new', due: null, stability: 0, difficulty: 0, last_review: null, step: null };
  review(ipad, CARD, 3, T(0));
  await ipad.sync.flush();
  assert.equal(effectiveCards([entry], [...phone.outbox.peerEvents(), ...phone.outbox.events()])[0].fresh, true);
  assert.deepEqual(await phone.sync.pullPeers(0), { status: 'ok', count: 1 });
  assert.equal(effectiveCards([entry], [...phone.outbox.peerEvents(), ...phone.outbox.events()])[0].fresh, false);
});

test('pulling peers follows pages and survives being offline', async () => {
  const w = world();
  const ipad = deviceWithIds(w, 'e5e5e5e5');
  for (let i = 0; i < 12; i += 1) review(ipad, CARD, 3, T(i * HOUR));
  await ipad.sync.flush();
  const phone = deviceWithIds(w, 'f6f6f6f6');
  assert.equal((await phone.sync.pullPeers(0)).count, 12);
  w.network.online = false;
  assert.equal((await phone.sync.pullPeers(0)).status, 'offline');
  assert.equal(phone.outbox.peerEvents().length, 12, 'what was heard before stays');
});

// --- the snapshot overtakes the events ------------------------------------------------------

test('once the computer publishes a snapshot that includes an event, it is not counted twice', async () => {
  const w = world();
  const phone = deviceWithIds(w, 'a7a7a7a7');
  const e1 = review(phone, CARD, 3, T(0));
  const e2 = review(phone, CARD, 3, T(10 * MIN));
  await phone.sync.flush();
  const seqs = w.rows().map((r) => r.seq);

  const oldSnapshot = { card_id: CARD, paper_id: 'p', state: 'new', due: null, stability: 0, difficulty: 0, last_review: null, step: null };
  const before = effectiveCards([oldSnapshot], phone.outbox.events())[0].card;

  // The computer pulls both events and publishes: through_seq covers them.
  const published = snapshotEntry(CARD, [e1, e2]);
  phone.outbox.reconcile(Math.max(...seqs));
  assert.equal(phone.outbox.events().length, 0, 'folded into the snapshot, dropped from the outbox');
  const after = effectiveCards([published], phone.outbox.events())[0].card;
  assert.deepEqual(describe(after), describe(before));

  // A new review on top of the new snapshot, still waiting to sync, folds forward from it.
  const e3 = review(phone, CARD, 3, T(1 * DAY));
  assert.deepEqual(
    describe(effectiveCards([published], phone.outbox.events())[0].card),
    describe(foldEvents(newCard(), [e1, e2, e3])),
  );
});

test('reviews not yet acknowledged survive a snapshot that does not include them', async () => {
  const w = world();
  const d = deviceWithIds(w, 'b8b8b8b8');
  w.network.online = false;
  review(d, CARD, 3, T(0));
  d.outbox.reconcile(9999);
  assert.equal(d.outbox.waiting(), 1, 'an unsent review is never dropped, whatever the snapshot says');
});

// --- the manager and the queue -------------------------------------------------------------

function scheduleStore(cards, through = 0) {
  return {
    refreshed: 0,
    refresh: async function refresh() { this.refreshed += 1; },
    schedule: async () => ({ through_seq: through, cards }),
  };
}

test('states are the snapshot with this device and other devices folded on top', async () => {
  const w = world();
  const d = deviceWithIds(w, 'c9c9c9c9');
  const cards = [
    { card_id: CARD, paper_id: 'p', state: 'new', due: null, stability: 0, difficulty: 0, last_review: null, step: null },
    { card_id: CARD_B, paper_id: 'p', state: 'new', due: null, stability: 0, difficulty: 0, last_review: null, step: null },
  ];
  const reviews = createReviews({ store: scheduleStore(cards), outbox: d.outbox, sync: d.sync, now: () => NOW_MS });
  review(d, CARD, 3, T(0));
  const states = await reviews.states();
  assert.deepEqual(states.map((s) => [s.card_id, s.fresh]), [[CARD, false], [CARD_B, true]]);
});

test('rating records first and sends behind it; a failed send leaves the review waiting', async () => {
  const w = world();
  const d = deviceWithIds(w, 'dadadada');
  const reviews = createReviews({ store: scheduleStore([]), outbox: d.outbox, sync: d.sync });
  w.network.online = false;
  let heard = 0;
  reviews.onChange(() => { heard += 1; });
  const event = reviews.rate(CARD, 3, { responseMs: 1500, sessionId: 'session-abcdef12' });
  assert.equal(d.outbox.waiting(), 1, 'written down before any network answer');
  assert.equal(event.session_id, 'session-abcdef12');
  await reviews.flush();
  assert.equal(reviews.status().last, 'offline');
  assert.equal(reviews.status().waiting, 1);
  assert.ok(heard >= 1);
  w.network.online = true;
  await reviews.sync();
  assert.equal(reviews.status().waiting, 0);
  assert.equal(w.rows().length, 1);
  assert.equal(w.rows()[0].response_time_ms, 1500);
});

test('sync asks for a fresh snapshot, sends, and listens from its through_seq', async () => {
  const w = world();
  const d = deviceWithIds(w, 'ebebebeb');
  const store = scheduleStore([], 17);
  const reviews = createReviews({ store, outbox: d.outbox, sync: d.sync });
  review(d, CARD, 3, T(0));
  await reviews.sync();
  assert.equal(store.refreshed, 1);
  assert.ok(w.network.calls.includes('POST /api/reviews'));
  assert.ok(w.network.calls.includes('GET /api/reviews'));
});

test('sync with nothing published yet does not throw', async () => {
  const w = world();
  const d = deviceWithIds(w, 'fcfcfcfc');
  const store = { refresh: async () => { throw new Error('empty'); }, schedule: async () => { throw new Error('empty'); } };
  const reviews = createReviews({ store, outbox: d.outbox, sync: d.sync });
  await assert.doesNotReject(reviews.sync());
});

test('the queue is what is due, longest overdue first, then a limited number of new cards', () => {
  const now = NOW_MS;
  const reviewed = (id, dueOffset) => ({
    card_id: id, paper_id: 'p', fresh: false,
    card: { ...newCard(), state: 2, step: null, stability: 5, difficulty: 5, due: now + dueOffset, lastReview: now - DAY },
  });
  const fresh = (id) => ({ card_id: id, paper_id: 'p', fresh: true, card: newCard() });
  const states = [reviewed('late', -3 * DAY), reviewed('soon', -1 * HOUR), reviewed('later', 2 * DAY),
    ...Array.from({ length: 14 }, (_, i) => fresh(`n${String(i).padStart(2, '0')}`))];
  const queue = buildQueue(states, now);
  assert.deepEqual(queue.slice(0, 2).map((s) => s.card_id), ['late', 'soon']);
  assert.equal(queue.length, 2 + NEW_PER_SESSION);
  assert.ok(!queue.some((s) => s.card_id === 'later'));
  // Reviewing new cards does not bring ever more of them in.
  const introduced = new Set(queue.slice(2).map((s) => s.card_id));
  assert.equal(buildQueue(states.filter((s) => !introduced.has(s.card_id)), now, { introduced }).filter((s) => s.fresh).length, 0);
  assert.deepEqual(summarize(states, now), { total: 17, due: 2, fresh: 14, upcoming: 1 });
  assert.equal(nextDueAt(states, now), now + 2 * DAY);
  assert.equal(dueCards(states, now).length, 2);
});

test('a card rated Again comes back in minutes, Good after the first step, and the queue shows it only then', () => {
  const entry = { card_id: CARD, paper_id: 'p', state: 'new', due: null, stability: 0, difficulty: 0, last_review: null, step: null };
  const event = { event_id: 'e1', card_id: CARD, rating: Rating.Again, reviewed_at: new Date(NOW_MS).toISOString() };
  const [state] = effectiveCards([entry], [event]);
  assert.equal(state.card.due - NOW_MS, 1 * MIN);
  assert.equal(buildQueue([state], NOW_MS).length, 0, 'not due yet');
  assert.equal(buildQueue([state], NOW_MS + 2 * MIN).length, 1);
});

test('a snapshot entry the computer could only read from the card row still folds', () => {
  const row = { card_id: CARD, paper_id: 'p', state: 'review', due: new Date(NOW_MS - DAY).toISOString(), stability: 6, difficulty: 5, last_review: new Date(NOW_MS - 7 * DAY).toISOString(), step: null, reps: null };
  assert.equal(cardFromSnapshot(row).due, NOW_MS - DAY);
  const [state] = effectiveCards([row], [{ event_id: 'x', card_id: CARD, rating: 3, reviewed_at: new Date(NOW_MS).toISOString() }]);
  assert.ok(state.card.due > NOW_MS + DAY);
});

test('status lines say only what matters', () => {
  assert.equal(statusLine({ waiting: 0, rejected: 0, durable: true, last: 'ok' }), '');
  assert.match(statusLine({ waiting: 1, rejected: 0, durable: true, last: 'offline' }), /1 review waiting to sync.*back online/);
  assert.match(statusLine({ waiting: 3, rejected: 0, durable: true, last: 'nothing' }), /3 reviews waiting to sync\./);
  assert.match(statusLine({ waiting: 0, rejected: 0, durable: false, last: 'ok' }), /not keeping reviews/);
});
