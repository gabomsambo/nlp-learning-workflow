// What the phone keeps and why: the newest three episodes, eviction, downloads that must hash to
// their name, the document copy behind the store, and what the page says about the connection and the outbox.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';

import { ReaderError, createStore } from '../public/js/data.js';
import { memoryKv } from '../public/js/kv.js';
import { createOffline, registerWorker } from '../public/js/offline.js';
import {
  connectionNotice, downloadProblem, episodeDocKeys, formatBytes, headerStatus, installHint, outboxPanel, planEpisodes,
  playableEpisodes, savedLabel,
} from '../public/js/offline-core.js';

const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const enc = (text) => new TextEncoder().encode(text);

// ---- a published bundle and a cloud that serves it ---------------------------------------

function bundle(episodeSpecs) {
  const audio = new Map();
  const docs = {};
  const episodes = episodeSpecs.map(({ id, day, noAudio, body = `audio of ${id}` }) => {
    const bytes = enc(body);
    const hash = sha(bytes);
    if (!noAudio) audio.set(hash, bytes);
    docs[`episodes/${id}.json`] = { id, paper_id: 'p1', follow_along: id === 'e-new' ? [{ doc: 'playgrounds/pg1.json' }] : [] };
    return {
      id, paper_id: 'p1', doc: `episodes/${id}.json`, title: `Episode ${id}`, duration_seconds: 60,
      created_at: `2026-10-${String(day).padStart(2, '0')}T12:00:00+00:00`,
      audio_key: noAudio ? null : `audio/${hash}.mp3`,
    };
  });
  docs['library.json'] = { pillars: [], papers: [{ id: 'p1', doc: 'papers/p1.json' }], episodes };
  docs['schedule.json'] = { through_seq: 0, cards: [] };
  docs['papers/p1.json'] = { id: 'p1', playground: { doc: 'playgrounds/pg1.json' } };
  docs['playgrounds/pg1.json'] = { id: 'pg1' };
  const bodies = Object.fromEntries(Object.entries(docs).map(([key, value]) => [key, enc(JSON.stringify(value))]));
  const manifest = {
    generated_at: '2026-10-25T00:00:00Z',
    docs: Object.entries(bodies).map(([key, bytes]) => ({ key, sha256: sha(bytes), bytes: bytes.length })),
  };
  return { manifest, bodies, audio, episodes };
}

/** A cloud behind `fetch`. Flip `state.online`/`state.session`; every request is recorded. */
function cloud(initial) {
  const state = { online: true, session: true, bundle: initial, requests: [], tamper: null };
  async function fetchFn(path, init = {}) {
    state.requests.push(path);
    if (!state.online) throw new TypeError('Failed to fetch');
    if (!state.session) return { type: 'opaqueredirect', status: 0, ok: false };
    if (path === '/api/manifest') return Response.json(state.bundle.manifest);
    if (path.startsWith('/api/doc/')) {
      const bytes = state.bundle.bodies[decodeURIComponent(path.slice('/api/doc/'.length))];
      return bytes ? new Response(bytes) : new Response('', { status: 404 });
    }
    if (path.startsWith('/api/audio/')) {
      const hash = path.slice('/api/audio/'.length, -'.mp3'.length);
      const bytes = state.tamper ? state.tamper(hash) : state.bundle.audio.get(hash);
      return bytes ? new Response(bytes, { headers: { 'content-length': String(bytes.length) } }) : new Response('', { status: 404 });
    }
    return new Response('', { status: 404 });
  }
  return { state, fetchFn };
}

/** The part of the Cache API the manager uses, over a Map. */
function fakeCacheStorage({ failPut = null } = {}) {
  const stores = new Map();
  const open = async (name) => {
    if (!stores.has(name)) stores.set(name, new Map());
    const map = stores.get(name);
    return {
      async put(path, response) {
        if (failPut) throw failPut;
        map.set(path, new Uint8Array(await response.arrayBuffer()));
      },
      async keys() { return [...map.keys()].map((path) => new Request(`https://reader.test${path}`)); },
      async delete(path) { return map.delete(path); },
      async match(path) { return map.has(path) ? new Response(map.get(path)) : undefined; },
    };
  };
  return { stores, open };
}

async function rig(initial, { failPut = null, kv = memoryKv(), cacheStorage = fakeCacheStorage({ failPut }), online = true } = {}) {
  const world = cloud(initial);
  const previous = globalThis.fetch;
  globalThis.fetch = world.fetchFn;
  const state = { online };
  const store = createStore({ kv });
  const persisted = { asked: 0 };
  const offline = createOffline({
    store, kv, cacheStorage, fetchFn: world.fetchFn, isOnline: () => state.online,
    storageManager: { persisted: async () => false, persist: async () => { persisted.asked += 1; return true; }, estimate: async () => ({ usage: 5, quota: 50 }) },
  });
  const audioIn = () => [...(cacheStorage.stores.get('nlp-audio-v1') || new Map()).keys()];
  return { world, store, offline, kv, cacheStorage, persisted, state, audioIn, restore: () => { globalThis.fetch = previous; } };
}

const specs = (...days) => days.map((day, i) => ({ id: `e${day}`, day, body: `audio ${day} ${i}` }));

// ---- which episodes the phone keeps -----------------------------------------------------

const ep = (id, audio) => ({ id, audio_key: `audio/${audio.repeat(64)}.mp3`, created_at: id });
const rows = [ep('e5', 'a'), ep('e4', 'b'), ep('e3', 'c'), ep('e2', 'd'), ep('e1', 'e')]; // newest first

test('the newest three are kept: the rest are downloaded or deleted to match', () => {
  assert.deepEqual(planEpisodes({ episodes: rows, saved: {}, keep: 3 }), { download: ['e5', 'e4', 'e3'], remove: [] });
  const saved = { e4: { audio_key: rows[1].audio_key }, e3: { audio_key: rows[2].audio_key }, e2: { audio_key: rows[3].audio_key } };
  // a newer episode arrived: e2 falls out of the newest three
  assert.deepEqual(planEpisodes({ episodes: rows, saved, keep: 3 }), { download: ['e5'], remove: ['e2'] });
  assert.deepEqual(planEpisodes({ episodes: rows, saved: { e5: { audio_key: rows[0].audio_key } }, keep: 3 }).remove, []);
});

test('an episode saved by hand is never pushed out by newer ones', () => {
  const saved = { e1: { audio_key: rows[4].audio_key } };
  assert.deepEqual(planEpisodes({ episodes: rows, saved, pinned: ['e1'], keep: 3 }), { download: ['e5', 'e4', 'e3'], remove: [] });
  // asked to save an old one that is not on the phone yet
  assert.deepEqual(planEpisodes({ episodes: rows, saved: {}, pinned: ['e2'], keep: 1 }).download, ['e5', 'e2']);
  // automatic keeping off: only what was saved by hand
  assert.deepEqual(planEpisodes({ episodes: rows, saved: { e5: { audio_key: rows[0].audio_key } }, pinned: [], keep: 0 }), { download: [], remove: ['e5'] });
});

test('a copy goes when its episode left the library or was re-recorded', () => {
  const saved = { gone: { audio_key: `audio/${'f'.repeat(64)}.mp3` }, e5: { audio_key: `audio/${'9'.repeat(64)}.mp3` } };
  const plan = planEpisodes({ episodes: rows, saved, pinned: ['gone'], keep: 3 });
  assert.deepEqual(plan.remove.sort(), ['e5', 'gone']);
  assert.ok(plan.download.includes('e5'), 'the new audio is fetched');
});

test('only episodes with audio count, newest first', () => {
  const library = { episodes: [
    { id: 'old', created_at: '2026-10-01T00:00:00Z', audio_key: `audio/${'a'.repeat(64)}.mp3` },
    { id: 'silent', created_at: '2026-10-09T00:00:00Z', audio_key: null },
    { id: 'new', created_at: '2026-10-05T00:00:00Z', audio_key: `audio/${'b'.repeat(64)}.mp3` },
  ] };
  assert.deepEqual(playableEpisodes(library).map((e) => e.id), ['new', 'old']);
});

test('an episode brings its paper and its playground cards with it', () => {
  const library = { papers: [{ id: 'p1', doc: 'papers/p1.json' }] };
  const entry = { id: 'e', doc: 'episodes/e.json', paper_id: 'p1' };
  assert.deepEqual(
    episodeDocKeys(entry, { follow_along: [{ doc: 'playgrounds/a.json' }, { doc: 'playgrounds/a.json' }, {}] }, { playground: { doc: 'playgrounds/b.json' } }, library),
    ['episodes/e.json', 'papers/p1.json', 'playgrounds/a.json', 'playgrounds/b.json'],
  );
  assert.deepEqual(episodeDocKeys(entry, null, null, { papers: [] }), ['episodes/e.json']);
});

// ---- the outbox as the page tells it -----------------------------------------------------

test('outbox: waiting reviews say what is happening and what to do, and never say they are lost', () => {
  assert.deepEqual(outboxPanel({ waiting: 0 }), { tone: 'ok', headline: 'All reviews are sent', detail: '', action: null, short: '' });
  const offline = outboxPanel({ waiting: 3, online: false });
  assert.equal(offline.headline, '3 reviews waiting to sync');
  assert.equal(offline.short, '3 waiting');
  assert.equal(offline.action, 'sync');
  assert.match(offline.detail, /back online\. Nothing is lost/);
  const one = outboxPanel({ waiting: 1, last: 'offline' });
  assert.equal(one.headline, '1 review waiting to sync');
  const ended = outboxPanel({ waiting: 2, last: 'auth' });
  assert.equal(ended.action, 'signin');
  assert.equal(ended.tone, 'warn');
  assert.match(ended.detail, /Sign in again.*Nothing is lost/);
  assert.equal(outboxPanel({ waiting: 2, last: 'error' }).tone, 'warn');
  assert.equal(outboxPanel({ waiting: 2, last: 'ok' }).detail, 'Sending now.');
});

test('outbox: rejected reviews and an unkept outbox are said once, with the rest', () => {
  const rejected = outboxPanel({ waiting: 0, rejected: 2 });
  assert.equal(rejected.tone, 'warn');
  assert.match(rejected.detail, /2 reviews could not be accepted/);
  assert.match(outboxPanel({ waiting: 0, durable: false }).detail, /not keeping reviews between visits/);
  assert.equal(outboxPanel({ waiting: 0, last: 'auth' }).action, 'signin');
  assert.match(outboxPanel({ waiting: 1, rejected: 1, last: 'offline' }).detail, /Nothing is lost\. 1 review could not/);
});

test('the header says the one thing that needs attention, in order', () => {
  const quiet = outboxPanel({ waiting: 0 });
  assert.deepEqual(headerStatus({ mode: 'live', outbox: quiet, updateReady: false }), { text: 'Offline & sync', tone: 'quiet' });
  assert.equal(headerStatus({ mode: 'signed-out', outbox: quiet }).text, 'Sign in again');
  assert.equal(headerStatus({ mode: 'live', outbox: outboxPanel({ waiting: 2, last: 'auth' }) }).text, 'Sign in again');
  assert.equal(headerStatus({ mode: 'live', outbox: quiet, updateReady: true }).text, 'Update ready');
  assert.equal(headerStatus({ mode: 'offline', outbox: quiet }).text, 'Offline');
  assert.equal(headerStatus({ mode: 'offline', outbox: outboxPanel({ waiting: 4 }) }).text, 'Offline · 4 waiting');
  assert.equal(headerStatus({ mode: 'live', outbox: outboxPanel({ waiting: 4 }) }).text, '4 waiting');
});

test('the banner appears only when the page is made of saved content', () => {
  assert.equal(connectionNotice({ mode: 'live' }), null);
  const out = connectionNotice({ mode: 'offline', publishedAt: '25 Oct' });
  assert.match(out.text, /published 25 Oct/);
  assert.equal(out.action, 'retry');
  const ended = connectionNotice({ mode: 'signed-out' });
  assert.equal(ended.action, 'signin');
  assert.equal(ended.tone, 'warn');
});

test('wording helpers', () => {
  assert.equal(formatBytes(512), '512 B');
  assert.equal(formatBytes(1536), '2 KB');
  assert.equal(formatBytes(5 * 1024 * 1024), '5.0 MB');
  assert.equal(formatBytes(26.4 * 1024 * 1024), '26 MB');
  assert.equal(formatBytes(-1), '');
  assert.equal(savedLabel({ status: 'saving', progress: 0.404 }), 'Saving 40%');
  assert.equal(savedLabel({ status: 'saved', pinned: true }), 'Saved on this device');
  assert.equal(savedLabel({ status: 'saved', pinned: false }), 'Kept offline');
  assert.equal(savedLabel({ status: 'none' }), '');
  for (const kind of ['auth', 'network', 'corrupt', 'full', 'weird']) assert.ok(downloadProblem(kind).length > 10);
  assert.match(installHint({ standalone: true }).text, /Installed/);
  assert.equal(installHint({ canPrompt: true }).prompt, true);
  assert.match(installHint({ ios: true }).text, /Share/);
  assert.match(installHint({}).text, /Add to Home Screen/);
});

// ---- the manager, with a cloud, a Cache API and IndexedDB behind it -----------------------

test('opening the app downloads the newest three episodes with their documents, and asks to keep them', async () => {
  const r = await rig(bundle([...specs(1, 2, 3, 4), { id: 'e-new', day: 9, body: 'newest' }]));
  try {
    const result = await r.offline.sync();
    assert.deepEqual({ status: result.status, downloaded: result.downloaded }, { status: 'ok', downloaded: 3 });
    const held = (await r.offline.episodes()).filter((e) => e.status === 'saved').map((e) => e.id).sort();
    assert.deepEqual(held, ['e-new', 'e3', 'e4']);
    assert.equal(r.audioIn().length, 3);
    // the documents an episode needs are on the device: the episode, its paper, the playground cards it follows
    for (const key of ['library.json', 'schedule.json', 'episodes/e-new.json', 'papers/p1.json', 'playgrounds/pg1.json']) {
      assert.ok(await r.kv.get(`doc:${key}`), key);
    }
    assert.equal(r.persisted.asked, 1);
  } finally { r.restore(); }
});

test('a newer episode pushes the oldest kept one out, and its audio is deleted', async () => {
  const r = await rig(bundle(specs(1, 2, 3, 4)));
  try {
    await r.offline.sync();
    const before = new Set(r.audioIn());
    assert.equal(before.size, 3);
    r.world.state.bundle = bundle(specs(1, 2, 3, 4, 5));
    await r.store.refresh();
    const result = await r.offline.sync();
    assert.equal(result.removed, 1);
    const held = (await r.offline.episodes()).filter((e) => e.status === 'saved').map((e) => e.id).sort();
    assert.deepEqual(held, ['e3', 'e4', 'e5']);
    assert.equal(r.audioIn().length, 3);
    assert.equal([...before].filter((p) => !r.audioIn().includes(p)).length, 1, 'exactly the oldest went');
  } finally { r.restore(); }
});

test('saved by hand survives newer episodes; removing it lets it go', async () => {
  const r = await rig(bundle(specs(1, 2, 3, 4)));
  try {
    await r.offline.sync();
    await r.offline.save('e1');
    assert.equal(r.audioIn().length, 4);
    r.world.state.bundle = bundle(specs(1, 2, 3, 4, 5, 6));
    await r.store.refresh();
    await r.offline.sync();
    const held = (await r.offline.episodes()).filter((e) => e.status === 'saved');
    assert.deepEqual(held.map((e) => e.id).sort(), ['e1', 'e4', 'e5', 'e6']);
    assert.equal(held.find((e) => e.id === 'e1').pinned, true);
    await r.offline.remove('e1');
    assert.deepEqual((await r.offline.episodes()).filter((e) => e.status === 'saved').map((e) => e.id).sort(), ['e4', 'e5', 'e6']);
    // removing a newest-three episode only un-saves it by hand: it is still kept
    await r.offline.remove('e6');
    assert.ok((await r.offline.episodes()).find((e) => e.id === 'e6').status === 'saved');
  } finally { r.restore(); }
});

test('with automatic keeping off, only what was saved by hand stays', async () => {
  const r = await rig(bundle(specs(1, 2, 3, 4)));
  try {
    await r.offline.sync();
    await r.offline.save('e2');
    await r.offline.setAutoKeep(false);
    assert.deepEqual((await r.offline.episodes()).filter((e) => e.status === 'saved').map((e) => e.id), ['e2']);
    assert.equal(await r.offline.autoKeep(), false);
    await r.offline.setAutoKeep(true);
    assert.equal((await r.offline.episodes()).filter((e) => e.status === 'saved').length, 4 - 1);
  } finally { r.restore(); }
});

test('with data saving on, the newest are not fetched unasked, but one saved by hand is', async () => {
  const r = await rig(bundle(specs(1, 2, 3)));
  try {
    const frugal = createOffline({
      store: r.store, kv: r.kv, cacheStorage: r.cacheStorage, fetchFn: r.world.fetchFn, isOnline: () => true,
      storageManager: null, autoAllowed: () => false,
    });
    await frugal.sync();
    assert.equal(r.audioIn().length, 0);
    await frugal.save('e1');
    assert.deepEqual((await frugal.episodes()).filter((e) => e.status === 'saved').map((e) => e.id), ['e1']);
  } finally { r.restore(); }
});

test('a download that is not the published file is never stored', async () => {
  const r = await rig(bundle(specs(1, 2, 3)));
  try {
    r.world.state.tamper = () => enc('<html>Sign in to continue</html>'); // a login page answering at 200
    await r.offline.sync();
    assert.equal(r.audioIn().length, 0);
    const rowsNow = await r.offline.episodes();
    assert.ok(rowsNow.every((row) => row.status === 'error' && row.problem === 'corrupt'));
  } finally { r.restore(); }
});

test('an ended session, a dropped connection and a full disk are told apart, and nothing partial stays', async () => {
  let r = await rig(bundle(specs(1)));
  try {
    const original = r.world.fetchFn;
    // documents load, then the session ends at the audio
    const audioPath = (await r.store.library()).episodes[0].audio_key.replace('audio/', '/api/audio/');
    r.world.state.session = true;
    const sessionEnds = createOffline({
      store: r.store, kv: r.kv, cacheStorage: r.cacheStorage, isOnline: () => true,
      fetchFn: async (path, init) => (path === audioPath ? { type: 'opaqueredirect', status: 0 } : original(path, init)),
      storageManager: null,
    });
    await sessionEnds.sync();
    assert.equal((await sessionEnds.episodes())[0].problem, 'auth');

    const drops = createOffline({
      store: r.store, kv: r.kv, cacheStorage: r.cacheStorage, isOnline: () => true,
      fetchFn: async (path, init) => { if (path === audioPath) throw new TypeError('dropped'); return original(path, init); },
      storageManager: null,
    });
    await drops.sync();
    assert.equal((await drops.episodes())[0].problem, 'network');

    const midway = createOffline({
      store: r.store, kv: r.kv, cacheStorage: r.cacheStorage, isOnline: () => true,
      fetchFn: async (path, init) => {
        if (path !== audioPath) return original(path, init);
        let sent = false;
        return new Response(new ReadableStream({ pull(controller) { if (sent) controller.error(new Error('reset')); else { sent = true; controller.enqueue(enc('aud')); } } }));
      },
      storageManager: null,
    });
    await midway.sync();
    assert.equal((await midway.episodes())[0].problem, 'network');
    assert.equal(r.audioIn().length, 0);
  } finally { r.restore(); }

  const quota = Object.assign(new Error('full'), { name: 'QuotaExceededError' });
  r = await rig(bundle(specs(1)), { failPut: quota });
  try {
    await r.offline.sync();
    assert.equal((await r.offline.episodes())[0].problem, 'full');
    assert.equal(r.audioIn().length, 0);
  } finally { r.restore(); }
});

test('a failure to reach the library deletes nothing', async () => {
  const kv = memoryKv();
  const cacheStorage = fakeCacheStorage();
  let r = await rig(bundle(specs(1, 2, 3)), { kv, cacheStorage });
  try {
    await r.offline.sync();
    assert.equal(r.audioIn().length, 3);
  } finally { r.restore(); }

  // a fresh page, no connection, and nothing in memory: the library comes from the device copy
  r = await rig(bundle(specs(1, 2, 3)), { kv, cacheStorage });
  try {
    r.world.state.online = false;
    r.state.online = false;
    const result = await r.offline.sync();
    assert.equal(result.status, 'ok');
    assert.equal(r.audioIn().length, 3);
    assert.equal(r.world.state.requests.filter((p) => p.startsWith('/api/audio/')).length, 0, 'no download attempted offline');
  } finally { r.restore(); }

  // no copy and no connection: cannot tell what is wanted, so touch nothing
  r = await rig(bundle(specs(1, 2, 3)), { kv: memoryKv(), cacheStorage });
  try {
    r.world.state.online = false;
    assert.equal((await r.offline.sync()).status, 'unavailable');
    assert.equal(r.audioIn().length, 3);
  } finally { r.restore(); }
});

test('audio the system took back is fetched again; audio nothing claims is dropped', async () => {
  const r = await rig(bundle(specs(1, 2, 3)));
  try {
    await r.offline.sync();
    const cache = await r.cacheStorage.open('nlp-audio-v1');
    const [first] = r.audioIn();
    await cache.delete(first);
    assert.equal(r.audioIn().length, 2);
    await (await r.cacheStorage.open('nlp-audio-v1')).put(`/api/audio/${'0'.repeat(64)}.mp3`, new Response('stray'));
    await r.offline.sync();
    assert.equal(r.audioIn().length, 3);
    assert.ok(r.audioIn().includes(first));
    assert.ok(!r.audioIn().some((p) => p.includes('0'.repeat(64))));
  } finally { r.restore(); }
});

test('one sync at a time: a second caller gets the first one’s result', async () => {
  const r = await rig(bundle(specs(1, 2, 3)));
  try {
    const [a, b] = await Promise.all([r.offline.sync(), r.offline.sync()]);
    assert.strictEqual(a, b);
    assert.equal(r.world.state.requests.filter((p) => p.startsWith('/api/audio/')).length, 3);
  } finally { r.restore(); }
});

// ---- the document copy behind the store ---------------------------------------------------

const small = () => bundle(specs(1));

test('with a good connection a kept document is not fetched again; when it changes, it is', async () => {
  const world = cloud(small());
  const previous = globalThis.fetch;
  globalThis.fetch = world.fetchFn;
  try {
    const kv = memoryKv();
    await createStore({ kv }).library();
    const fresh = createStore({ kv });
    world.state.requests.length = 0;
    await fresh.library();
    assert.deepEqual(world.state.requests, ['/api/manifest'], 'only the manifest is asked for');
    // a new publish changes library.json: the kept copy no longer matches its hash
    world.state.bundle = bundle(specs(1, 2));
    const later = createStore({ kv });
    assert.equal((await later.library()).episodes.length, 2);
  } finally { globalThis.fetch = previous; }
});

test('offline, kept documents are shown and the store says why; what was never kept says that', async () => {
  const world = cloud(small());
  const previous = globalThis.fetch;
  globalThis.fetch = world.fetchFn;
  try {
    const kv = memoryKv();
    const first = createStore({ kv });
    await first.library();
    assert.equal(first.status().mode, 'live');

    world.state.online = false;
    const offline = createStore({ kv });
    let changes = 0;
    offline.onStatus(() => { changes += 1; });
    assert.equal((await offline.library()).episodes.length, 1);
    assert.deepEqual(offline.status(), { mode: 'offline', publishedAt: '2026-10-25T00:00:00Z' });
    assert.equal(changes, 1);
    await assert.rejects(offline.doc('papers/p1.json'), (e) => e instanceof ReaderError && e.kind === 'unavailable');

    // back online: refresh says so
    world.state.online = true;
    await offline.refresh();
    assert.equal(offline.status().mode, 'live');
    assert.ok(await offline.doc('papers/p1.json'));
  } finally { globalThis.fetch = previous; }
});

test('an ended session shows what is kept and asks for a sign-in; with nothing kept it is a plain auth error', async () => {
  const world = cloud(small());
  const previous = globalThis.fetch;
  globalThis.fetch = world.fetchFn;
  try {
    const kv = memoryKv();
    await createStore({ kv }).library();
    world.state.session = false;
    const store = createStore({ kv });
    assert.ok(await store.library());
    assert.equal(store.status().mode, 'signed-out');
    await assert.rejects(store.doc('papers/p1.json'), (e) => e.kind === 'auth');
    await assert.rejects(createStore({ kv: memoryKv() }).library(), (e) => e.kind === 'auth');
  } finally { globalThis.fetch = previous; }
});

test('documents the publisher dropped are dropped from the device; an unreachable network drops nothing', async () => {
  const world = cloud(bundle(specs(1, 2)));
  const previous = globalThis.fetch;
  globalThis.fetch = world.fetchFn;
  try {
    const kv = memoryKv();
    const store = createStore({ kv });
    await store.doc('episodes/e1.json');
    await store.doc('episodes/e2.json');
    world.state.online = false;
    await createStore({ kv }).refresh();
    assert.ok(await kv.get('doc:episodes/e1.json'), 'offline: nothing is deleted');
    world.state.online = true;
    world.state.bundle = bundle(specs(2));
    await createStore({ kv }).refresh();
    assert.equal(await kv.get('doc:episodes/e1.json'), undefined);
    assert.ok(await kv.get('doc:episodes/e2.json'));
  } finally { globalThis.fetch = previous; }
});

test('a device store that fails is the same as having none', async () => {
  const world = cloud(small());
  const previous = globalThis.fetch;
  globalThis.fetch = world.fetchFn;
  try {
    const broken = { get: async () => { throw new Error('denied'); }, set: async () => { throw new Error('denied'); }, delete: async () => { throw new Error('denied'); }, keys: async () => { throw new Error('denied'); } };
    const store = createStore({ kv: broken });
    assert.equal((await store.library()).episodes.length, 1);
    world.state.online = false;
    await assert.rejects(createStore({ kv: broken }).library(), (e) => e.kind === 'network');
  } finally { globalThis.fetch = previous; }
});

// ---- registering the worker ---------------------------------------------------------------

function fakeNavigator({ controller = null, waiting = null, rejects = false } = {}) {
  const container = new EventTarget();
  container.controller = controller;
  const registration = Object.assign(new EventTarget(), { installing: null, waiting, update: async () => {} });
  const calls = { registered: [] };
  container.register = async (url, options) => {
    calls.registered.push([url, options]);
    if (rejects) throw new Error('blocked');
    return registration;
  };
  return { nav: { serviceWorker: container }, container, registration, calls };
}
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

test('the worker is registered at the root, bypassing the HTTP cache so updates are noticed', async () => {
  const fake = fakeNavigator();
  const worker = registerWorker({ nav: fake.nav });
  await tick();
  assert.deepEqual(fake.calls.registered, [['/sw.js', { scope: '/', updateViaCache: 'none' }]]);
  assert.equal(worker.state().registered, true);
  assert.equal(worker.state().updateReady, false);
});

test('an update waits for the person: nothing reloads until they ask', async () => {
  const waiting = { posted: [], postMessage(message) { this.posted.push(message); } };
  const fake = fakeNavigator({ controller: {}, waiting });
  let reloads = 0;
  let changes = 0;
  const worker = registerWorker({ nav: fake.nav, reload: () => { reloads += 1; }, onChange: () => { changes += 1; } });
  await tick();
  assert.equal(worker.state().updateReady, true);
  assert.ok(changes > 0);
  fake.container.dispatchEvent(new Event('controllerchange'));
  assert.equal(reloads, 1, 'the new worker took over, so the page reloads onto its code');
  fake.container.dispatchEvent(new Event('controllerchange'));
  assert.equal(reloads, 1, 'once');
  worker.applyUpdate();
  assert.deepEqual(waiting.posted, [{ type: 'SKIP_WAITING' }]);
});

test('the first install does not reload the page, and a browser without workers is fine', async () => {
  const first = fakeNavigator(); // no controller yet: this is the first worker
  let reloads = 0;
  registerWorker({ nav: first.nav, reload: () => { reloads += 1; } });
  await tick();
  first.container.dispatchEvent(new Event('controllerchange'));
  assert.equal(reloads, 0);

  const none = registerWorker({ nav: {} });
  assert.equal(none.state().supported, false);
  none.applyUpdate();
  none.check();

  const blocked = registerWorker({ nav: fakeNavigator({ rejects: true }).nav });
  await tick();
  assert.equal(blocked.state().error, 'blocked');
});
