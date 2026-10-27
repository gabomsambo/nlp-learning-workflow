import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { ReaderError, createStore } from '../public/js/data.js';
import {
  PLAYBACK_RATES, audioSrc, authorLine, dueSummary, episodeHref, episodeInitial, episodeMeta, factcheckState, filterPapers,
  listenRows, paperHref, parseRoute, pillarGlance, rateOrDefault, readAlongNote, readableScript, resumeSeconds, safePdfHref,
  sha256Hex,
} from '../public/js/lib.js';

test('routes round-trip paper ids that contain colons and slashes', () => {
  for (const id of ['arxiv:1706.03762', 'cs/0301001', 'file:abc123def456']) {
    assert.deepEqual(parseRoute(paperHref(id)), { name: 'paper', id, tab: 'summary' });
    assert.deepEqual(parseRoute(paperHref(id, 'cards')), { name: 'paper', id, tab: 'cards' });
  }
});

test('routes: known pages, and anything else is not found', () => {
  assert.equal(parseRoute('').name, 'today');
  assert.equal(parseRoute('#/').name, 'today');
  assert.equal(parseRoute('#/library').name, 'library');
  assert.equal(parseRoute('#/lessons').name, 'lessons');
  assert.equal(parseRoute('#/review').name, 'review');
  for (const bad of ['#/review/x', '#/paper', '#/paper/x/nope', '#/paper/%E0%A4%A', '#/library/x']) {
    assert.equal(parseRoute(bad).name, 'notfound', bad);
  }
});

test('due summary: new cards, due cards and scheduled cards are counted apart', () => {
  const now = Date.parse('2026-10-25T12:00:00Z');
  const cards = [
    { state: 'new', due: null },
    { state: 'review', due: '2026-10-24T12:00:00+00:00' },
    { state: 'review', due: '2026-10-26T12:00:00+00:00' },
    { state: 'learning', due: 'not a date' },
  ];
  assert.deepEqual(dueSummary(cards, now), { total: 4, due: 1, fresh: 1, upcoming: 2 });
  assert.deepEqual(dueSummary(undefined, now), { total: 0, due: 0, fresh: 0, upcoming: 0 });
});

test('pillar glance counts papers, lessons, cards and due cards', () => {
  const library = {
    pillars: [{ id: 'a', name: 'A' }, { id: 'b', name: 'B' }],
    papers: [
      { id: 'p1', pillar_id: 'a', has_lesson: true, card_count: 3 },
      { id: 'p2', pillar_id: 'a', has_lesson: false, card_count: 0 },
    ],
  };
  const schedule = { cards: [{ paper_id: 'p1', state: 'review', due: '2026-01-01T00:00:00Z' }, { paper_id: 'p1', state: 'new' }] };
  const [a, b] = pillarGlance(library, schedule, Date.parse('2026-10-25T00:00:00Z'));
  assert.deepEqual(a, { id: 'a', name: 'A', papers: 2, lessons: 1, cards: 3, due: 1 });
  assert.equal(b.papers, 0);
});

test('library filter: text, pillar and status', () => {
  const papers = [
    { id: '1', title: 'Attention', pillar_id: 'a', authors: ['Vaswani'], has_lesson: true, card_count: 2, episode_ids: ['e'] },
    { id: '2', title: 'Mamba', pillar_id: 'b', authors: ['Gu'], has_lesson: false, card_count: 0, episode_ids: [] },
  ];
  assert.deepEqual(filterPapers(papers, { query: 'vaswani' }).map((p) => p.id), ['1']);
  assert.deepEqual(filterPapers(papers, { pillar: 'b' }).map((p) => p.id), ['2']);
  assert.deepEqual(filterPapers(papers, { status: 'episode' }).map((p) => p.id), ['1']);
  assert.deepEqual(filterPapers(papers, { status: 'cards', query: 'mamba' }), []);
  assert.equal(filterPapers(papers).length, 2);
});

test('only web links are offered for a PDF; a stored upload path is not', () => {
  assert.equal(safePdfHref('https://arxiv.org/pdf/1706.03762'), 'https://arxiv.org/pdf/1706.03762');
  for (const bad of ['file:///app/data/uploads/x.pdf', 'javascript:alert(1)', 'data:text/html,x', null, '']) {
    assert.equal(safePdfHref(bad), null, String(bad));
  }
});

test('the script is shown as reading text, without cue lines or speaker tags', () => {
  const script = '[MUSIC]\n\n[HOST]: Welcome back.\n\n[PAUSE]\n\n[HOST]: Next topic.\n\n\n\n[SFX: chime]';
  assert.equal(readableScript(script), 'Welcome back.\n\nNext topic.');
});

test('author line abbreviates long lists', () => {
  assert.equal(authorLine(['A', 'B'], 4), 'A, B');
  assert.equal(authorLine(['A', 'B', 'C', 'D', 'E', 'F'], 2), 'A, B and 4 more');
  assert.equal(authorLine([]), '');
});

// ---- the store: a document that does not match the manifest is never shown ----

async function jsonResponse(value) {
  return new Response(JSON.stringify(value), { status: 200 });
}

async function withFetch(handler, run) {
  const original = globalThis.fetch;
  globalThis.fetch = handler;
  try {
    return await run();
  } finally {
    globalThis.fetch = original;
  }
}

test('store: a document that matches its manifest hash is returned', async () => {
  const body = JSON.stringify({ papers: [] });
  const sha = await sha256Hex(new TextEncoder().encode(body));
  const manifest = { docs: [{ key: 'library.json', sha256: sha }] };
  await withFetch(async (url) => (url === '/api/manifest' ? jsonResponse(manifest) : new Response(body)), async () => {
    assert.deepEqual(await createStore().library(), { papers: [] });
  });
});

test('store: a mismatch refetches the manifest once, then reports a publish in progress', async () => {
  const manifest = { docs: [{ key: 'library.json', sha256: 'f'.repeat(64) }] };
  let manifestFetches = 0;
  await withFetch(async (url) => {
    if (url === '/api/manifest') { manifestFetches += 1; return jsonResponse(manifest); }
    return new Response('{"papers":[]}');
  }, async () => {
    await assert.rejects(createStore().library(), (error) => error instanceof ReaderError && error.kind === 'stale');
  });
  assert.equal(manifestFetches, 2);
});

test('store: sign-in and missing-bundle responses are told apart', async () => {
  for (const [response, kind] of [[new Response('', { status: 401 }), 'auth'], [new Response('', { status: 404 }), 'empty'], [new Response('', { status: 500 }), 'network']]) {
    await withFetch(async () => response.clone(), async () => {
      await assert.rejects(createStore().library(), (error) => error.kind === kind, kind);
    });
  }
  await withFetch(async () => { throw new TypeError('offline'); }, async () => {
    await assert.rejects(createStore().library(), (error) => error.kind === 'network');
  });
});

// ---- Listen ---------------------------------------------------------------

const SHA = 'ab'.repeat(32);

test('routes: Listen and one episode, ids are percent-encoded', () => {
  assert.deepEqual(parseRoute('#/listen'), { name: 'listen' });
  const id = '11111111-1111-4111-8111-111111111111';
  assert.deepEqual(parseRoute(episodeHref(id)), { name: 'episode', id });
  assert.equal(parseRoute('#/listen/a/b').name, 'notfound');
  assert.equal(parseRoute('#/listen/%E0%A4%A').name, 'notfound');
});

test('audioSrc only builds the one URL shape the Worker serves', () => {
  assert.equal(audioSrc(`audio/${SHA}.mp3`), `/api/audio/${SHA}.mp3`);
  for (const bad of [null, undefined, '', 'audio/x.mp3', `audio/${SHA}.wav`, `../audio/${SHA}.mp3`, `audio/${SHA}.mp3?x=1`, `audio/${SHA.toUpperCase()}.mp3`]) {
    assert.equal(audioSrc(bad), null, String(bad));
  }
});

test('episodes you can play come first, newest first inside each group', () => {
  const library = {
    episodes: [
      { id: 'a', created_at: '2026-10-20T00:00:00Z', audio_key: null },
      { id: 'b', created_at: '2026-10-21T00:00:00Z', audio_key: `audio/${SHA}.mp3` },
      { id: 'c', created_at: '2026-10-23T00:00:00Z', audio_key: null },
      { id: 'd', created_at: '2026-10-22T00:00:00Z', audio_key: `audio/${SHA}.mp3` },
    ],
  };
  assert.deepEqual(listenRows(library).map((e) => e.id), ['d', 'b', 'c', 'a']);
  assert.deepEqual(listenRows({}), []);
});

test('a fact check is only painted onto the script it was run on', () => {
  assert.equal(factcheckState({ fact_check: null }).kind, 'none');
  assert.equal(factcheckState({}).kind, 'none');
  const claims = [{ id: 1 }];
  assert.equal(factcheckState({ fact_check: { stale: true, claims } }).kind, 'stale');
  const done = factcheckState({ fact_check: { stale: false, claims } });
  assert.equal(done.kind, 'done');
  assert.deepEqual(done.claims, claims);
  assert.deepEqual(factcheckState({ fact_check: { stale: false } }).claims, []);
});

test('resume: not the first seconds, not the last ten, never junk', () => {
  assert.equal(resumeSeconds('62', 600), 62);
  assert.equal(resumeSeconds(3, 600), 0);
  assert.equal(resumeSeconds(595, 600), 0);
  assert.equal(resumeSeconds(595, 0), 595);
  for (const junk of [null, undefined, '', 'abc', NaN, -5]) assert.equal(resumeSeconds(junk, 600), 0, String(junk));
});

test('playback rate falls back to 1 for anything the menu does not offer', () => {
  for (const rate of PLAYBACK_RATES) assert.equal(rateOrDefault(String(rate)), rate);
  for (const junk of [null, '', 'fast', '3', 0, -1]) assert.equal(rateOrDefault(junk), 1, String(junk));
});

test('the notice above the text says what is missing, and is silent when nothing is', () => {
  assert.match(readAlongNote({ hasAudio: false }), /no audio/i);
  assert.match(readAlongNote({ hasAudio: true, timelineState: 'missing' }), /no read-along timing/i);
  assert.match(readAlongNote({ hasAudio: true, timelineState: 'stale' }), /edited after/i);
  assert.match(readAlongNote({ hasAudio: true, timelineState: 'current', wordLevel: false }), /only the line/i);
  assert.equal(readAlongNote({ hasAudio: true, timelineState: 'current', wordLevel: true }), '');
});

test('episode cover letter skips the "Deep Dive:" every title starts with', () => {
  assert.equal(episodeInitial('Deep Dive: Attention Is All You Need'), 'A');
  assert.equal(episodeInitial('deep dive : mamba'), 'M');
  assert.equal(episodeInitial('Plain title'), 'P');
  assert.equal(episodeInitial('…'), 'E');
  assert.equal(episodeInitial(null), 'E');
});

test('episode meta joins what is known', () => {
  const meta = episodeMeta({ duration_seconds: 1500, created_at: '2026-10-22T12:00:00+00:00' }, 'Linguistic Theory');
  assert.equal(meta, 'Linguistic Theory · 25 min · 22 Oct 2026');
  assert.equal(episodeMeta({}, ''), '');
});

test('the page loads the shared scripts before the module, and the asset build copies them', () => {
  const html = readFileSync(new URL('../public/index.html', import.meta.url), 'utf8');
  const build = readFileSync(new URL('../scripts/build-assets.sh', import.meta.url), 'utf8');
  const shared = ['factcheck.js', 'readalong.js', 'listen-core.js', 'playground-core.js', 'playground.js', 'follow-along.js'];
  const order = shared.map((name) => html.indexOf(`/static/${name}`));
  const module = html.indexOf('/js/reader.js');
  for (const at of order) assert.ok(at > -1 && at < module);
  // follow-along.js reads ReadAlong and ListenCore, and playground.js PlaygroundCore, when they load.
  assert.ok(order[5] > order[1] && order[5] > order[2] && order[4] > order[3]);
  for (const name of shared) assert.ok(build.includes(name), name);
});
