import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ReaderError, createStore } from '../public/js/data.js';
import {
  authorLine, dueSummary, filterPapers, paperHref, parseRoute, pillarGlance, readableScript, safePdfHref, sha256Hex,
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
