/*
 * Pure helpers for Add papers. The DOM renderer and the poller are not covered
 * here; node --test has no document.
 */
const { test } = require('node:test');
const assert = require('node:assert');

const {
  paperPageHref,
  inLibrary,
  modeFromSearch,
  safePdfHref,
} = require('../../webui/static/add-papers.js');

test('paper links encode ids that contain a colon', () => {
  assert.strictEqual(paperPageHref('arxiv:2403.05525'), '/papers/arxiv%3A2403.05525');
  assert.strictEqual(paperPageHref('file:2dd76e910fbc'), '/papers/file%3A2dd76e910fbc');
  assert.strictEqual(paperPageHref(''), null);
  assert.strictEqual(paperPageHref(null), null);
});

test('in library is only claimed when the id set is known', () => {
  const ids = new Set(['arxiv:1']);
  assert.strictEqual(inLibrary('arxiv:1', ids), true);
  assert.strictEqual(inLibrary('arxiv:2', ids), false);
  assert.strictEqual(inLibrary('arxiv:1', null), false);
  assert.strictEqual(inLibrary('', ids), false);
});

test('mode comes from the query, then the page default', () => {
  assert.strictEqual(modeFromSearch('?pillar=ai-agents-tool-use', 'find'), 'find');
  assert.strictEqual(modeFromSearch('?mode=auto&run=abc', 'find'), 'auto');
  assert.strictEqual(modeFromSearch('mode=upload', 'find'), 'upload');
  assert.strictEqual(modeFromSearch('?mode=nope', 'auto'), 'auto');
  assert.strictEqual(modeFromSearch('', 'auto'), 'auto');
});

test('pdf links allow http, https, and file, and refuse scripts', () => {
  assert.strictEqual(
    safePdfHref('https://arxiv.org/pdf/2403.05525'),
    'https://arxiv.org/pdf/2403.05525',
  );
  assert.strictEqual(
    safePdfHref('file:///app/data/uploads/a.pdf'),
    'file:///app/data/uploads/a.pdf',
  );
  assert.strictEqual(safePdfHref('javascript:alert(1)'), null);
  assert.strictEqual(safePdfHref('data:text/html,hi'), null);
  assert.strictEqual(safePdfHref(''), null);
});

test('fit sort orders by verdict then rank, keeps every row, and puts unscored last', () => {
  const { sortCandidates, fitInfo } = require('../../webui/static/add-papers.js');
  const list = ['a', 'b', 'c', 'd', 'e'].map((id) => ({ paper: { id } }));
  const fits = {
    a: { verdict: 'off_topic', rank_fraction: 0.9 },
    b: { verdict: 'close_related', rank_fraction: 0.2 },
    c: { verdict: 'close_strong', rank_fraction: 0.04 },
    d: { verdict: 'close_related', rank_fraction: 0.1 },
    // e has no verdict at all
  };
  const ids = (rows) => rows.map((c) => c.paper.id);
  assert.deepStrictEqual(ids(sortCandidates(list, 'fit', fits)), ['c', 'd', 'b', 'a', 'e']);
  assert.deepStrictEqual(ids(sortCandidates(list, 'default', fits)), ['a', 'b', 'c', 'd', 'e']);
  assert.deepStrictEqual(ids(sortCandidates(list, 'fit', null)), ['a', 'b', 'c', 'd', 'e']);
  assert.strictEqual(fitInfo(undefined).label, 'Not scored');
  assert.strictEqual(fitInfo({ verdict: 'close_strong' }).label, 'Strong fit');
});
