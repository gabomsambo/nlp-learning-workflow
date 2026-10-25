/*
 * Fact-check helpers shared by Listen and the paper page. DOM wiring is
 * exercised in the browser. Run with `node --test tests/js/*.test.js`.
 */
const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const {
  panelState,
  highSeverityItems,
  scriptLineHref,
  truncateText,
  factcheckSummary,
} = require('../../webui/static/factcheck.js');
const { chipText } = require('../../webui/static/paper-page.js');

const OK_RUN = { ok: true, run: null };
const HASH = { ok: true, value: 'h1' };
const check = (data, status = 200) => ({ status, data });
const claim = (over) => Object.assign(
  { verdict: 'contradicted', severity: 'high', line_index: 3, span_text: 'x', evidence: [] },
  over,
);

test('deep links are zero-based, encoded, and fall back to the script when the line is unknown', () => {
  assert.equal(scriptLineHref('abc', 0), '/podcast/?script=abc&line=0');
  assert.equal(scriptLineHref('a b/c', 12), '/podcast/?script=a%20b%2Fc&line=12');
  assert.equal(scriptLineHref('abc', null), '/podcast/?script=abc');
  assert.equal(scriptLineHref('abc', -1), '/podcast/?script=abc');
  assert.equal(scriptLineHref('abc', 1.5), '/podcast/?script=abc');
  assert.equal(scriptLineHref('', 1), null);
});

test('only high-severity, undismissed issues are listed, in line order', () => {
  const items = highSeverityItems([
    claim({ line_index: 9 }),
    claim({ verdict: 'supported', severity: null }),
    claim({ verdict: 'partially_supported', severity: 'medium' }),
    claim({ verdict: 'unsupported', severity: 'low' }),
    claim({ line_index: 2, dismissed_at: '2026-10-25T00:00:00Z' }),
    claim({ line_index: 4, verdict: 'unsupported' }),
    claim({ verdict: 'not_checkable', severity: null }),
  ]);
  assert.deepEqual(items.map((c) => c.line_index), [4, 9]);
});

test('a stored check renders as done and carries its claims', () => {
  const state = panelState({ check: check({ script_sha256: 'h1', claims: [claim()] }), run: OK_RUN, hash: HASH });
  assert.equal(state.kind, 'done');
  assert.equal(state.claims.length, 1);
  assert.deepEqual(state.caveats, []);
});

test('no stored check and no run is "not checked yet", never an empty result', () => {
  const state = panelState({ check: check({}, 404), run: OK_RUN, hash: HASH });
  assert.equal(state.kind, 'none');
  assert.equal(state.claims, undefined);
});

test('a running check shows as running even when an older check exists', () => {
  const run = { ok: true, run: { id: 'r', status: 'running' } };
  assert.equal(panelState({ check: check({}, 404), run, hash: HASH }).kind, 'running');
  assert.equal(panelState({ check: check({ script_sha256: 'h1', claims: [] }), run, hash: HASH }).kind, 'running');
});

test('a failed run with no stored check says failed and keeps the reason', () => {
  const run = { ok: true, run: { id: 'r', status: 'failed', error: 'paper text missing' } };
  const state = panelState({ check: check({}, 404), run, hash: HASH });
  assert.equal(state.kind, 'failed');
  assert.equal(state.detail, 'paper text missing');
  const cancelled = panelState({ check: check({}, 404), run: { ok: true, run: { status: 'interrupted' } }, hash: HASH });
  assert.equal(cancelled.kind, 'failed');
  assert.match(cancelled.message, /interrupted/);
});

test('a missing migration (503) is its own state', () => {
  const state = panelState({ check: check({ detail: 'apply 022' }, 503), run: OK_RUN, hash: HASH });
  assert.equal(state.kind, 'migration');
  assert.equal(state.detail, 'apply 022');
});

test('a script that changed after its check is stale and hides the claims', () => {
  const state = panelState({ check: check({ script_sha256: 'old', claims: [claim()] }), run: OK_RUN, hash: HASH });
  assert.equal(state.kind, 'stale');
  assert.equal(state.claims, undefined);
});

test('a read that fails is unavailable, not "no issues"', () => {
  assert.equal(panelState({ check: { status: 0, data: { detail: 'offline' } }, run: OK_RUN, hash: HASH }).kind, 'unavailable');
  assert.equal(panelState({ check: check({}, 500), run: OK_RUN, hash: HASH }).kind, 'unavailable');
  const bothMissing = panelState({ check: check({}, 404), run: { ok: false, detail: 'boom' }, hash: HASH });
  assert.equal(bothMissing.kind, 'unavailable');
  assert.equal(bothMissing.detail, 'boom');
  const orphan = panelState({ check: check({}, 404), run: { ok: true, run: { status: 'succeeded' } }, hash: HASH });
  assert.equal(orphan.kind, 'unavailable');
});

test('a check that is shown with a doubt says what the doubt is', () => {
  const data = { script_sha256: 'h1', created_at: '2026-10-24T10:00:00Z', claims: [] };
  const noHash = panelState({ check: check(data), run: OK_RUN, hash: { ok: false, value: '' } });
  assert.equal(noHash.kind, 'done');
  assert.match(noHash.caveats[0], /unchanged/);
  const noRuns = panelState({ check: check(data), run: { ok: false, detail: 'x' }, hash: HASH });
  assert.match(noRuns.caveats[0], /run history/);
  const failedAfter = panelState({
    check: check(data),
    run: { ok: true, run: { status: 'failed', created_at: '2026-10-25T10:00:00Z' } },
    hash: HASH,
  });
  assert.match(failedAfter.caveats[0], /previous completed check/);
  const failedBefore = panelState({
    check: check(data),
    run: { ok: true, run: { status: 'failed', created_at: '2026-10-23T10:00:00Z' } },
    hash: HASH,
  });
  assert.deepEqual(failedBefore.caveats, []);
});

test('the summary is Listen\'s own, so the two pages cannot disagree', () => {
  assert.equal(
    factcheckSummary([claim(), { verdict: 'supported' }, { verdict: 'not_checkable' }]),
    '3 claims checked · 1 contradict the paper · 1 opinion (not checked) · 1 supported',
  );
});

test('the rail chip names the state, and a clean check still shows how many claims were graded', () => {
  assert.equal(chipText({ kind: 'done', claims: [claim(), claim()] }), 'Fact check: 2 high-severity · 2 claims checked');
  assert.equal(
    chipText({ kind: 'done', claims: [{ verdict: 'supported' }] }),
    'Fact check: no high-severity flags · 1 claims checked',
  );
  assert.equal(chipText({ kind: 'stale' }), 'Fact check is stale');
  assert.equal(chipText({ kind: 'migration' }), 'Fact check unavailable');
});

test('long quotes are cut at a word, short ones are untouched', () => {
  assert.equal(truncateText('  a   b ', 20), 'a b');
  assert.equal(truncateText('alpha beta gamma delta', 12), 'alpha beta…');
  assert.equal(truncateText(null, 5), '');
});

test('the paper page draws with text nodes only', () => {
  const source = fs.readFileSync(path.join(__dirname, '../../webui/static/paper-page.js'), 'utf8');
  assert.equal(source.includes('innerHTML'), false);
  assert.equal(source.includes('/factcheck\''), true);
  assert.equal(/fetch\([^)]*factcheck[^)]*\{\s*method/.test(source), false);
});
