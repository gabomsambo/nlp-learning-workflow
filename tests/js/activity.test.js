/*
 * Plain-language run summaries for the Activity page.
 * Run with `node --test tests/js/*.test.js`.
 */
const { test } = require('node:test');
const assert = require('node:assert');

const {
  outcomePhrase,
  toneOf,
  formatDuration,
  humaniseTime,
  paperHref,
  matchesFilters,
  runDuration,
} = require('../../webui/static/activity.js');

const NOW = new Date('2026-10-23T12:00:00Z');

test('a paper id with a colon is encoded in the paper link', () => {
  assert.equal(paperHref('file:2dd76e910fbc'), '/papers/file%3A2dd76e910fbc');
  assert.equal(paperHref(''), null);
  assert.equal(paperHref(null), null);
});

test('the summary never includes the raw exception', () => {
  const run = {
    kind: 'upload',
    status: 'failed',
    error: 'Traceback (most recent call last):\n  File "upload.py", line 1\ninsufficient_quota',
    result_title: 'Attention Is All You Need',
    result_added: 'true',
  };
  const phrase = outcomePhrase(run);
  assert.equal(phrase.includes('Traceback'), false);
  assert.equal(phrase.includes('insufficient_quota'), false);
  assert.match(phrase, /Added .*Attention Is All You Need/);
  assert.match(phrase, /did not finish/);
  assert.equal(toneOf(run), 'warn');
});

test('a finished upload names the paper', () => {
  const phrase = outcomePhrase({
    kind: 'upload',
    status: 'succeeded',
    result_title: 'Attention Is All You Need',
  });
  assert.match(phrase, /Added .*Attention Is All You Need/);
  assert.equal(toneOf({ kind: 'upload', status: 'succeeded' }), 'ok');
});

test('an empty daily run is not described as a failure', () => {
  assert.equal(
    outcomePhrase({ kind: 'run_daily', status: 'succeeded', papers_processed: 0, papers_failed: 0 }),
    'Finished with nothing new to process',
  );
});

test('filters hide probe rows and match kind plus pillar', () => {
  const run = { kind: 'discover', pillar_id: 'ai-agents-tool-use', status: 'succeeded' };
  assert.equal(matchesFilters(run, 'all', ''), true);
  assert.equal(matchesFilters(run, 'discover', 'ai-agents-tool-use'), true);
  assert.equal(matchesFilters(run, 'upload', ''), false);
  assert.equal(matchesFilters({ pillar_id: 'status-probe', kind: 'upload' }, 'all', ''), false);
});

test('duration and relative time are human', () => {
  assert.equal(formatDuration('2026-10-23T12:00:00Z', '2026-10-23T12:02:05Z'), '2m 5s');
  assert.equal(
    runDuration({
      status: 'succeeded',
      started_at: '2026-10-23T11:00:00Z',
      finished_at: '2026-10-23T12:00:00Z',
    }, NOW),
    '1h',
  );
  assert.equal(humaniseTime('2026-10-23T09:00:00Z', NOW), '3 hours ago');
  assert.equal(humaniseTime('not-a-date', NOW), '');
});
