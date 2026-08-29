/*
 * Tests for the pure helpers in webui/static/run-progress.js.
 *
 * Run with `node --test tests/js/` — stdlib only, no npm install, no bundler. The
 * project has no build step and refuses one, so this is the only way to test any of
 * this file's logic at all. run-progress.js exports these helpers through a CommonJS
 * guard at the bottom while still loading in the browser as a plain <script>.
 *
 * Scope, honestly: this covers the pure functions only — roughly the formatting and
 * state-classification logic. The polling loop, AbortController teardown and the
 * 404-clears-localStorage path all need a DOM and are NOT covered here. Testing those
 * means jsdom (npm) or Playwright (a browser download in CI); both were considered
 * and rejected as disproportionate. The Chrome MCP session that verified this branch
 * exercised them by hand.
 */
const { test } = require('node:test');
const assert = require('node:assert');

const {
  isTerminal,
  stageLabel,
  summarise,
  statusColor,
  countCompleted,
  displayStatus,
  duration,
  candidateCount,
  safely,
} = require('../../webui/static/run-progress.js');

/** Eleven stages, the first `done` of them completed. */
function stages(done, extra = []) {
  const names = ['discovery', 'search', 'enqueue', 'pop_queue', 'process',
                 'ingest', 'summarize', 'synthesize', 'quiz', 'persist', 'vectors'];
  return names.map((name, i) => ({
    name,
    status: i < done ? 'completed' : 'pending',
    ...(extra[i] || {}),
  }));
}

/* ------------------------------------------------------------- isTerminal */

test('every terminal status stops polling', () => {
  for (const s of ['succeeded', 'failed', 'cancelled', 'interrupted']) {
    assert.equal(isTerminal(s), true, `${s} should be terminal`);
  }
});

test('in-flight statuses keep polling', () => {
  assert.equal(isTerminal('running'), false);
  assert.equal(isTerminal('pending'), false);
});

/* ------------------------------------------------------------- stageLabel */

test('machine slugs are given human labels', () => {
  // The user was shown "pop_queue" and "Running — pop_queue".
  assert.equal(stageLabel('pop_queue'), 'Taking papers off the queue');
  assert.equal(stageLabel('vectors'), 'Indexing for search');
});

test('an unmapped stage falls back to its raw name rather than blank', () => {
  assert.equal(stageLabel('some_future_stage'), 'some_future_stage');
});

/* --------------------------------------------------------------- summarise */

test('a run that found nothing does not say Failed', () => {
  // The most frequent outcome once a pillar has caught up. It used to render a bare
  // crimson "Failed" above eleven green stages, with no reason, because there was
  // no error to report.
  const run = { status: 'succeeded', papers_processed: 0, papers_failed: 0, stages: stages(11) };
  assert.equal(summarise(run), 'Done — no new papers to process');
});

test('a partial success reports the count AND the reason', () => {
  const run = {
    status: 'succeeded', papers_processed: 2, papers_failed: 1,
    error: 'Failed to process paper 2401.1: 404', stages: stages(11),
  };
  const text = summarise(run);
  assert.match(text, /2 paper\(s\) processed/);
  assert.match(text, /1 failed/);
  assert.match(text, /404/);   // the reason must reach the user, not just the count
});

test('a running run reports its position, not a slug', () => {
  const run = { status: 'running', current_stage: 'summarize', stages: stages(6) };
  assert.equal(summarise(run), 'Step 7 of 11 — Summarizing');
});

test('a failure carries its error text', () => {
  const run = { status: 'failed', error: 'RateLimitError 429', stages: stages(6) };
  assert.equal(summarise(run), 'Failed — RateLimitError 429');
});

test('an interrupted run explains itself rather than blaming the pipeline', () => {
  const run = { status: 'interrupted', stages: stages(4) };
  assert.match(summarise(run), /server stopped/);
});

test('a pending run with no stage yet says Starting', () => {
  assert.equal(summarise({ status: 'pending', stages: stages(0) }), 'Starting…');
});

/* ------------------------------------------------- summarise: discovery runs */

/** The seven discovery stages, the first `done` of them completed. */
function discoverStages(done) {
  const names = ['discover_context', 'discover_queries', 'discover_vectors',
                 'discover_arxiv', 'discover_semantic_scholar',
                 'discover_citations', 'discover_rank'];
  return names.map((name, i) => ({
    name, status: i < done ? 'completed' : 'pending',
  }));
}

test('discovery steps are named for what the user gets, not for the function', () => {
  assert.equal(stageLabel('discover_arxiv'), 'Searching arXiv');
  assert.equal(stageLabel('discover_vectors'), 'Searching your library by meaning');
  assert.equal(stageLabel('discover_queries'), 'Writing search queries');
});

test('a discovery run counts candidates, not processed papers', () => {
  // papers_processed is 0 on a discovery run on purpose: it processed no papers.
  // Reporting that number here would say "0 papers" over a full table of results.
  const run = {
    kind: 'discover', status: 'succeeded', papers_processed: 0, papers_failed: 0,
    stages: discoverStages(7),
    result: { candidates: [{}, {}, {}], sources_used: ['arxiv'] },
  };
  assert.equal(summarise(run), 'Done — 3 candidate paper(s) found');
});

test('a discovery that matched nothing says so plainly', () => {
  const run = {
    kind: 'discover', status: 'succeeded', papers_processed: 0,
    stages: discoverStages(7), result: { candidates: [] },
  };
  assert.equal(summarise(run), 'Done — no papers matched');
});

test('a discovery that succeeded with a source down still says which', () => {
  // The point of the exercise: "ten papers" and "ten papers, but arXiv was
  // rate-limited" are different claims, and only the user can decide what to do.
  const run = {
    kind: 'discover', status: 'succeeded', papers_processed: 0,
    stages: discoverStages(7),
    error: 'rate-limited — try again in a few minutes',
    result: { candidates: [{}, {}] },
  };
  assert.match(summarise(run), /2 candidate\(s\)|2 candidate paper\(s\) found/);
  assert.match(summarise(run), /rate-limited/);
});

test('a discovery run in flight reports its step like any other run', () => {
  const run = {
    kind: 'discover', status: 'running', current_stage: 'discover_arxiv',
    stages: discoverStages(3),
  };
  assert.equal(summarise(run), 'Step 4 of 7 — Searching arXiv');
});

test('a missing or malformed payload counts as zero candidates, not a crash', () => {
  assert.equal(candidateCount({}), 0);
  assert.equal(candidateCount({ result: null }), 0);
  assert.equal(candidateCount({ result: { candidates: 'not an array' } }), 0);
  assert.equal(candidateCount({ result: { candidates: [{}] } }), 1);
});

/* ------------------------------------------------------------- statusColor */

test('a clean success is green but a partial success is not', () => {
  // Painting a run that lost papers the same green as a perfect one is how partial
  // failures went unnoticed.
  assert.equal(statusColor({ status: 'succeeded', papers_failed: 0 }), 'green');
  assert.equal(statusColor({ status: 'succeeded', papers_failed: 1 }), 'darkorange');
});

test('failure is crimson, cancellation and interruption are orange', () => {
  assert.equal(statusColor({ status: 'failed' }), 'crimson');
  assert.equal(statusColor({ status: 'cancelled' }), 'darkorange');
  assert.equal(statusColor({ status: 'interrupted' }), 'darkorange');
});

/* ----------------------------------------------------------- displayStatus */

test('a terminal run never shows a stage as still running', () => {
  // Nothing rewrites child stage rows when a run dies, so a failed run carried a
  // stage marked running and rendered a bold spinner on a run that ended minutes ago.
  assert.equal(displayStatus({ status: 'running' }, { status: 'failed' }), 'unknown');
  assert.equal(displayStatus({ status: 'running' }, { status: 'interrupted' }), 'unknown');
});

test('a live run shows a running stage as running', () => {
  assert.equal(displayStatus({ status: 'running' }, { status: 'running' }), 'running');
});

test('a succeeded run resolves a stuck stage to completed, not unknown', () => {
  // If the run as a whole succeeded, an unclosed stage did in fact finish.
  assert.equal(displayStatus({ status: 'running' }, { status: 'succeeded' }), 'completed');
});

test('settled stages are shown as they are stored', () => {
  assert.equal(displayStatus({ status: 'failed' }, { status: 'failed' }), 'failed');
  assert.equal(displayStatus({ status: 'skipped' }, { status: 'succeeded' }), 'skipped');
});

/* ---------------------------------------------------------- countCompleted */

test('skipped stages count as done for progress purposes', () => {
  // Quiz is skipped when disabled; a bar that never reaches the end because of a
  // deliberately-skipped stage reads as a stall.
  const s = stages(4);
  s[4] = { name: 'process', status: 'skipped' };
  assert.equal(countCompleted({ stages: s }), 5);
});

test('a run with no stages counts zero rather than throwing', () => {
  assert.equal(countCompleted({}), 0);
});

/* ----------------------------------------------------------------- duration */

test('a stage that never started has no duration rather than NaN', () => {
  assert.equal(duration({ started_at: null }), '');
});

test('microsecond-precision PostgREST timestamps parse without NaN', () => {
  // PostgREST emits 6 fractional digits; ECMA-262 only defines 3.
  const stage = {
    started_at: '2026-08-16T12:00:00.123456+00:00',
    finished_at: '2026-08-16T12:00:04.654321+00:00',
  };
  assert.match(duration(stage), /^4\.5\d s?$|^4\.\d+s$/);
});

test('durations over a minute are shown as minutes and seconds', () => {
  const stage = {
    started_at: '2026-08-16T12:00:00+00:00',
    finished_at: '2026-08-16T12:01:30+00:00',
  };
  assert.equal(duration(stage), '1m30s');
});

test('an unfinished stage measures against now', () => {
  const started = new Date(Date.now() - 5000).toISOString();
  assert.match(duration({ started_at: started, finished_at: null }), /^[45]\.\ds$/);
});


/*
 * safely(): a rendering handler must not be able to break the poll.
 *
 * Before this, handlers ran inside tick()'s try, so a TypeError in a page's render
 * was caught by the *transport* catch and announced as "Lost contact with the
 * server — retrying" while the throw skipped the isTerminal check, leaving the page
 * polling a finished run. Measured against a discovery run whose stored candidates
 * were the wrong shape.
 */
test('a throwing handler is contained rather than propagated', () => {
  const errors = [];
  const realError = console.error;
  console.error = (...args) => errors.push(args);
  try {
    assert.doesNotThrow(() => safely(() => { throw new TypeError('boom'); }, {}, 'onDone'));
  } finally {
    console.error = realError;
  }
  assert.equal(errors.length, 1, 'the failure is logged, not swallowed silently');
  assert.match(String(errors[0][0]), /onDone/);
});

test('a handler that works is called with its argument, once', () => {
  const seen = [];
  safely((run) => seen.push(run), { id: 'r1' }, 'onUpdate');
  assert.deepEqual(seen, [{ id: 'r1' }]);
});

test('a missing handler is not an error', () => {
  assert.doesNotThrow(() => safely(undefined, {}, 'onGone'));
  assert.doesNotThrow(() => safely(null, {}, 'onGone'));
});
