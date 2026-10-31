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
  applyStatusClass,
  stageClass,
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

test('the two upload-only stages have prose labels', () => {
  // A user reading "upload_fetch" is reading a machine slug. The six steps after
  // these two deliberately share _process_paper's existing labels.
  assert.equal(stageLabel('upload_fetch'), 'Downloading the PDF');
  assert.equal(stageLabel('upload_metadata'), 'Looking up paper details');
  assert.equal(stageLabel('ingest'), 'Downloading and reading the PDF');
});

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

test('a durable audio waiter shows IndexTTS state and only its start command', () => {
  const waiting = summarise({
    kind: 'podcast_audio', status: 'pending', current_stage: null, stages: [],
    result: {
      queue_state: 'waiting_for_indextts',
      queue_message: 'IndexTTS is not running on port 3026.',
      start_command: './start-indextts.sh',
    },
  });
  assert.match(waiting, /^Waiting for IndexTTS/);
  assert.match(waiting, /Start command: \.\/start-indextts\.sh/);

  const wrong = summarise({
    kind: 'podcast_audio', status: 'pending', current_stage: null, stages: [],
    result: { queue_state: 'wrong_service', queue_message: 'Dots TTS answered.' },
  });
  assert.match(wrong, /Wrong service/);
  assert.doesNotMatch(wrong, /Start command/);

  const mismatch = summarise({
    kind: 'podcast_audio', status: 'pending', current_stage: null, stages: [],
    result: { queue_state: 'contract_mismatch', queue_message: 'Missing \/gen_single.' },
  });
  assert.match(mismatch, /contract mismatch/i);
  assert.doesNotMatch(mismatch, /Start command/);
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

/* --------------------------------------------------- summarise: upload runs */

/** Eight stages, the first `done` of them completed. */
function uploadStages(done) {
  const names = ['upload_fetch', 'upload_metadata', 'ingest', 'summarize',
                 'synthesize', 'quiz', 'persist', 'vectors'];
  return names.map((name, i) => ({
    name,
    status: i < done ? 'completed' : 'pending',
  }));
}

test('an upload in flight names the step, not "1 paper(s) processed"', () => {
  const run = {
    kind: 'upload',
    status: 'running',
    current_stage: 'upload_fetch',
    stages: uploadStages(0),
  };
  assert.equal(summarise(run), 'Step 1 of 8 — Downloading the PDF');
});

test('a clean upload says what it added', () => {
  const run = {
    kind: 'upload',
    status: 'succeeded',
    papers_processed: 1,
    stages: uploadStages(8),
    result: { added: true, title: 'Attention Is All You Need' },
  };
  assert.match(summarise(run), /Attention Is All You Need/);
  assert.match(summarise(run), /finished processing/);
});

test('a failed upload whose paper WAS added does not tell the user to start over', () => {
  // The paper is in the library. "Failed" full stop sends them to re-upload
  // something they already have.
  const run = {
    kind: 'upload',
    status: 'failed',
    papers_failed: 1,
    stages: uploadStages(8),
    error: 'summarizer: model refused',
    result: { added: true, title: 'A Paper' },
  };
  const text = summarise(run);
  assert.match(text, /Added "A Paper" to the library/);
  assert.match(text, /some steps did not finish/);
  assert.match(text, /model refused/);
});

test('an upload that never reached the library says nothing was added', () => {
  const run = {
    kind: 'upload',
    status: 'failed',
    stages: uploadStages(1),
    error: 'Could not download the PDF: 404',
  };
  const text = summarise(run);
  assert.match(text, /nothing was added/);
  assert.match(text, /404/);
});

test('an upload with no result payload never claims a paper was added', () => {
  const run = { kind: 'upload', status: 'failed', stages: uploadStages(0) };
  assert.doesNotMatch(summarise(run), /library/);
});

/* ------------------------------------------------------------- statusColor */

test('a clean success is ok but a partial success is a warning', () => {
  // Painting a run that lost papers the same success tone as a perfect one is how
  // partial failures went unnoticed. The strings are class names; app.css maps them
  // onto the theme tokens.
  assert.equal(statusColor({ status: 'succeeded', papers_failed: 0 }), 'run-ok');
  assert.equal(statusColor({ status: 'succeeded', papers_failed: 1 }), 'run-warn');
});

test('failure is danger, cancellation and interruption are warnings', () => {
  assert.equal(statusColor({ status: 'failed' }), 'run-danger');
  assert.equal(statusColor({ status: 'cancelled' }), 'run-warn');
  assert.equal(statusColor({ status: 'interrupted' }), 'run-warn');
});

test('a failed upload whose paper is in the library is a partial failure', () => {
  // A danger tone there says "start again", which is the wrong instruction.
  assert.equal(
    statusColor({ status: 'failed', result: { added: true } }), 'run-warn');
  assert.equal(statusColor({ status: 'failed', result: {} }), 'run-danger');
  // Still safe for the string form, which several callers use.
  assert.equal(statusColor('failed'), 'run-danger');
});

test('stage rows use a class per displayed status', () => {
  assert.equal(stageClass('completed'), 'run-stage-completed');
  assert.equal(stageClass('running'), 'run-stage-running');
  assert.equal(stageClass('failed'), 'run-stage-failed');
  assert.equal(stageClass('unknown'), 'run-stage-unknown');
  assert.equal(stageClass('not-a-status'), 'run-stage-pending');
});

test('applyStatusClass swaps the tone class and clears an inline colour', () => {
  const removed = [];
  const added = [];
  const el = {
    classList: {
      remove(...names) { removed.push(...names); },
      add(name) { added.push(name); },
    },
    style: { removeProperty(prop) { this.removed = prop; } },
  };
  applyStatusClass(el, { status: 'failed' });
  assert.deepEqual(added, ['run-danger']);
  assert.ok(removed.includes('run-ok'));
  assert.equal(el.style.removed, 'color');
  added.length = 0;
  applyStatusClass(el, 'run-warn');
  assert.deepEqual(added, ['run-warn']);
});

/* --------------------------------------------- summarise: podcast_script */

function podcastScriptStages(done) {
  const names = [
    'podcast_prepare', 'podcast_facts_outline', 'podcast_core_concepts',
    'podcast_metrics', 'podcast_limitations', 'podcast_synthesize',
    'podcast_persist',
  ];
  return names.map((name, i) => ({
    name,
    status: i < done ? 'completed' : (i === done ? 'running' : 'pending'),
  }));
}

test('podcast script stages have prose labels', () => {
  assert.equal(stageLabel('podcast_facts_outline'), 'Extracting facts outline');
  assert.equal(stageLabel('podcast_synthesize'), 'Writing the podcast script');
  assert.equal(stageLabel('podcast_persist'), 'Saving the script');
});

test('a podcast script run in flight names the Ground Pack step', () => {
  const run = {
    kind: 'podcast_script',
    status: 'running',
    current_stage: 'podcast_metrics',
    stages: podcastScriptStages(3),
  };
  assert.equal(summarise(run), 'Step 4 of 7 — Extracting metrics and datasets');
});

test('a saved podcast script run names the title and word count', () => {
  const run = {
    kind: 'podcast_script',
    status: 'succeeded',
    stages: podcastScriptStages(7),
    result: { saved: true, title: 'Deep Dive: Attention', word_count: 4803 },
  };
  assert.match(summarise(run), /Deep Dive: Attention/);
  assert.match(summarise(run), /4803 words/);
});

test('fact-check stages have prose labels and a claim summary', () => {
  assert.equal(stageLabel('factcheck_extract'), 'Splitting the script into claims');
  assert.equal(stageLabel('factcheck_screen'), 'Checking claims against the paper');
  assert.equal(stageLabel('factcheck_rejudge'), 'Re-checking flagged claims');
  assert.equal(stageLabel('factcheck_anchor'), 'Locating evidence quotes');
  assert.equal(stageLabel('factcheck_persist'), 'Saving the fact check');
  const names = [
    'factcheck_extract', 'factcheck_screen', 'factcheck_rejudge',
    'factcheck_anchor', 'factcheck_persist',
  ];
  const running = {
    kind: 'podcast_factcheck',
    status: 'running',
    current_stage: 'factcheck_screen',
    stages: names.map((name, i) => ({
      name,
      status: i < 1 ? 'completed' : (i === 1 ? 'running' : 'pending'),
    })),
  };
  assert.equal(
    summarise(running),
    'Step 2 of 5 — Checking claims against the paper',
  );
  const done = {
    kind: 'podcast_factcheck',
    status: 'succeeded',
    stages: names.map((name) => ({ name, status: 'completed' })),
    result: { counts: { claims: 12, high: 2, medium: 1, low: 0 } },
  };
  assert.equal(summarise(done), 'Done — 12 claims, 2 high, 1 medium, 0 low');
  const failed = {
    kind: 'podcast_factcheck',
    status: 'failed',
    stages: names.map((name) => ({ name, status: 'completed' })),
    error: 'no stored paper text',
  };
  assert.match(summarise(failed), /Fact check failed/);
  assert.match(summarise(failed), /script was not changed/);
  assert.equal(statusColor(failed), 'run-danger');
});

test('map build stages have prose labels and the map is summarised', () => {
  const names = ['map_load', 'map_graph', 'map_communities', 'map_lineage',
                 'map_layout', 'map_suggest', 'map_persist'];
  for (const name of names) {
    assert.notEqual(stageLabel(name), name);
  }
  const running = {
    kind: 'map_build',
    status: 'running',
    current_stage: 'map_lineage',
    stages: names.map((name, i) => ({
      name,
      status: i < 3 ? 'completed' : (i === 3 ? 'running' : 'pending'),
    })),
  };
  assert.equal(summarise(running), 'Step 4 of 7 — Tracing the line of development');
  const done = {
    kind: 'map_build',
    status: 'succeeded',
    stages: names.map((name) => ({ name, status: 'completed' })),
    result: { pillar_id: 'p', stats: { works: 457, topics: 4, suggestions: 1 } },
  };
  assert.equal(summarise(done), 'Done — map saved: 457 works, 4 topics, 1 suggestion');
  const failed = {
    kind: 'map_build',
    status: 'failed',
    stages: names.map((name) => ({ name, status: 'completed' })),
    error: 'Apply docs/migrations/034_map_snapshots.sql by hand.',
  };
  assert.match(summarise(failed), /previous map is unchanged/);
  assert.match(summarise(failed), /034_map_snapshots/);
});

test('an unsaved podcast script is not painted as a clean success', () => {
  const run = {
    kind: 'podcast_script',
    status: 'succeeded',
    stages: podcastScriptStages(7),
    result: { saved: false, title: 'Deep Dive: X', word_count: 100, script: '[HOST]: hi' },
  };
  assert.match(summarise(run), /NOT saved/);
  assert.equal(statusColor(run), 'run-warn');
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
