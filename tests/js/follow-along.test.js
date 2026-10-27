/*
 * Follow-along's pure part: time -> script line -> span -> card, and what the
 * panel offers for an episode. The panel's DOM (mounting, the drawer, the
 * poller) needs a browser and was exercised in Chrome.
 * Run with `node --test tests/js/*.test.js`.
 */
const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const RA = require('../../webui/static/readalong.js');
const FA = require('../../webui/static/follow-along.js');
const { summarise, stageLabel } = require('../../webui/static/run-progress.js');

// A synthetic episode: intro, two code passages with a gap between, an outro.
const SCRIPT = [
  '[HOST]: Welcome back, today we read a small notebook.', // 0 intro, no code
  '[HOST]: First it loads the points file.',              // 1 cell 2
  '[HOST]: It prints the first five rows.',               // 2 cell 2
  '[PAUSE]',                                              // 3 silent
  '[HOST]: Why does that matter for the interview?',      // 4 gap, no code
  '[HOST]: Then it fits a straight line.',                // 5 cell 4
  '[HOST]: Thanks for listening.',                        // 6 outro
];

function words(n, start, step) {
  const out = [];
  for (let i = 0; i < n; i++) out.push([start + i * step, start + (i + 1) * step, 1]);
  return out;
}

function timeline(over) {
  const lines = [0, 1, 2, 4, 5, 6].map((line, i) => {
    const start = 1000 + i * 4000;
    const count = RA.tokenize(SCRIPT[line]).length;
    return { line, start_ms: start, end_ms: start + 3000, words: words(count, start, 3000 / count) };
  });
  return Object.assign({ version: 1, script_sha256: 'sha-1', duration_ms: 26000, lines }, over || {});
}

// Line starts: 0 -> 1000, 1 -> 5000, 2 -> 9000, 4 -> 13000, 5 -> 17000, 6 -> 21000.
const model = () => RA.buildModel(timeline(), SCRIPT, 'sha-1');

const STEPS = [
  { id: 'step-1', title: 'Load the points', goal: 'Read them.', cells: [{ cell_index: 2, explanation: 'Reads the points.' }] },
  { id: 'step-2', title: 'Fit a line', goal: 'Fit it.', cells: [{ cell_index: 4, explanation: 'Fits a line.' }, { cell_index: 5, explanation: 'Plots it.' }] },
];

const PLAYGROUND = {
  id: 'pg-1',
  paper_id: 'arxiv:1706.03762',
  steps: STEPS,
  cells: [
    { index: 0, type: 'markdown', source: '# Toy' },
    { index: 2, type: 'code', source: 'df = load()\ndf.head()', execution_count: 1, outputs: [] },
    { index: 4, type: 'code', source: 'fit(df)', execution_count: 2, outputs: [] },
    { index: 5, type: 'code', source: 'plot()', execution_count: 3, outputs: [] },
  ],
};

const MAPPING = [
  { line_start: 5, line_end: 5, cell_index: 4, step_id: 'step-2' },
  { line_start: 1, line_end: 2, cell_index: 2, step_id: 'step-1' },
];

const spans = () => FA.normalizeSpans(MAPPING, STEPS);

test('spans are sorted by line and checked against the study guide as it is now', () => {
  assert.deepStrictEqual(spans().map((s) => s.cell_index), [2, 4]);
  const checked = FA.normalizeSpans([
    ...MAPPING,
    { line_start: 6, line_end: 6, cell_index: 99, step_id: 'step-2' },   // cell gone
    { line_start: 6, line_end: 6, cell_index: 5, step_id: 'step-1' },    // wrong step
    { line_start: 2, line_end: 3, cell_index: 5, step_id: 'step-2' },    // overlaps 1-2
    { line_start: 7, line_end: 6, cell_index: 5, step_id: 'step-2' },    // backwards
    { line_start: '6', line_end: 6, cell_index: 5, step_id: 'step-2' },  // not an int
    null,
  ], STEPS);
  assert.deepStrictEqual(checked, spans());
  assert.deepStrictEqual(FA.normalizeSpans('nope', STEPS), []);
  assert.deepStrictEqual(FA.normalizeSpans(MAPPING, []), []);
});

test('spanIndexAt: before the first span, inside, in a gap, after the last', () => {
  const s = spans();
  assert.deepStrictEqual(FA.spanIndexAt(s, 0), { index: -1, inside: false });
  assert.deepStrictEqual(FA.spanIndexAt(s, 1), { index: 0, inside: true });
  assert.deepStrictEqual(FA.spanIndexAt(s, 2), { index: 0, inside: true });
  assert.deepStrictEqual(FA.spanIndexAt(s, 4), { index: 0, inside: false });
  assert.deepStrictEqual(FA.spanIndexAt(s, 5), { index: 1, inside: true });
  assert.deepStrictEqual(FA.spanIndexAt(s, 6), { index: 1, inside: false });
  assert.deepStrictEqual(FA.spanIndexAt([], 3), { index: -1, inside: false });
});

test('time -> line -> span -> card while playing forwards', () => {
  const m = model();
  const s = spans();
  const at = (ms) => FA.followAt(m, s, ms);
  assert.deepStrictEqual(at(0), { line: -1, index: -1, inside: false });        // before any speech
  assert.deepStrictEqual(at(2000), { line: 0, index: -1, inside: false });      // intro: no card yet
  assert.deepStrictEqual(at(6000), { line: 1, index: 0, inside: true });
  assert.deepStrictEqual(at(10000), { line: 2, index: 0, inside: true });
  assert.deepStrictEqual(at(14000), { line: 4, index: 0, inside: false });      // gap: last card, dimmed
  assert.deepStrictEqual(at(18000), { line: 5, index: 1, inside: true });
  assert.deepStrictEqual(at(25000), { line: 6, index: 1, inside: false });      // outro keeps the last card

  const card = FA.cardFor(s[at(18000).index], PLAYGROUND);
  assert.strictEqual(card.number, 2);
  assert.strictEqual(card.step.title, 'Fit a line');
  assert.strictEqual(card.guide.explanation, 'Fits a line.');
  assert.strictEqual(card.cell.source, 'fit(df)');
});

test('seeking backwards lands on the same answer as playing there', () => {
  const m = model();
  const s = spans();
  const forward = [25000, 18000, 14000, 10000, 6000, 2000, 0].map((ms) => FA.followAt(m, s, ms));
  const replay = [25000, 18000, 14000, 10000, 6000, 2000, 0].reverse().map((ms) => FA.followAt(m, s, ms)).reverse();
  assert.deepStrictEqual(forward, replay);
  // Seeking back from the second span into the gap shows the first card, dimmed.
  assert.deepStrictEqual(FA.followAt(m, s, 14500), { line: 4, index: 0, inside: false });
  // And back before the first span shows none.
  assert.strictEqual(FA.followAt(m, s, 3000).index, -1);
});

test('no model, no spans, or a silent cue line never produce a card out of nowhere', () => {
  assert.strictEqual(FA.followAt(null, spans(), 6000).index, -1);
  assert.strictEqual(FA.followAt(model(), [], 6000).index, -1);
  // Line 3 is a [PAUSE] with no timing; 12500 is still on line 2's slot.
  assert.strictEqual(FA.followAt(model(), spans(), 12500).line, 2);
});

test('cardFor refuses a cell that is gone or is not code', () => {
  assert.strictEqual(FA.cardFor({ cell_index: 9, step_id: 'step-1' }, PLAYGROUND), null);
  const markdownStep = { ...PLAYGROUND, steps: [{ id: 'step-1', cells: [{ cell_index: 0 }] }] };
  assert.strictEqual(FA.cardFor({ cell_index: 0, step_id: 'step-1' }, markdownStep), null);
  assert.strictEqual(FA.cardFor(null, PLAYGROUND), null);
});

test('the card opens that cell on the paper page Playground tab', () => {
  assert.strictEqual(FA.cellHref('arxiv:1706.03762', 4), '/papers/arxiv%3A1706.03762?cell=4#playground');
  assert.strictEqual(FA.cellHref('', 4), null);
});

const READY = { playground_id: 'pg-1', status: 'ready', script_sha256: 'sha-1', mapping: MAPPING };

test('an episode whose timeline is missing offers no follow-along and says why', () => {
  const out = FA.availability({ hasAudio: true, timeline: null, scriptSha: 'sha-1', links: [READY] });
  assert.strictEqual(out.mode, 'off');
  assert.match(out.reason, /no read-along timing/);
  assert.match(out.reason, /timeline backfill/);
  const bad = FA.availability({ hasAudio: true, timeline: { version: 2, lines: [] }, scriptSha: 'sha-1', links: [READY] });
  assert.strictEqual(bad.mode, 'off');
});

test('no audio, or a timeline for an older script, also offers none', () => {
  assert.strictEqual(FA.availability({ hasAudio: false, timeline: timeline(), scriptSha: 'sha-1' }).mode, 'off');
  const stale = FA.availability({ hasAudio: true, timeline: timeline(), scriptSha: 'sha-2', links: [READY] });
  assert.strictEqual(stale.mode, 'off');
  assert.match(stale.reason, /edited after this audio was recorded/);
});

test('a ready link follows; failed, stale or absent links offer the picker', () => {
  const base = { hasAudio: true, timeline: timeline(), scriptSha: 'sha-1' };
  assert.strictEqual(FA.availability({ ...base, links: [READY] }).mode, 'ready');
  assert.deepStrictEqual(FA.availability({ ...base, links: [] }), { mode: 'pick', link: null });
  const failed = FA.availability({ ...base, links: [{ playground_id: 'pg-2', status: 'failed', error: 'Overloaded.' }] });
  assert.strictEqual(failed.mode, 'pick');
  assert.match(failed.reason, /last match failed: Overloaded\. Try again/);
  const changed = FA.availability({
    ...base, scriptSha: 'sha-1', links: [{ ...READY, script_sha256: 'sha-0' }],
  });
  assert.strictEqual(changed.mode, 'pick');
  assert.match(changed.reason, /script changed/);
});

test('chooseLink prefers the one asked for, then a ready one, then the newest', () => {
  const failed = { playground_id: 'pg-2', status: 'failed' };
  assert.strictEqual(FA.chooseLink([failed, READY], 'sha-1').link, READY);
  assert.strictEqual(FA.chooseLink([failed, READY], 'sha-1', 'pg-2').link, failed);
  assert.strictEqual(FA.chooseLink([failed], 'sha-1').link, failed);
  assert.deepStrictEqual(FA.chooseLink(null, 'sha-1'), { link: null, stale: false });
});

test('the picker defaults to the episode paper and filters by paper or notebook', () => {
  const rows = [
    { id: 'a', paper_id: 'arxiv:1', paper_title: 'Isolation Forest study guide', title: 'aml.ipynb' },
    { id: 'b', paper_id: 'arxiv:2', paper_title: 'Attention', title: 'Toy regression', status: 'imported' },
  ];
  assert.strictEqual(FA.defaultPlaygroundId({ id: 'b' }, rows), 'b');
  assert.strictEqual(FA.defaultPlaygroundId(null, rows), 'a');
  assert.strictEqual(FA.defaultPlaygroundId(null, []), null);
  assert.deepStrictEqual(FA.filterPlaygrounds(rows, 'forest').map((r) => r.id), ['a']);
  assert.deepStrictEqual(FA.filterPlaygrounds(rows, 'TOY').map((r) => r.id), ['b']);
  assert.deepStrictEqual(FA.filterPlaygrounds(rows, 'arxiv:2').map((r) => r.id), ['b']);
  assert.strictEqual(FA.filterPlaygrounds(rows, '').length, 2);
  assert.match(FA.pickerLabel(rows[1]), /no study guide yet/);
});

test('the run panel names the alignment stages and says what a pass did', () => {
  assert.strictEqual(stageLabel('playground_align'), 'Matching script lines to code cells');
  assert.notStrictEqual(stageLabel('playground_align_prepare'), 'playground_align_prepare');
  assert.notStrictEqual(stageLabel('playground_align_persist'), 'playground_align_persist');
  assert.match(summarise({ kind: 'playground_align', status: 'succeeded', result: { spans: 3 } }), /3 code cards/);
  assert.match(summarise({ kind: 'playground_align', status: 'succeeded', result: { spans: 0 } }), /no line/);
  assert.match(summarise({ kind: 'playground_align', status: 'failed', error: 'boom' }), /boom/);
});

test('podcast.html loads the card renderer and follow-along before listen.js, with every element it looks up', () => {
  const root = path.join(__dirname, '../../webui');
  const html = fs.readFileSync(path.join(root, 'templates/podcast.html'), 'utf8');
  const order = ['/static/readalong.js', '/static/listen-core.js', '/static/playground.js', '/static/follow-along.js', '/static/listen.js']
    .map((src) => html.indexOf(src));
  assert.ok(order.every((i) => i > -1), 'all scripts are loaded');
  assert.deepStrictEqual(order.slice().sort((a, b) => a - b), order);
  const js = fs.readFileSync(path.join(root, 'static/follow-along.js'), 'utf8');
  const ids = new Set();
  for (const m of js.matchAll(/(?:querySelector|getElementById)\('#?(follow-[a-z-]+)'\)/g)) ids.add(m[1]);
  assert.ok(ids.size >= 12, 'found the panel lookups');
  // The run card's ids are run_panel() macro arguments, so they appear quoted.
  for (const id of ids) {
    assert.ok(html.includes('id="' + id + '"') || html.includes("_id='" + id + "'"), id + ' is missing from the template');
  }
});

test('the card reuses the Playground tab renderer instead of a copy', () => {
  const PG = require('../../webui/static/playground.js');
  assert.strictEqual(typeof PG.renderCode, 'function');
  assert.strictEqual(typeof PG.renderOutput, 'function');
  const js = fs.readFileSync(path.join(__dirname, '../../webui/static/follow-along.js'), 'utf8');
  assert.match(js, /PG\.renderCode\(/);
  assert.match(js, /PG\.renderOutput\(/);
  assert.doesNotMatch(js, /innerHTML/);
  assert.strictEqual(PG.requestedCell('?cell=12'), 12);
  assert.strictEqual(PG.requestedCell('?cell=x'), null);
  assert.strictEqual(PG.requestedCell(''), null);
});
