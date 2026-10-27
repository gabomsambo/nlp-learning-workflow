/*
 * Tests for the pure helpers in webui/static/playground.js.
 *
 * Run with `node --test tests/js/*.test.js`. DOM-free except buildTree, which
 * gets a minimal fake document so the allow-list and attribute rules can be
 * checked without jsdom. The renderer itself was exercised in Chrome.
 */
const { test } = require('node:test');
const assert = require('node:assert');

const {
  buildTree,
  lineDiff,
  groupCells,
  explanationsByCell,
  datasetSummary,
  statusText,
  explainLabel,
  safePng,
  markdownParts,
} = require('../../webui/static/playground.js');
const { summarise, stageLabel } = require('../../webui/static/run-progress.js');

function fakeDoc() {
  function node(tag) {
    return {
      tag: tag,
      attrs: {},
      children: [],
      appendChild(child) { this.children.push(child); return child; },
      setAttribute(name, value) { this.attrs[name] = value; },
    };
  }
  return {
    createElement: node,
    createDocumentFragment() { return node('#fragment'); },
    createTextNode(text) { return { text: text }; },
  };
}

function tags(n, out) {
  out = out || [];
  if (n.tag) out.push(n.tag);
  (n.children || []).forEach((c) => tags(c, out));
  return out;
}

function text(n) {
  if (n.text != null) return n.text;
  return (n.children || []).map(text).join('');
}

/* ---------------------------------------------------------------- buildTree */

test('buildTree builds allowed tags and unwraps the rest', () => {
  const frag = buildTree([
    { t: 'table', c: [{ t: 'tr', c: [{ t: 'td', a: { colspan: '2' }, c: ['1.5'] }] }] },
    { t: 'script', c: ['alert(1)'] },
    { t: 'a', a: { href: 'javascript:alert(1)' }, c: ['link text'] },
    'tail',
  ], fakeDoc());
  assert.deepEqual(tags(frag), ['#fragment', 'table', 'tr', 'td']);
  assert.equal(text(frag), '1.5alert(1)link texttail');
});

test('buildTree sets no attribute but a numeric colspan/rowspan on a cell', () => {
  const frag = buildTree([
    { t: 'td', a: { colspan: '3', rowspan: '2 onclick=x', onclick: 'x', style: 'a' }, c: [] },
    { t: 'div', a: { colspan: '2' }, c: [] },
  ], fakeDoc());
  assert.deepEqual(frag.children[0].attrs, { colspan: '3' });
  assert.deepEqual(frag.children[1].attrs, {});
});

test('buildTree ignores junk nodes', () => {
  const frag = buildTree([null, 42, { c: ['x'] }], fakeDoc());
  assert.equal(text(frag), 'x');
});

/* ---------------------------------------------------------------- lineDiff */

test('lineDiff marks the changed line and keeps the rest', () => {
  const diff = lineDiff('a\nb = 1\nc', 'a\nb = 2\nc');
  assert.deepEqual(diff, [
    { op: 'same', text: 'a' },
    { op: 'del', text: 'b = 1' },
    { op: 'add', text: 'b = 2' },
    { op: 'same', text: 'c' },
  ]);
});

test('lineDiff handles added and removed lines', () => {
  assert.deepEqual(lineDiff('a', 'a\nb').map((r) => r.op), ['same', 'add']);
  assert.deepEqual(lineDiff('a\nb', 'b').map((r) => r.op), ['del', 'same']);
});

/* --------------------------------------------------------------- groupCells */

const CELLS = [
  { index: 0, type: 'markdown', source: '# Title' },
  { index: 1, type: 'code', source: 'import x' },
  { index: 2, type: 'markdown', source: '## Fit' },
  { index: 3, type: 'code', source: 'fit()' },
  { index: 4, type: 'code', source: 'plot()' },
];

test('groupCells puts the introducing markdown with its step', () => {
  const steps = [
    { title: 'Load', cells: [{ cell_index: 1 }] },
    { title: 'Fit', cells: [{ cell_index: 3 }, { cell_index: 4 }] },
  ];
  const groups = groupCells(CELLS, steps);
  assert.deepEqual(groups.map((g) => g.cells.map((c) => c.index)), [[0, 1], [2, 3, 4]]);
  assert.equal(groups[1].step.title, 'Fit');
});

test('groupCells keeps unexplained cells visible in a final group', () => {
  const groups = groupCells(CELLS, [{ title: 'Load', cells: [{ cell_index: 1 }] }]);
  assert.equal(groups.length, 2);
  assert.equal(groups[1].step, null);
  assert.deepEqual(groups[1].cells.map((c) => c.index), [2, 3, 4]);
});

test('groupCells with no steps shows the whole notebook', () => {
  const groups = groupCells(CELLS, []);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].cells.length, 5);
});

test('explanationsByCell indexes by cell', () => {
  const map = explanationsByCell([{ cells: [{ cell_index: 3, explanation: 'Fits.' }] }]);
  assert.equal(map[3].explanation, 'Fits.');
});

/* ------------------------------------------------------------------ labels */

test('datasetSummary says how many datasets were saved', () => {
  assert.equal(datasetSummary([]), 'No dataset URLs in the code.');
  assert.equal(datasetSummary([{ url: 'a', bytes: 2500000 }]), '1 dataset saved for offline runs (2.5 MB).');
  assert.match(datasetSummary([{ url: 'a', bytes: 1 }, { url: 'b', error: 'HTTP 404' }]), /1 of 2 datasets saved .*1 could not be downloaded/);
});

test('statusText and explainLabel follow the stored status', () => {
  assert.equal(statusText(null, null), '');
  assert.equal(statusText({ status: 'imported', steps: [] }, null), 'Notebook imported. No study guide yet.');
  assert.equal(statusText({ status: 'ready', steps: [1, 2], model: 'm' }, null), '2 study steps · written by m');
  assert.equal(statusText({ status: 'ready', steps: [] }, { status: 'running' }), 'Writing the study guide…');
  assert.equal(explainLabel({ status: 'imported' }), 'Generate study guide');
  assert.equal(explainLabel({ status: 'ready' }), 'Regenerate study guide');
  assert.equal(explainLabel({ status: 'partial' }), 'Retry study guide');
  assert.equal(explainLabel({ status: 'failed' }), 'Retry study guide');
});

test('safePng only passes base64', () => {
  assert.equal(safePng('iVBORw0KGgoAAAANSUhEUg=='), 'iVBORw0KGgoAAAANSUhEUg==');
  assert.equal(safePng('x" onerror="alert(1)'), null);
  assert.equal(safePng(null), null);
});

/* ------------------------------------------------------------- run-progress */

test('run-progress labels the playground stages', () => {
  assert.equal(stageLabel('playground_explain'), 'Explaining the code cells');
  assert.equal(stageLabel('playground_persist'), 'Saving the study guide');
});

test('run-progress says what a playground run left behind', () => {
  const base = { kind: 'playground_explain', stages: [] };
  assert.equal(
    summarise(Object.assign({}, base, { status: 'succeeded', result: { cells_explained: 5, cells_total: 5 } })),
    'Done — study guide saved (5 of 5 cells)');
  assert.equal(
    summarise(Object.assign({}, base, { status: 'failed', error: 'gap', result: { status: 'partial', cells_explained: 3, cells_total: 5 } })),
    'Saved a partial study guide (3 of 5 cells) — gap');
  assert.equal(
    summarise(Object.assign({}, base, { status: 'failed', error: 'boom', result: null })),
    'Failed — the notebook is unchanged — boom');
});

test('markdownParts shows short text whole and folds long text under its heading', () => {
  assert.equal(markdownParts('   '), null);
  assert.deepEqual(markdownParts('## Fit\nA line.'), { head: '## Fit\nA line.', rest: '' });
  const long = '## Data has been preloaded\n' + 'word '.repeat(80);
  const parts = markdownParts(long);
  assert.equal(parts.head, 'Data has been preloaded');
  assert.ok(parts.rest.startsWith('word'));
});

test('setupCode runs what was saved, else the suggestion; an empty save runs nothing', () => {
  assert.strictEqual(Playground.setupCode(null), '');
  assert.strictEqual(Playground.setupCode({ setup_code: null, setup_suggested: 'import numpy as np\n' }), 'import numpy as np\n');
  assert.strictEqual(Playground.setupCode({ setup_code: 'import pandas as pd\n', setup_suggested: 'x' }), 'import pandas as pd\n');
  assert.strictEqual(Playground.setupCode({ setup_code: '', setup_suggested: 'import numpy as np\n' }), '');
});

test('setupSummary counts the lines that do something', () => {
  assert.strictEqual(Playground.setupSummary('import numpy as np\n\nimport pandas as pd\n'), 'Setup · 2 lines');
  assert.strictEqual(Playground.setupSummary('import numpy as np'), 'Setup · 1 line');
  assert.strictEqual(Playground.setupSummary(''), 'Setup · nothing to run');
  assert.strictEqual(Playground.setupRunText({ status: 'ok' }), 'Ran when this session started.');
  assert.strictEqual(Playground.setupRunText({ status: 'none' }), '');
});
