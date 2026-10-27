/*
 * The pure logic behind the live Playground tab (webui/static/playground-core.js).
 * Run with `node --test tests/js/*.test.js`. No DOM.
 */
const { test } = require('node:test');
const assert = require('node:assert');

const Core = require('../../webui/static/playground-core.js');
const Playground = require('../../webui/static/playground.js');

test('playground.js re-exports the core line diff', () => {
  assert.strictEqual(Playground.lineDiff, Core.lineDiff);
});

/* ------------------------------------------------------------ code diffs */

test('compactDiff keeps changes with one line of context and counts the gaps', () => {
  const before = ['a', 'b', 'c', 'd', 'e', 'f', 'g'].join('\n');
  const after = ['a', 'b', 'c', 'X', 'e', 'f', 'g'].join('\n');
  const rows = Core.compactDiff(Core.lineDiff(before, after), 1);
  assert.deepStrictEqual(rows, [
    { op: 'gap', count: 2 },
    { op: 'same', text: 'c' },
    { op: 'del', text: 'd' },
    { op: 'add', text: 'X' },
    { op: 'same', text: 'e' },
    { op: 'gap', count: 2 },
  ]);
  assert.deepStrictEqual(Core.compactDiff(Core.lineDiff('same', 'same'), 1), []);
});

test('editSummary names what the edit did', () => {
  assert.strictEqual(Core.editSummary('a\nb', 'a\nb'), 'No changes from the original.');
  assert.strictEqual(Core.editSummary('a\nb', 'a\nc'), 'Your edit: 1 line added, 1 removed.');
  assert.strictEqual(Core.editSummary('a', 'a\nb\nc'), 'Your edit: 2 lines added.');
});

/* ------------------------------------------------------------- running */

test('planRunUpTo runs earlier cells not yet run, in order, then the target', () => {
  const ran = new Set([2, 7]);
  assert.deepStrictEqual(Core.planRunUpTo([2, 4, 7, 9, 12], 9, ran), [4, 9]);
  assert.deepStrictEqual(Core.planRunUpTo([2, 4, 7, 9, 12], 9, new Set()), [2, 4, 7, 9]);
  // The target always runs, even when it ran before.
  assert.deepStrictEqual(Core.planRunUpTo([2, 4], 4, new Set([2, 4])), [4]);
  assert.deepStrictEqual(Core.planRunUpTo([2, 4], 2, null), [2]);
});

test('formatElapsed reads like a stopwatch', () => {
  assert.strictEqual(Core.formatElapsed(0), '0.0 s');
  assert.strictEqual(Core.formatElapsed(1234), '1.2 s');
  assert.strictEqual(Core.formatElapsed(42000), '42 s');
  assert.strictEqual(Core.formatElapsed(125000), '2:05');
  assert.strictEqual(Core.formatElapsed(NaN), '0.0 s');
});

test('longerTimeout doubles up to the runner maximum, then stops offering', () => {
  assert.strictEqual(Core.longerTimeout(180, 900), 360);
  assert.strictEqual(Core.longerTimeout(600, 900), 900);
  assert.strictEqual(Core.longerTimeout(900, 900), null);
  assert.strictEqual(Core.longerTimeout(10, 900), 40);
  assert.strictEqual(Core.longerTimeout(0, 900), null);
  assert.strictEqual(Core.seconds(360), '6 min');
  assert.strictEqual(Core.seconds(90), '90 s');
});

test('runSummary says what happened, in the runner\'s own terms', () => {
  assert.deepStrictEqual(Core.runSummary({ status: 'ok', execution_ms: 1500, execution_count: 4 }, 180),
    { tone: 'ok', text: 'In [4] · Ran in 1.5 s.' });
  const timeout = Core.runSummary({ status: 'timeout', execution_ms: 180000 }, 180);
  assert.strictEqual(timeout.tone, 'warn');
  assert.match(timeout.text, /Stopped at the 3 min limit\. Variables from earlier cells are kept\./);
  assert.match(Core.runSummary({ status: 'error', interrupted: true, execution_ms: 3000 }).text, /^Interrupted/);
  const restarted = Core.runSummary({ status: 'error', kernel_restarted: true });
  assert.strictEqual(restarted.tone, 'error');
  assert.match(restarted.text, /every variable is gone/);
  assert.match(Core.runSummary({ status: 'error', execution_ms: 10 }).text, /Raised an error/);
  const cut = Core.runSummary({ status: 'ok', truncated: true, dropped_outputs: 2 });
  assert.match(cut.text, /Output cut at the runner’s size limit\. 2 later outputs were dropped\./);
});

test('errorMessage keeps the runner detail where it helps', () => {
  assert.match(Core.errorMessage(404), /session had ended/);
  assert.match(Core.errorMessage(409), /still running/);
  assert.strictEqual(Core.errorMessage(429, 'The runner already has 3 sessions open'), 'The runner already has 3 sessions open');
  assert.strictEqual(Core.errorMessage(503, 'not reachable. Start it with ./workstation up'), 'not reachable. Start it with ./workstation up');
  assert.match(Core.errorMessage(0), /Could not reach the app/);
});

test('sessionChip covers every phase, with the start hint when the runner is down', () => {
  const phases = ['idle', 'starting', 'ready', 'busy', 'stopped', 'unavailable', 'error'];
  phases.forEach((p) => {
    const chip = Core.sessionChip(p, '');
    assert.ok(chip.label, p);
    assert.ok(['neutral', 'accent', 'ok', 'warn', 'danger'].includes(chip.tone), p);
  });
  const down = Core.sessionChip('unavailable', 'The playground runner is not reachable. Start it with ./workstation up');
  assert.strictEqual(down.label, 'Runner not running');
  assert.match(down.hint, /workstation up/);
  assert.strictEqual(Core.sessionChip('busy', '2 waiting').hint, '2 waiting');
});

test('isRunShortcut is Shift+Enter and nothing else', () => {
  assert.ok(Core.isRunShortcut({ key: 'Enter', shiftKey: true }));
  assert.ok(!Core.isRunShortcut({ key: 'Enter' }));
  assert.ok(!Core.isRunShortcut({ key: 'Enter', shiftKey: true, ctrlKey: true }));
  assert.ok(!Core.isRunShortcut({ key: 'Enter', shiftKey: true, metaKey: true }));
  assert.ok(!Core.isRunShortcut({ key: 'a', shiftKey: true }));
  assert.ok(!Core.isRunShortcut(null));
});

/* ------------------------------------------------------- comparing outputs */

const TABLE = (a, b) => [{ t: 'table', c: [
  { t: 'thead', c: [{ t: 'tr', c: [{ t: 'th', c: [''] }, { t: 'th', c: ['score'] }] }] },
  { t: 'tbody', c: [
    { t: 'tr', c: [{ t: 'th', c: ['0'] }, { t: 'td', c: [a] }] },
    { t: 'tr', c: [{ t: 'th', c: ['1'] }, { t: 'td', c: [b] }] },
  ] },
] }];

test('tableChanges flags the cells that differ in same-shaped tables', () => {
  assert.deepStrictEqual(Core.tableCells(TABLE('0.1', '0.2')), [['', 'score'], ['0', '0.1'], ['1', '0.2']]);
  assert.deepStrictEqual(Core.tableChanges(TABLE('0.1', '0.2'), TABLE('0.1', '0.9')),
    [false, false, false, false, false, true]);
});

test('tableChanges gives up when the shapes differ', () => {
  const longer = [{ t: 'table', c: [{ t: 'tr', c: [{ t: 'td', c: ['1'] }, { t: 'td', c: ['2'] }, { t: 'td', c: ['3'] }] }] }];
  assert.strictEqual(Core.tableChanges(TABLE('1', '2'), longer), null);
  assert.strictEqual(Core.tableChanges([], []), null);
});

test('compareOutputs pairs by position and marks text lines or table cells', () => {
  const before = [
    { kind: 'stream', name: 'stdout', text: 'rows: 1000\ncols: 4' },
    { kind: 'html', tree: TABLE('0.1', '0.2') },
    { kind: 'png', data: 'x' },
  ];
  const after = [
    { kind: 'stream', name: 'stdout', text: 'rows: 10\ncols: 4' },
    { kind: 'html', tree: TABLE('0.1', '0.3') },
  ];
  const pairs = Core.compareOutputs(before, after);
  assert.strictEqual(pairs.length, 3);
  assert.deepStrictEqual(pairs[0].text, { before: [true, false], after: [true, false] });
  assert.deepStrictEqual(pairs[1].table, [false, false, false, false, false, true]);
  assert.strictEqual(pairs[2].after, null);
  assert.strictEqual(pairs[2].text, null);
  // A text output against a table is shown, not highlighted.
  const mixed = Core.compareOutputs([{ kind: 'text', text: 'a' }], [{ kind: 'html', tree: TABLE('1', '2') }]);
  assert.strictEqual(mixed[0].text, null);
  assert.strictEqual(mixed[0].table, null);
});

test('sameOutputs notices when a try changed nothing', () => {
  assert.ok(Core.sameOutputs([{ kind: 'text', text: '1' }], [{ kind: 'text', text: '1' }]));
  assert.ok(!Core.sameOutputs([{ kind: 'text', text: '1' }], [{ kind: 'text', text: '2' }]));
  assert.ok(!Core.sameOutputs([{ kind: 'text', text: '1' }], [{ kind: 'text', text: '1' }, { kind: 'text', text: '' }]));
});

test('outputsText flattens every kind for the explain-the-difference request', () => {
  const text = Core.outputsText([
    { kind: 'stream', text: 'hello' },
    { kind: 'html', tree: TABLE('0.1', '0.2') },
    { kind: 'html', plain: '   score\n0  0.1', tree: [] },
    { kind: 'png', alt: '<Figure size 640x480>' },
    { kind: 'unsupported', mime: ['application/x-widget'] },
  ]);
  assert.strictEqual(text, [
    'hello',
    'score\n0\t0.1\n1\t0.2',
    '   score\n0  0.1',
    '[figure: <Figure size 640x480>]',
    '[output of type application/x-widget]',
  ].join('\n'));
  assert.strictEqual(Core.outputsText([{ kind: 'text', text: 'x'.repeat(50) }], 10), 'xxxxxxxxxx\n[cut]');
});

test('treeText matches the server\'s tree_text for a table', () => {
  // nlp_pillars/playground/notebook.py::tree_text: cells tab-separated, rows on lines.
  assert.strictEqual(Core.treeText(TABLE('0.1', '0.2')), 'score\n0\t0.1\n1\t0.2');
});

/* ------------------------------------------------------------- the editor */

test('indentLines inserts four spaces at a caret', () => {
  assert.deepStrictEqual(Core.indentLines('ab', 1, 1, false), { value: 'a    b', start: 5, end: 5 });
});

test('indentLines indents and outdents every selected line', () => {
  const src = 'for x in y:\nprint(x)\nprint(y)';
  const start = src.indexOf('print(x)');
  const end = src.length;
  const ind = Core.indentLines(src, start, end, false);
  assert.strictEqual(ind.value, 'for x in y:\n    print(x)\n    print(y)');
  assert.strictEqual(ind.value.slice(ind.start, ind.end), 'print(x)\n    print(y)');
  const out = Core.indentLines(ind.value, ind.start, ind.end, true);
  assert.strictEqual(out.value, src);
  // Outdent at a caret on an unindented line changes nothing.
  assert.strictEqual(Core.indentLines('x = 1', 2, 2, true).value, 'x = 1');
});

test('indentLines leaves the line after a trailing newline alone', () => {
  const src = 'a\nb\nc';
  const r = Core.indentLines(src, 0, 2, false); // "a\n" selected
  assert.strictEqual(r.value, '    a\nb\nc');
});

test('newlineIndent keeps the indent and adds one after a colon', () => {
  assert.strictEqual(Core.newlineIndent('    x = 1', 9), '\n    ');
  assert.strictEqual(Core.newlineIndent('def f(a):', 9), '\n    ');
  assert.strictEqual(Core.newlineIndent('    if a:  # check', 18), '\n        ');
  assert.strictEqual(Core.newlineIndent('x = "a:b"', 9), '\n');
});

test('lineNumbers numbers every line, including a trailing empty one', () => {
  assert.strictEqual(Core.lineNumbers('a\nb\n'), '1\n2\n3');
  assert.strictEqual(Core.lineNumbers(''), '1');
});

test('setupFailure is the chip hint for a failed setup, and empty otherwise', () => {
  assert.strictEqual(Core.setupFailure(null), '');
  assert.strictEqual(Core.setupFailure({ status: 'ok' }), '');
  assert.strictEqual(Core.setupFailure({ status: 'none' }), '');
  assert.strictEqual(
    Core.setupFailure({ status: 'error', ename: 'ModuleNotFoundError', evalue: "No module named 'seaborn'" }),
    "Setup code failed (ModuleNotFoundError: No module named 'seaborn'). Fix it in Setup, then Restart.",
  );
});
