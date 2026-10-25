/*
 * Read-along's pure helpers: time -> line/word, click -> seek time, and how a
 * script line is split into words. DOM and audio wiring live in listen.js and
 * are exercised in the browser. Run with `node --test tests/js/*.test.js`.
 */
const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const RA = require('../../webui/static/readalong.js');

const SCRIPT = [
  '[HOST]: Welcome back to the show.',   // 0
  '',                                     // 1
  '[PAUSE]',                              // 2
  '[HOST]: It scored 3.5 points — a big jump.', // 3
  '[MUSIC]',                              // 4
  '[host]:   Thats all.',                 // 5
];

function timeline(over) {
  return Object.assign({
    version: 1,
    script_sha256: 'abc',
    duration_ms: 20000,
    lines: [
      { line: 0, start_ms: 1000, end_ms: 3000, words: [[1000, 1400, 1], [1400, 1900, 1], [1900, 2100, 1], [2100, 2300, 1], [2300, 3000, 1]] },
      { line: 3, start_ms: 4000, end_ms: 8000, words: [[4000, 4300, 1], [4300, 4700, 1], [4700, 5300, 0], [5300, 5800, 1], [5800, 5900, 1], [5900, 6000, 1], [6000, 6200, 1], [6200, 8000, 1]] },
      { line: 5, start_ms: 10000, end_ms: 11000, words: [[10000, 10500, 1], [10500, 11000, 1]] },
    ],
  }, over || {});
}

const model = () => RA.buildModel(timeline(), SCRIPT, 'abc');

test('tokenize splits what the narrator says, with offsets into the raw line', () => {
  const tokens = RA.tokenize('[HOST]: Welcome back.');
  assert.deepStrictEqual(tokens.map((t) => t.text), ['Welcome', 'back.']);
  assert.strictEqual('[HOST]: Welcome back.'.slice(tokens[0].start, tokens[0].end), 'Welcome');
  assert.strictEqual('[HOST]: Welcome back.'.slice(tokens[1].start, tokens[1].end), 'back.');
});

test('tokenize reads the host prefix case-insensitively and ignores extra spaces', () => {
  const line = '  [host]:   Thats all.';
  const tokens = RA.tokenize(line);
  assert.deepStrictEqual(tokens.map((t) => t.text), ['Thats', 'all.']);
  assert.strictEqual(line.slice(tokens[0].start, tokens[0].end), 'Thats');
});

test('tokenize says nothing for cues and blank lines, and narrates plain prose whole', () => {
  assert.strictEqual(RA.tokenize('[PAUSE]'), null);
  assert.strictEqual(RA.tokenize('[music]'), null);
  assert.strictEqual(RA.tokenize('   '), null);
  assert.strictEqual(RA.tokenize(''), null);
  assert.strictEqual(RA.tokenize('[HOST]:'), null);
  assert.deepStrictEqual(RA.tokenize('Plain narration here').map((t) => t.text), ['Plain', 'narration', 'here']);
});

test('timelineState tells current from stale from missing', () => {
  assert.strictEqual(RA.timelineState(null, 'abc'), 'missing');
  assert.strictEqual(RA.timelineState({}, 'abc'), 'missing');
  assert.strictEqual(RA.timelineState(timeline({ version: 2 }), 'abc'), 'missing');
  assert.strictEqual(RA.timelineState(timeline({ lines: [] }), 'abc'), 'missing');
  assert.strictEqual(RA.timelineState(timeline(), 'abc'), 'current');
  assert.strictEqual(RA.timelineState(timeline(), 'different'), 'stale');
  // No hash to compare is not evidence of staleness.
  assert.strictEqual(RA.timelineState(timeline(), ''), 'current');
});

test('buildModel orders lines and keeps per-word times where counts agree', () => {
  const m = model();
  assert.deepStrictEqual(m.lines.map((l) => l.line), [0, 3, 5]);
  assert.strictEqual(m.wordLevel, true);
  assert.strictEqual(m.lines[0].words.length, 5);
  assert.strictEqual(m.lines[1].words[2].heard, false);
  assert.strictEqual(m.byLine[3], 1);
});

test('buildModel drops word times for a line whose word count disagrees, keeping its line times', () => {
  const t = timeline();
  t.lines[1].words.pop();
  const m = RA.buildModel(t, SCRIPT, 'abc');
  assert.strictEqual(m.lines[1].words, null);
  assert.strictEqual(m.lines[1].start, 4000);
  assert.strictEqual(m.disagreements, 1);
  assert.ok(m.lines[0].words);
});

test('buildModel is line-level when the timeline carries no words', () => {
  const t = timeline();
  t.lines.forEach((l) => { delete l.words; });
  const m = RA.buildModel(t, SCRIPT, 'abc');
  assert.strictEqual(m.wordLevel, false);
  assert.deepStrictEqual(RA.positionAt(m, 4500), { line: 1, word: -1 });
});

test('buildModel refuses a stale or empty timeline and skips lines the script does not have', () => {
  assert.strictEqual(RA.buildModel(timeline(), SCRIPT, 'other'), null);
  assert.strictEqual(RA.buildModel(null, SCRIPT, 'abc'), null);
  const t = timeline();
  t.lines.push({ line: 99, start_ms: 12000, end_ms: 13000 });
  t.lines.push({ line: 2, start_ms: 13000, end_ms: 14000 });
  assert.deepStrictEqual(RA.buildModel(t, SCRIPT, 'abc').lines.map((l) => l.line), [0, 3, 5]);
});

test('lineIndexAt: before the first line, inside, between, and after the last', () => {
  const m = model();
  assert.strictEqual(RA.lineIndexAt(m, 0), -1);
  assert.strictEqual(RA.lineIndexAt(m, 999), -1);
  assert.strictEqual(RA.lineIndexAt(m, 1000), 0);
  assert.strictEqual(RA.lineIndexAt(m, 3500), 0); // the gap holds the last line
  assert.strictEqual(RA.lineIndexAt(m, 4000), 1);
  assert.strictEqual(RA.lineIndexAt(m, 99999), 2);
  assert.strictEqual(RA.lineIndexAt(null, 5), -1);
});

test('wordIndexAt follows the words and lets go after a long pause', () => {
  const m = model();
  const line = m.lines[0];
  assert.strictEqual(RA.wordIndexAt(line, 500), -1);
  assert.strictEqual(RA.wordIndexAt(line, 1000), 0);
  assert.strictEqual(RA.wordIndexAt(line, 1399), 0);
  assert.strictEqual(RA.wordIndexAt(line, 1400), 1);
  assert.strictEqual(RA.wordIndexAt(line, 2999), 4);
  // The last word holds through a short pause, then releases.
  assert.strictEqual(RA.wordIndexAt(line, 3000 + RA.WORD_HOLD_MS), 4);
  assert.strictEqual(RA.wordIndexAt(line, 3000 + RA.WORD_HOLD_MS + 1), -1);
});

test('positionAt gives line and word together', () => {
  const m = model();
  assert.deepStrictEqual(RA.positionAt(m, 4800), { line: 1, word: 2 });
  assert.deepStrictEqual(RA.positionAt(m, 10600), { line: 2, word: 1 });
  assert.deepStrictEqual(RA.positionAt(m, 100), { line: -1, word: -1 });
});

test('seek maps a clicked line or word back to a time, a touch early', () => {
  const m = model();
  assert.strictEqual(RA.seekSecondsForLine(m, 3), (4000 - RA.SEEK_LEAD_MS) / 1000);
  assert.strictEqual(RA.seekSecondsForWord(m, 3, 3), (5300 - RA.SEEK_LEAD_MS) / 1000);
  // A word without its own time falls back to its line.
  assert.strictEqual(RA.seekSecondsForWord(m, 3, 99), (4000 - RA.SEEK_LEAD_MS) / 1000);
  // Never before the start of the audio.
  const early = RA.buildModel(timeline({ lines: [{ line: 0, start_ms: 50, end_ms: 900 }] }), SCRIPT, 'abc');
  assert.strictEqual(RA.seekSecondsForLine(early, 0), 0);
  // A line the timeline does not time cannot be sought to.
  assert.strictEqual(RA.seekSecondsForLine(m, 2), null);
  assert.strictEqual(RA.seekSecondsForLine(null, 0), null);
});

test('seek and lookup agree: seeking to a word lands on that word', () => {
  const m = model();
  m.lines.forEach((line, lineIdx) => {
    line.words.forEach((word, wordIdx) => {
      const ms = Math.round(RA.seekSecondsForWord(m, line.line, wordIdx) * 1000) + RA.SEEK_LEAD_MS;
      assert.deepStrictEqual(RA.positionAt(m, ms), { line: lineIdx, word: wordIdx });
    });
  });
});

test('scrollTopFor keeps the line at the anchor and stays inside the scrollable range', () => {
  // A 400px-tall view over 2000px of content; the line is 100px tall at y=1000.
  assert.strictEqual(RA.scrollTopFor(1000, 100, 400, 2000, 0.4), 1000 - 160 + 50);
  assert.strictEqual(RA.scrollTopFor(10, 100, 400, 2000, 0.4), 0);
  assert.strictEqual(RA.scrollTopFor(1950, 100, 400, 2000, 0.4), 1600);
  // Content shorter than the view never scrolls.
  assert.strictEqual(RA.scrollTopFor(100, 50, 400, 300, 0.4), 0);
});

test('clockLabel', () => {
  assert.strictEqual(RA.clockLabel(0), '0:00');
  assert.strictEqual(RA.clockLabel(187000), '3:07');
  assert.strictEqual(RA.clockLabel(3729000), '1:02:09');
  assert.strictEqual(RA.clockLabel(-5), '0:00');
});

test('podcast.html loads readalong.js before listen.js', () => {
  const html = fs.readFileSync(path.join(__dirname, '../../webui/templates/podcast.html'), 'utf8');
  const read = html.indexOf('/static/readalong.js');
  const listen = html.indexOf('/static/listen.js');
  assert.ok(read > -1 && listen > -1 && read < listen);
});

test('every element listen.js looks up for read-along exists in podcast.html', () => {
  const root = path.join(__dirname, '../../webui');
  const html = fs.readFileSync(path.join(root, 'templates/podcast.html'), 'utf8');
  const js = fs.readFileSync(path.join(root, 'static/listen.js'), 'utf8');
  const ids = new Set();
  for (const m of js.matchAll(/getElementById\('((?:readalong|view-|script-view)[a-z-]*)'\)/g)) ids.add(m[1]);
  assert.ok(ids.size >= 10, 'found the read-along lookups');
  for (const id of ids) assert.ok(html.includes('id="' + id + '"'), id + ' is missing from the template');
});

test('the read-along stage has a label in the run panel', () => {
  const js = fs.readFileSync(path.join(__dirname, '../../webui/static/run-progress.js'), 'utf8');
  assert.match(js, /tts_align:\s*'/);
});
