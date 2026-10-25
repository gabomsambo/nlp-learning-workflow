// The pure Listen helpers that the desktop page and the cloud reader both load.
// scriptLines / resolveSpan / claimSegments are also pinned through listen.js in
// listen.test.js, which re-exports them; these are the ones that are new here.
const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const Core = require('../../webui/static/listen-core.js');
const RA = require('../../webui/static/readalong.js');
const Listen = require('../../webui/static/listen.js');

const LINE = '[HOST]: Training took 3.5 days on eight GPUs and beat every earlier system.';

test('listen.js re-exports the very functions listen-core.js defines', () => {
  for (const name of ['scriptLines', 'resolveSpan', 'visibleClaimSpans', 'claimSegments', 'formatClock', 'formatValueText']) {
    assert.strictEqual(Listen[name], Core[name], name);
  }
});

test('claimsByLine buckets by line and drops lines the script does not have', () => {
  const claims = [
    { id: 'a', line_index: 2 },
    { id: 'b', line_index: 0 },
    { id: 'c', line_index: 2 },
    { id: 'd', line_index: 9 },
    { id: 'e', line_index: -1 },
    { id: 'f', line_index: '1' },
    { id: 'g', line_index: 1.5 },
    { id: 'h' },
    null,
  ];
  const byLine = Core.claimsByLine(claims, 3);
  assert.deepStrictEqual(Object.keys(byLine).sort(), ['0', '1', '2']);
  assert.deepStrictEqual(byLine[2].map((c) => c.id), ['a', 'c']);
  assert.deepStrictEqual(byLine[1].map((c) => c.id), ['f']);
  assert.deepStrictEqual(Core.claimsByLine(undefined, 3), {});
});

test('groupTokens: an unflagged line is one group per word, indexed from zero', () => {
  const tokens = RA.tokenize(LINE);
  const groups = Core.groupTokens(tokens, Core.claimSegments(LINE, [], false));
  assert.strictEqual(groups.length, tokens.length);
  groups.forEach((group, i) => {
    assert.strictEqual(group.segment, null);
    assert.deepStrictEqual(group.words.map((w) => w.index), [i]);
    assert.strictEqual(group.words[0].token, tokens[i]);
  });
});

test('groupTokens: the words of one claim share a group, the rest stand alone', () => {
  const span = '3.5 days on eight GPUs';
  const start = LINE.indexOf(span);
  const claim = { verdict: 'contradicted', line_index: 0, span_start: start, span_end: start + span.length, span_text: span };
  const tokens = RA.tokenize(LINE);
  const groups = Core.groupTokens(tokens, Core.claimSegments(LINE, [claim], false));
  const flagged = groups.filter((g) => g.segment);
  assert.strictEqual(flagged.length, 1);
  assert.deepStrictEqual(flagged[0].words.map((w) => w.token.text), ['3.5', 'days', 'on', 'eight', 'GPUs']);
  assert.deepStrictEqual(flagged[0].words.map((w) => w.index), [2, 3, 4, 5, 6]);
  assert.deepStrictEqual(flagged[0].segment.claims, [claim]);
  // every word is in exactly one group, in order
  assert.deepStrictEqual(groups.flatMap((g) => g.words.map((w) => w.index)), tokens.map((_, i) => i));
});

test('groupTokens: two different claims side by side are two marks, not one', () => {
  const a = { verdict: 'unsupported', span_start: LINE.indexOf('Training'), span_end: LINE.indexOf('Training') + 'Training took'.length, span_text: 'Training took' };
  const b = { verdict: 'unsupported', span_start: LINE.indexOf('3.5'), span_end: LINE.indexOf('3.5') + 3, span_text: '3.5' };
  const tokens = RA.tokenize(LINE);
  const groups = Core.groupTokens(tokens, Core.claimSegments(LINE, [a, b], false));
  const flagged = groups.filter((g) => g.segment);
  assert.strictEqual(flagged.length, 2);
  assert.deepStrictEqual(flagged.map((g) => g.words.map((w) => w.token.text)), [['Training', 'took'], ['3.5']]);
});

test('groupTokens: a supported claim is hidden unless asked for', () => {
  const claim = { verdict: 'supported', span_start: LINE.indexOf('eight'), span_end: LINE.indexOf('eight') + 5, span_text: 'eight' };
  const tokens = RA.tokenize(LINE);
  assert.strictEqual(Core.groupTokens(tokens, Core.claimSegments(LINE, [claim], false)).filter((g) => g.segment).length, 0);
  assert.strictEqual(Core.groupTokens(tokens, Core.claimSegments(LINE, [claim], true)).filter((g) => g.segment).length, 1);
});

test('groupTokens tolerates nothing to group', () => {
  assert.deepStrictEqual(Core.groupTokens(null, null), []);
  assert.deepStrictEqual(Core.groupTokens([], []), []);
  const tokens = RA.tokenize('hello there');
  assert.strictEqual(Core.groupTokens(tokens, null).length, 2);
});

test('podcast.html loads listen-core.js after readalong.js and before listen.js', () => {
  const html = fs.readFileSync(path.join(__dirname, '../../webui/templates/podcast.html'), 'utf8');
  const read = html.indexOf('/static/readalong.js');
  const core = html.indexOf('/static/listen-core.js');
  const listen = html.indexOf('/static/listen.js');
  assert.ok(read > -1 && core > read && listen > core);
});
