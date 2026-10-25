/*
 * Pure helpers for the shell palette, shortcut registry and mini player.
 * DOM wiring is exercised in the browser. Run with `node --test tests/js/*.test.js`.
 */
const { test } = require('node:test');
const assert = require('node:assert');

const {
  NAV_COMMANDS,
  isTextField,
  authorText,
  paperHref,
  filterPalette,
  formatClock,
  formatValueText,
  clampSeek,
  isSafeAudioSrc,
  toStoredSrc,
  parsePlayerState,
  createRegistry,
  groupShortcuts,
  matchSearch,
  matchHelp,
  resolveShortcut,
  moveIndex,
} = require('../../webui/static/shell.js');

const PAPERS = [
  { id: '1706.03762', title: 'Attention Is All You Need', authors: ['Ashish Vaswani', 'Noam Shazeer'], year: 2017 },
  { id: '1810.04805', title: 'BERT: Pre-training of Deep Bidirectional Transformers', authors: ['Jacob Devlin'], year: 2019 },
  { id: '2005.14165', title: 'Language Models are Few-Shot Learners', authors: ['Tom B. Brown'], year: 2020 },
];

function field(tag, type, editable) {
  return { tagName: tag, type: type, isContentEditable: !!editable };
}

test('Activity in the command palette opens the activity page', () => {
  const activity = NAV_COMMANDS.find((command) => command.id === 'activity');
  assert.equal(activity.href, '/activity/');
});

test('isTextField accepts text-like controls and ignores buttons', () => {
  assert.equal(isTextField(field('INPUT', 'text')), true);
  assert.equal(isTextField(field('INPUT', 'search')), true);
  assert.equal(isTextField(field('TEXTAREA')), true);
  assert.equal(isTextField(field('SELECT')), true);
  assert.equal(isTextField(field('DIV', '', true)), true);
  assert.equal(isTextField(field('INPUT', 'checkbox')), false);
  assert.equal(isTextField(field('INPUT', 'range')), false);
  assert.equal(isTextField(field('BUTTON')), false);
  assert.equal(isTextField(null), false);
});

test('authorText joins names and ignores junk', () => {
  assert.equal(authorText(['Ashish Vaswani', '', null, 'Noam Shazeer']), 'Ashish Vaswani, Noam Shazeer');
  assert.equal(authorText('Ada Lovelace'), 'Ada Lovelace');
  assert.equal(authorText(null), '');
});

test('paperHref stays on the library page', () => {
  assert.equal(paperHref('1706.03762'), '/papers/?paper=1706.03762');
  assert.equal(paperHref('a/b c'), '/papers/?paper=a%2Fb%20c');
  assert.equal(paperHref(''), null);
  assert.equal(paperHref(null), null);
});

test('filterPalette lists commands when the query is empty and papers when it is not', () => {
  const empty = filterPalette('', PAPERS);
  assert.equal(empty.papers.length, 0);
  assert.equal(empty.commands.length, NAV_COMMANDS.length);
  assert.ok(empty.commands.some((command) => command.label === 'Settings'));
  assert.ok(empty.commands.some((command) => command.label === 'Add papers'));

  const byAuthor = filterPalette('vaswani', PAPERS);
  assert.deepEqual(byAuthor.papers.map((paper) => paper.id), ['1706.03762']);

  const byTitle = filterPalette('few-shot', PAPERS);
  assert.deepEqual(byTitle.papers.map((paper) => paper.id), ['2005.14165']);

  const both = filterPalette('attention vaswani', PAPERS);
  assert.deepEqual(both.papers.map((paper) => paper.id), ['1706.03762']);

  const library = filterPalette('library', PAPERS);
  assert.equal(library.papers.length, 0);
  assert.deepEqual(library.commands.map((command) => command.label), ['Library']);

  const none = filterPalette('zzzz-not-a-paper', PAPERS);
  assert.equal(none.papers.length, 0);
  assert.equal(none.commands.length, 0);
});

test('title matches sort ahead of author-only matches', () => {
  const rows = [
    { id: 'author-only', title: 'Unrelated note', authors: ['Attention Team'] },
    { id: 'title-hit', title: 'Attention matters', authors: ['Someone'] },
  ];
  const found = filterPalette('attention', rows);
  assert.deepEqual(found.papers.map((paper) => paper.id), ['title-hit', 'author-only']);
});

test('clock and scrubber text match the player contract', () => {
  assert.equal(formatClock(0), '0:00');
  assert.equal(formatClock(65), '1:05');
  assert.equal(formatClock(3661), '1:01:01');
  assert.equal(formatClock(Number.NaN), '0:00');
  assert.equal(formatValueText(192, 778), '3 minutes 12 of 12 minutes 58');
  assert.equal(clampSeek(10, -15, 100), 0);
  assert.equal(clampSeek(10, 30, 100), 40);
  assert.equal(clampSeek(90, 30, 100), 100);
  assert.equal(clampSeek(10, 30, Number.NaN), 40);
});

test('player state keeps a safe audio path and refuses anything else', () => {
  assert.equal(parsePlayerState(null), null);
  assert.equal(parsePlayerState('not-json'), null);
  assert.equal(parsePlayerState('{}'), null);
  assert.equal(parsePlayerState(JSON.stringify({ src: 'javascript:alert(1)', time: 1, rate: 1 })), null);
  assert.equal(isSafeAudioSrc('/api/podcast/audio/../secret.mp3'), false);
  assert.equal(
    toStoredSrc('http://127.0.0.1:18810/api/podcast/audio/attention.mp3'),
    '/api/podcast/audio/attention.mp3',
  );

  const state = parsePlayerState(JSON.stringify({
    script_id: 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    src: 'http://127.0.0.1:18810/api/podcast/audio/attention.mp3',
    time: -4,
    rate: 1.25,
    title: '  Attention,   explained  ',
    href: 'javascript:alert(1)',
  }));
  assert.equal(state.src, '/api/podcast/audio/attention.mp3');
  assert.equal(state.time, 0);
  assert.equal(state.rate, 1.25);
  assert.equal(state.title, 'Attention, explained');
  assert.equal(state.href, '/podcast/?script=a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11');

  const oddRate = parsePlayerState(JSON.stringify({
    src: '/api/podcast/audio/attention.mp3',
    time: 12,
    rate: 9,
  }));
  assert.equal(oddRate.rate, 1);
  assert.equal(oddRate.href, '/podcast/');
});

test('shortcut registry is replaceable and groups Global first', () => {
  const registry = createRegistry();
  registry.register({ id: 'review-space', group: 'Review', label: 'Reveal the answer', keys: 'Space' });
  registry.register({ id: 'search', group: 'Global', label: 'Search', keys: 'Ctrl+K' });
  registry.register({ id: 'review-space', group: 'Review', label: 'Reveal', keys: 'Space' });
  assert.equal(registry.list().length, 2);
  assert.equal(registry.list().filter((entry) => entry.id === 'review-space')[0].label, 'Reveal');
  const groups = groupShortcuts(registry.list());
  assert.deepEqual(groups.map((group) => group.name), ['Global', 'Review']);
  registry.unregister('search');
  assert.deepEqual(registry.list().map((entry) => entry.id), ['review-space']);
  assert.throws(() => registry.register({ label: 'x', keys: 'x' }), /id required/);
});

test('Cmd-K works in a text field and letter chords do not', () => {
  const entries = [
    { id: 'search', allowInField: true, match: matchSearch },
    { id: 'help', match: matchHelp },
    { id: 'go-today', sequence: ['g', 't'] },
  ];
  const input = field('INPUT', 'text');
  const search = resolveShortcut(entries, null, {
    key: 'k', metaKey: true, ctrlKey: false, altKey: false, shiftKey: false, target: input,
  }, 1000);
  assert.equal(search.firedId, 'search');
  assert.equal(search.pending, null);

  const ctrl = resolveShortcut(entries, null, {
    key: 'K', metaKey: false, ctrlKey: true, altKey: false, shiftKey: false, target: null,
  }, 1000);
  assert.equal(ctrl.firedId, 'search');

  const typed = resolveShortcut(entries, null, {
    key: 'g', target: input,
  }, 1000);
  assert.equal(typed.firedId, null);
  assert.equal(typed.pending, null);

  const armed = resolveShortcut(entries, null, { key: 'g', target: field('BUTTON') }, 1000);
  assert.equal(armed.pending.key, 'g');
  const fired = resolveShortcut(entries, armed.pending, { key: 't', target: field('BODY') }, 1100);
  assert.equal(fired.firedId, 'go-today');

  const late = resolveShortcut(entries, { key: 'g', at: 0 }, { key: 't', target: null }, 5000);
  assert.equal(late.firedId, null);

  const helpInDialog = resolveShortcut(entries, null, {
    key: '?', target: field('BUTTON'), dialogOpen: true,
  }, 1000);
  assert.equal(helpInDialog.firedId, 'help');

  const chordInDialog = resolveShortcut(entries, { key: 'g', at: 1000 }, {
    key: 't', target: field('BUTTON'), dialogOpen: true,
  }, 1100);
  assert.equal(chordInDialog.firedId, null);
  assert.equal(matchHelp({ key: '?', metaKey: true }), false);
});

test('moveIndex wraps the palette selection', () => {
  assert.equal(moveIndex(0, -1, 3), 2);
  assert.equal(moveIndex(2, 1, 3), 0);
  assert.equal(moveIndex(-1, 1, 3), 0);
  assert.equal(moveIndex(0, 1, 0), -1);
});
