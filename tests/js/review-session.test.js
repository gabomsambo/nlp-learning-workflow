/*
 * Pure helpers for the review session. The card renderer needs a DOM and is
 * not covered here; the same limit as paper-modal.test.js.
 *
 * Run with `node --test tests/js/*.test.js`.
 */
const { test } = require('node:test');
const assert = require('node:assert');

require('../../webui/static/paper-modal.js');
const {
  isTextField,
  shortcutCommand,
  registerShortcuts,
  progressLabel,
  labelDifficulty,
  summariseSession,
  formatNextReview,
  REVIEW_SHORTCUTS,
} = require('../../webui/static/review-session.js');

test('progressLabel is the n of m the session shows', () => {
  assert.equal(progressLabel(1, 12), '1 of 12');
  assert.equal(progressLabel(3, 12), '3 of 12');
});

test('labelDifficulty is paper-modal difficultyLabel, not a second map', () => {
  assert.equal(labelDifficulty(1), 'Easy');
  assert.equal(labelDifficulty(2), 'Medium');
  assert.equal(labelDifficulty(3), 'Hard');
  assert.equal(labelDifficulty('DifficultyLevel.EASY'), null);
  assert.equal(labelDifficulty(null), null);
});

test('shortcuts ignore text fields and modifier chords', () => {
  assert.equal(shortcutCommand({ key: ' ', target: { tagName: 'DIV' } }).type, 'reveal');
  assert.deepEqual(
    shortcutCommand({ key: '3', target: { tagName: 'BODY' } }),
    { type: 'rate', rating: 3 }
  );
  assert.equal(shortcutCommand({ key: ' ', target: { tagName: 'INPUT', type: 'text' } }), null);
  assert.equal(shortcutCommand({ key: '1', target: { tagName: 'TEXTAREA' } }), null);
  assert.equal(shortcutCommand({ key: '2', target: { tagName: 'SELECT' } }), null);
  assert.equal(shortcutCommand({ key: '4', target: { tagName: 'DIV', isContentEditable: true } }), null);
  assert.equal(shortcutCommand({ key: '1', metaKey: true, target: { tagName: 'DIV' } }), null);
  assert.equal(shortcutCommand({ key: 'k', target: { tagName: 'DIV' } }), null);
  // A button is not a text field: Space still means reveal, and the button remains clickable.
  assert.equal(shortcutCommand({ key: ' ', target: { tagName: 'BUTTON' } }).type, 'reveal');
  assert.equal(isTextField({ tagName: 'INPUT', type: 'checkbox' }), false);
});

test('Review registers with the shell overlay, or leaves the hook', () => {
  const previous = global.NlpShortcuts;
  const previousHook = global.NLPReviewShortcuts;
  try {
    delete global.NlpShortcuts;
    assert.equal(registerShortcuts(), 'hook');
    assert.equal(global.NLPReviewShortcuts, REVIEW_SHORTCUTS);
    assert.deepEqual(
      REVIEW_SHORTCUTS.map((item) => item.keys),
      ['Space', '1', '2', '3', '4']
    );

    const seen = [];
    global.NlpShortcuts = {
      register(entry) { seen.push(entry); }
    };
    let revealed = 0;
    let rated = null;
    assert.equal(registerShortcuts(null, {
      reveal() { revealed += 1; },
      rate(n) { rated = n; },
    }), 'registered');
    assert.deepEqual(seen.map((entry) => entry.id), [
      'review-reveal', 'review-again', 'review-hard', 'review-good', 'review-easy',
    ]);
    assert.equal(seen[0].group, 'Review');
    assert.equal(seen[0].keys, 'Space');
    assert.equal(seen[0].label, 'Reveal the answer');
    assert.equal(seen[0].match({ key: ' ' }), true);
    assert.equal(seen[0].match({ key: 'k' }), false);
    assert.equal(seen[3].match({ key: '3' }), true);
    seen[0].run();
    seen[3].run();
    assert.equal(revealed, 1);
    assert.equal(rated, 3);
  } finally {
    if (previous === undefined) delete global.NlpShortcuts;
    else global.NlpShortcuts = previous;
    if (previousHook === undefined) delete global.NLPReviewShortcuts;
    else global.NLPReviewShortcuts = previousHook;
  }
});

test('the end screen counts ratings and the soonest next review', () => {
  const summary = summariseSession([
    { rating: 3, nextReview: '2026-10-30T00:00:00Z' },
    { rating: 1, nextReview: '2026-10-23T13:00:00Z' },
    { rating: 3, nextReview: null },
  ]);
  assert.equal(summary.reviewed, 3);
  assert.deepEqual(summary.counts, { 1: 1, 2: 0, 3: 2, 4: 0 });
  assert.equal(summary.nextDue, '2026-10-23T13:00:00Z');
});

test('a missing next review stays the same word the old page used', () => {
  assert.equal(formatNextReview(null), 'Soon');
  assert.equal(formatNextReview('not-a-date'), 'Soon');
  assert.notEqual(formatNextReview('2026-10-24T09:00:00Z'), 'Soon');
});
