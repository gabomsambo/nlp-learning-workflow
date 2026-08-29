/*
 * Tests for the pure helpers in webui/static/paper-modal.js.
 *
 * Run with `node --test tests/js/*.test.js` — stdlib only, no npm, no bundler, same
 * arrangement as run-progress.test.js and for the same reason: the project has no build
 * step, so a CommonJS guard at the bottom of the module is the only way to reach any of
 * it from a test.
 *
 * Scope, honestly: this covers the DOM-free helpers only — the difficulty/author/
 * confidence formatting, the href scheme check, and the toggle's label. The rendering
 * itself, the fetch, the dialog wiring and the localStorage-backed answers toggle all
 * need a DOM and are NOT covered here; jsdom means npm and Playwright means a browser
 * download, both already rejected for this repo. Those were exercised by hand in Chrome
 * against the running app.
 */
const { test } = require('node:test');
const assert = require('node:assert');

const {
  difficultyLabel,
  authorsLine,
  confidencePercent,
  isSafeHref,
  toggleLabel,
} = require('../../webui/static/paper-modal.js');

/* --------------------------------------------------------- difficultyLabel */

test('difficultyLabel maps 1-3 to words', () => {
  assert.equal(difficultyLabel(1), 'Easy');
  assert.equal(difficultyLabel(2), 'Medium');
  assert.equal(difficultyLabel(3), 'Hard');
});

test('difficultyLabel returns null for anything out of range', () => {
  // The badge is omitted rather than rendered as "undefined", which is what the
  // template-literal version produced for a card with no difficulty.
  assert.equal(difficultyLabel(0), null);
  assert.equal(difficultyLabel(4), null);
  assert.equal(difficultyLabel(null), null);
  assert.equal(difficultyLabel(undefined), null);
  assert.equal(difficultyLabel('nonsense'), null);
});

test('difficultyLabel accepts numeric strings, as PostgREST can return them', () => {
  assert.equal(difficultyLabel('2'), 'Medium');
});

/* ------------------------------------------------------------- authorsLine */

test('authorsLine joins with commas', () => {
  assert.equal(authorsLine(['Ada', 'Grace']), 'Ada, Grace');
});

test('authorsLine drops blanks and non-strings', () => {
  assert.equal(authorsLine(['Ada', '', '   ', null, 42, 'Grace']), 'Ada, Grace');
});

test('authorsLine returns empty string for a missing list', () => {
  assert.equal(authorsLine(null), '');
  assert.equal(authorsLine(undefined), '');
  assert.equal(authorsLine('Ada'), '');
  assert.equal(authorsLine([]), '');
});

/* -------------------------------------------------------- confidencePercent */

test('confidencePercent rounds to a whole percent', () => {
  assert.equal(confidencePercent(0.844), '84%');
  assert.equal(confidencePercent(1), '100%');
  assert.equal(confidencePercent(0), '0%');
});

test('confidencePercent returns null when there is no score', () => {
  assert.equal(confidencePercent(null), null);
  assert.equal(confidencePercent(undefined), null);
  assert.equal(confidencePercent('0.5'), null);
  assert.equal(confidencePercent(NaN), null);
});

/* ---------------------------------------------------------------- isSafeHref */

test('isSafeHref allows the schemes papers actually use', () => {
  assert.ok(isSafeHref('https://arxiv.org/pdf/1706.03762'));
  assert.ok(isSafeHref('http://example.org/p.pdf'));
  // Uploaded PDFs are stored as file:///app/data/uploads/<hash>.pdf. The browser will
  // not follow it, but showing the path is honest and it cannot execute.
  assert.ok(isSafeHref('file:///app/data/uploads/abc.pdf'));
  assert.ok(isSafeHref('/papers/details/1706.03762'));
  assert.ok(isSafeHref('relative/path.pdf'));
});

test('isSafeHref rejects executable and data schemes', () => {
  assert.equal(isSafeHref('javascript:alert(1)'), false);
  assert.equal(isSafeHref('JavaScript:alert(1)'), false);
  assert.equal(isSafeHref('  javascript:alert(1)'), false);
  assert.equal(isSafeHref('data:text/html,<script>alert(1)</script>'), false);
  assert.equal(isSafeHref('vbscript:msgbox(1)'), false);
});

test('isSafeHref rejects empty and non-string input', () => {
  assert.equal(isSafeHref(''), false);
  assert.equal(isSafeHref('   '), false);
  assert.equal(isSafeHref(null), false);
  assert.equal(isSafeHref(undefined), false);
  assert.equal(isSafeHref(42), false);
});

/* --------------------------------------------------------------- toggleLabel */

test('toggleLabel names the action, not the current state', () => {
  // A button reading "Answers shown" leaves the user guessing what pressing it does.
  assert.equal(toggleLabel(true), 'Hide answers');
  assert.equal(toggleLabel(false), 'Show answers');
});
