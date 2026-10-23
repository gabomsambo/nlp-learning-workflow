/*
 * Pure helpers for the library list. DOM filtering and j/k are exercised in
 * the browser; these lock the match rules the page depends on.
 */
const { test } = require('node:test');
const assert = require('node:assert');

const { matchesRow, nextIndex, countLabel } = require('../../webui/static/library.js');

const row = {
  pillar: 'ml-systems-production',
  search: 'graph feature ada lovelace arxiv:2402.08593',
  lesson: true,
  cards: false,
  episode: true,
};

test('matchesRow filters by query, pillar and status together', () => {
  assert.equal(matchesRow(row, '', '', ''), true);
  assert.equal(matchesRow(row, 'lovelace', '', ''), true);
  assert.equal(matchesRow(row, 'missing', '', ''), false);
  assert.equal(matchesRow(row, '', 'ml-systems-production', ''), true);
  assert.equal(matchesRow(row, '', 'ai-safety-alignment', ''), false);
  assert.equal(matchesRow(row, '', '', 'lesson'), true);
  assert.equal(matchesRow(row, '', '', 'cards'), false);
  assert.equal(matchesRow(row, '', '', 'episode'), true);
  assert.equal(matchesRow(row, 'graph', 'ml-systems-production', 'episode'), true);
});

test('nextIndex wraps and starts from either end', () => {
  assert.equal(nextIndex(-1, 1, 3), 0);
  assert.equal(nextIndex(-1, -1, 3), 2);
  assert.equal(nextIndex(2, 1, 3), 0);
  assert.equal(nextIndex(0, -1, 3), 2);
  assert.equal(nextIndex(0, 1, 0), -1);
});

test('countLabel never says zero', () => {
  assert.equal(countLabel(0), '');
  assert.equal(countLabel(1), '1 paper');
  assert.equal(countLabel(2), '2 papers');
});
