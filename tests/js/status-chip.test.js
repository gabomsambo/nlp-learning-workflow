/*
 * Shell status chip helpers. The fetch loop is not covered here.
 */
const { test } = require('node:test');
const assert = require('node:assert');

const { chipLabel, chipState, shouldRefresh } = require('../../webui/static/status-chip.js');

test('the chip counts services that are not ok', () => {
  assert.equal(chipLabel(null), 'Status');
  assert.equal(chipLabel({ checks: [{ state: 'ok' }, { state: 'ok' }] }), 'All up');
  assert.equal(chipLabel({ checks: [{ state: 'ok' }, { state: 'down' }] }), '1 down');
  assert.equal(chipLabel({ checks: [{ state: 'down' }, { state: 'down' }] }), '2 down');
  assert.equal(chipState({ checks: [{ state: 'ok' }] }), 'ok');
  assert.equal(chipState({ checks: [{ state: 'down' }] }), 'down');
});

test('a fresh cache is kept and a stale one is fetched again', () => {
  const now = 1_000_000;
  assert.equal(shouldRefresh(now - 10_000, now, 60_000), false);
  assert.equal(shouldRefresh(now - 60_000, now, 60_000), true);
  assert.equal(shouldRefresh(null, now, 60_000), true);
});
