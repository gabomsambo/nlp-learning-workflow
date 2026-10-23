/*
 * Pure helpers for the Today page. The polling loop and the DOM ticks need a
 * browser; these tests lock the calendar, the 90% listen mark, and the
 * status-chip rule that a missing endpoint draws nothing.
 */
const { test } = require('node:test');
const assert = require('node:assert');

const {
  streakFromTimestamps,
  heardPastNinety,
  readRatio,
  readDone,
  statusChipText,
} = require('../../webui/static/today.js');

function pad(n) { return (n < 10 ? '0' : '') + n; }

/** Local dates at a fixed offset, independent of the machine timezone. */
function atOffset(minutes) {
  return function (ms) {
    const shifted = new Date(ms + minutes * 60000);
    return shifted.getUTCFullYear() + '-' + pad(shifted.getUTCMonth() + 1) + '-' + pad(shifted.getUTCDate());
  };
}

const NOW = Date.parse('2026-10-23T08:00:00Z');
const berlin = atOffset(120);

test('streak follows the local calendar, not UTC', () => {
  // 23:30 UTC is already Friday in a UTC+2 zone.
  const alone = streakFromTimestamps(['2026-10-22T23:30:00Z'], NOW, berlin);
  assert.equal(alone.count, 1);
  assert.equal(alone.dots[4].isToday, true);
  assert.equal(alone.dots[4].reviewed, true);

  const open = streakFromTimestamps([
    '2026-10-20T10:00:00Z',
    '2026-10-21T10:00:00Z',
    '2026-10-22T10:00:00Z',
  ], NOW, berlin);
  assert.equal(open.count, 3);
  assert.deepEqual(open.dots.map((dot) => dot.reviewed), [false, true, true, true, false, false, false]);
  assert.equal(open.dots[4].isToday, true);
  assert.equal(open.dots[4].reviewed, false);
});

test('a gap ends the streak', () => {
  const broken = streakFromTimestamps([
    '2026-10-20T10:00:00Z',
    '2026-10-23T07:00:00Z',
  ], NOW, berlin);
  assert.equal(broken.count, 1);
  assert.equal(broken.dots[4].reviewed, true);
});

test('an episode counts once ninety percent has played', () => {
  assert.equal(heardPastNinety(90, 100), true);
  assert.equal(heardPastNinety(89, 100), false);
  assert.equal(heardPastNinety(10, 0), false);
  assert.equal(heardPastNinety(10, NaN), false);
});

test('reading progress accepts a ratio or a complete flag', () => {
  const store = {};
  global.localStorage = {
    getItem(key) { return Object.prototype.hasOwnProperty.call(store, key) ? store[key] : null; },
  };
  assert.equal(readRatio('missing'), null);
  store['nlp:readProgress:a'] = '0.4';
  assert.equal(readRatio('a'), 0.4);
  assert.equal(readDone(readRatio('a')), false);
  store['nlp:readProgress:b'] = JSON.stringify({ complete: true });
  assert.equal(readDone(readRatio('b')), true);
  store['nlp:readProgress:c'] = JSON.stringify({ ratio: 1 });
  assert.equal(readDone(readRatio('c')), true);
});

test('the status chip stays blank unless the endpoint answers', () => {
  assert.equal(statusChipText(404, { ok: true }), null);
  assert.equal(statusChipText(500, { summary: 'down' }), null);
  assert.equal(statusChipText(200, null), null);
  assert.deepEqual(statusChipText(200, { ok: true }), { ok: true, text: 'All services up' });
  assert.deepEqual(
    statusChipText(200, { summary: 'IndexTTS not running' }),
    { ok: true, text: 'IndexTTS not running' },
  );
  assert.deepEqual(
    statusChipText(200, { ok: false, summary: 'IndexTTS not running' }),
    { ok: false, text: 'IndexTTS not running' },
  );
  assert.deepEqual(
    statusChipText(200, { services: [{ name: 'Database', ok: false }] }),
    { ok: false, text: 'Database needs attention' },
  );
});
