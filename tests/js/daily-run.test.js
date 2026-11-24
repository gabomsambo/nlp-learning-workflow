'use strict';
// Pure helpers of webui/static/daily-run.js. The DOM wiring is not covered.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const dailyRun = require(path.join(__dirname, '..', '..', 'webui', 'static', 'daily-run.js'));

test('enableMessage names the time, zone, daily cost and cap', () => {
  const text = dailyRun.enableMessage('08:30', 'Europe/Berlin', 0.2343, 2);
  assert.match(text, /08:30 \(Europe\/Berlin\)/);
  assert.match(text, /about \$0\.23 a day/);
  assert.match(text, /\$2\.00 daily cap/);
});

test('errorText prefers FastAPI detail, then validation messages, then error', () => {
  assert.equal(dailyRun.errorText({ detail: 'Run time out of range' }, 422), 'Run time out of range');
  assert.equal(dailyRun.errorText({ detail: [{ msg: 'field required' }, { msg: 'bad' }] }, 422), 'field required; bad');
  assert.equal(dailyRun.errorText({ error: 'nope' }, 500), 'nope');
  assert.equal(dailyRun.errorText(null, 502), 'Request failed (HTTP 502).');
});

test('runNowSummary counts started and skipped runs', () => {
  assert.equal(dailyRun.runNowSummary({ runs: [{ run_id: 'a' }] }), 'Started 1 run.');
  assert.equal(
    dailyRun.runNowSummary({ runs: [{ run_id: 'a' }, { run_id: null }, { run_id: 'b' }] }),
    'Started 2 runs. 1 not started; see Last run.'
  );
});
