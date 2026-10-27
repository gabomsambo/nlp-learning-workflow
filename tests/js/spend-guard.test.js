/*
 * Pure helpers in webui/static/spend-guard.js: which requests count as costly,
 * and how the budget reads. The dialog and the fetch wrapper need a DOM and are
 * not covered here; the server-side cap and confirmation are covered in
 * tests/webui/test_spend_guard.py.
 */
const { test } = require('node:test');
const assert = require('node:assert');

const { matchAction, pathOf, methodOf, budgetText, money } = require('../../webui/static/spend-guard.js');

const COSTLY = [
  '/pipeline/run?pillar=neural-architectures-language&papers=1',
  '/api/pillars/neural-architectures-language/select',
  '/api/pillars/neural-architectures-language/upload/url',
  '/api/pillars/neural-architectures-language/upload/pdf',
  '/api/podcast/generate',
  '/api/podcast/tts/generate',
  '/api/podcast/0b1c2d3e-0000-4000-8000-000000000000/factcheck',
  '/api/follow-along',
];

test('every costly POST is recognised, query strings ignored', () => {
  for (const url of COSTLY) {
    assert.ok(matchAction('POST', pathOf(url)), url);
  }
});

test('reads and cheap writes are not gated', () => {
  assert.equal(matchAction('GET', pathOf(COSTLY[4])), null);
  assert.equal(matchAction('GET', pathOf('/api/podcast/abc/factcheck')), null);
  assert.equal(matchAction('POST', pathOf('/api/quiz/review')), null);
  assert.equal(matchAction('POST', pathOf('/api/pillars/x/discover')), null);
  assert.equal(matchAction('POST', pathOf('/api/pillars/x/queue')), null);
  assert.equal(matchAction('POST', pathOf('/api/pipeline-runs/abc/cancel')), null);
  assert.equal(matchAction('POST', pathOf('/api/papers/abc/refresh-metadata')), null);
  assert.equal(matchAction('GET', pathOf('/api/follow-along?script_id=abc')), null);
  assert.equal(matchAction('GET', pathOf('/api/follow-along/playgrounds')), null);
});

test('method comes from init, then the request, defaulting to GET', () => {
  assert.equal(methodOf('/x', { method: 'post' }), 'POST');
  assert.equal(methodOf({ method: 'DELETE' }, undefined), 'DELETE');
  assert.equal(methodOf('/x', undefined), 'GET');
});

test('budget text', () => {
  assert.equal(money(1.5), '$1.50');
  assert.equal(money('nope'), '$?');
  assert.equal(
    budgetText({ cap_usd: 2, spent_usd: 0.5, remaining_usd: 1.5, exhausted: false }),
    '$1.50 left of $2.00 today',
  );
  assert.match(budgetText({ cap_usd: 2, spent_usd: 2.1, remaining_usd: 0, exhausted: true }), /cap reached/i);
  assert.equal(budgetText(null), 'Budget unavailable');
});
