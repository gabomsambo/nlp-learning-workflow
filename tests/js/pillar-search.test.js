const { test } = require('node:test');
const assert = require('node:assert');

const { paperPageHref, enqueueMessage } = require('../../webui/static/pillar-search.js');

test('View Paper links to the paper page, encoding colon ids', () => {
  assert.equal(paperPageHref('2403.05525'), '/papers/2403.05525');
  assert.equal(paperPageHref('arxiv:2402.08593'), '/papers/arxiv%3A2402.08593');
  assert.equal(paperPageHref('file:abc'), '/papers/file%3Aabc');
  assert.equal(paperPageHref(''), null);
  assert.equal(paperPageHref(null), null);
});

test('enqueue feedback names the auto-run queue', () => {
  assert.equal(enqueueMessage('queued'), 'Queued for the next auto-run.');
  assert.equal(enqueueMessage('already_queued'), 'Already waiting for the next auto-run.');
  assert.equal(enqueueMessage('nope'), 'Could not queue this paper.');
});
