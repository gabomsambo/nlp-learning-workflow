const { test } = require('node:test');
const assert = require('node:assert');

const { paperPageHref, enqueueMessage } = require('../../webui/static/pillar-search.js');

test('paper page hook stays null until the library route exists', () => {
  assert.equal(paperPageHref('2403.05525'), null);
  assert.equal(paperPageHref('file:abc'), null);
});

test('enqueue feedback names the auto-run queue', () => {
  assert.equal(enqueueMessage('queued'), 'Queued for the next auto-run.');
  assert.equal(enqueueMessage('already_queued'), 'Already waiting for the next auto-run.');
  assert.equal(enqueueMessage('nope'), 'Could not queue this paper.');
});
