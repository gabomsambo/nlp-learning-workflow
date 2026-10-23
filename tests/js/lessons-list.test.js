/*
 * j/k wrapping for the lessons list. The key listener needs a DOM and is
 * exercised in the browser.
 */
const { test } = require('node:test');
const assert = require('node:assert');

const { nextIndex } = require('../../webui/static/lessons.js');

test('nextIndex wraps and starts from either end', () => {
  assert.equal(nextIndex(-1, 1, 3), 0);
  assert.equal(nextIndex(-1, -1, 3), 2);
  assert.equal(nextIndex(2, 1, 3), 0);
  assert.equal(nextIndex(0, -1, 3), 2);
  assert.equal(nextIndex(0, 1, 0), -1);
});
