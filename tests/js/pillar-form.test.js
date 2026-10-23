const { test } = require('node:test');
const assert = require('node:assert');

const { validatePillarField } = require('../../webui/static/pillar-form.js');

test('name is required and capped at 100 characters', () => {
  assert.equal(validatePillarField('name', '   '), 'Name is required');
  assert.equal(validatePillarField('name', 'a'.repeat(101)), 'Name must be 100 characters or less');
  assert.equal(validatePillarField('name', 'Agents'), '');
});

test('goal must be between 10 and 500 characters', () => {
  assert.equal(validatePillarField('goal', 'short'), 'Goal must be at least 10 characters');
  assert.equal(validatePillarField('goal', ''), 'Goal is required');
  assert.equal(validatePillarField('goal', 'a'.repeat(501)), 'Goal must be 500 characters or less');
  assert.equal(validatePillarField('goal', 'Learn how agents use tools.'), '');
});
