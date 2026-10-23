/*
 * Pure helpers for the paper page. The answers toggle shares
 * nlp:quizAnswersVisible with the modal; reading progress is local only.
 */
const { test } = require('node:test');
const assert = require('node:assert');

const {
  progressKey,
  progressLabel,
  depthPercent,
  mergeProgress,
  toggleLabel,
  ANSWERS_KEY,
} = require('../../webui/static/paper-page.js');

test('answers key matches the modal', () => {
  assert.equal(ANSWERS_KEY, 'nlp:quizAnswersVisible');
  assert.equal(toggleLabel(true), 'Hide answers');
  assert.equal(toggleLabel(false), 'Show answers');
});

test('progress key is namespaced by paper id', () => {
  assert.equal(progressKey('arxiv:2402.08593'), 'nlp:readProgress:arxiv:2402.08593');
});

test('progress label does not show a zero', () => {
  assert.equal(progressLabel(0), 'On this browser · not started');
  assert.equal(progressLabel(40), 'On this browser · 40% of the lesson');
});

test('depthPercent is null when the lesson is not laid out', () => {
  assert.equal(depthPercent(0, 0, 800), null);
  assert.equal(depthPercent(0, 400, 800), 100);
  assert.equal(depthPercent(600, 400, 800), 50);
  assert.equal(depthPercent(900, 400, 800), 0);
});

test('mergeProgress keeps the furthest point and ignores an unknown read', () => {
  assert.equal(mergeProgress(10, null), 10);
  assert.equal(mergeProgress(40, 10), 40);
  assert.equal(mergeProgress(40, 80), 80);
  assert.equal(mergeProgress(0, 120), 100);
});
