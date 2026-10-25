// The phone's scheduler against py-fsrs. tests/fixtures/fsrs/sequences.json is made by
// scripts/gen_fsrs_fixtures.py from the Python fold the computer uses; this file holds the
// JavaScript port to it. A case is a review history and, after each distinct event in fold
// order, the card py-fsrs derives: state, step, stability, difficulty, due and last review.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  PARAMETERS, Rating, State, cardFromSnapshot, describe, foldEvents, formatInterval, newCard, orderEvents,
  previewIntervals, reviewCard, toMs,
} from '../public/js/fsrs.js';

const fixture = JSON.parse(readFileSync(new URL('../../../tests/fixtures/fsrs/sequences.json', import.meta.url), 'utf8'));
const MIN = 60 * 1000;
const DAY = 24 * 60 * MIN;

function expectSame(actual, expected, label) {
  const got = describe(actual);
  assert.equal(got.state, expected.state, `${label}: state`);
  assert.equal(got.step, expected.step, `${label}: step`);
  assert.equal(got.due_ms, expected.due_ms, `${label}: due`);
  assert.equal(got.last_review_ms, expected.last_review_ms, `${label}: last review`);
  // The schedule above is exact. Stability and difficulty are doubles that go through pow(), and the
  // C library behind Python and V8's own differ in the last digit or two (relative error ~1e-15 here),
  // so they are held to 1e-12: far inside anything that moves a due date, which is rounded to a day.
  close(got.stability, expected.stability, `${label}: stability`);
  close(got.difficulty, expected.difficulty, `${label}: difficulty`);
}

function close(actual, expected, label) {
  assert.ok(Math.abs(actual - expected) <= 1e-12 * Math.max(1, Math.abs(expected)), `${label}: ${actual} vs ${expected}`);
}

test('the fixture was made with the scheduler this port copies', () => {
  assert.deepEqual([...PARAMETERS], fixture.scheduler.parameters);
  assert.equal(fixture.scheduler.library, 'py-fsrs');
  assert.equal(fixture.scheduler.enable_fuzzing, false);
  assert.equal(fixture.scheduler.desired_retention, 0.9);
  assert.equal(fixture.scheduler.maximum_interval, 36500);
  assert.deepEqual(fixture.scheduler.learning_steps_seconds, [60, 600]);
  assert.deepEqual(fixture.scheduler.relearning_steps_seconds, [600]);
  assert.ok(fixture.cases.length >= 14);
});

for (const c of fixture.cases) {
  test(`${c.name}: every step matches py-fsrs`, () => {
    let card = newCard();
    const ordered = orderEvents(c.events);
    assert.equal(ordered.length, c.after_each.length, 'one expected card per distinct event');
    ordered.forEach((event, i) => {
      card = reviewCard(card, event.rating, toMs(event.reviewed_at));
      expectSame(card, c.after_each[i], `after review ${i + 1}`);
    });
    expectSame(card, c.expected, 'final');
  });

  test(`${c.name}: any arrival order, repeats included, folds to the same card`, () => {
    const shuffled = [...c.events, ...c.events.slice(0, 2)].sort((a, b) => (a.event_id < b.event_id ? 1 : -1));
    expectSame(foldEvents(newCard(), shuffled), c.expected, 'shuffled');
  });

  test(`${c.name}: carrying on from a published snapshot gives the same card`, () => {
    // The phone starts from schedule.json and folds the events after it. For every point the
    // desktop could have published at, the rest of the history must land on the same card.
    const ordered = orderEvents(c.events);
    for (let k = 0; k < ordered.length; k += 1) {
      const published = c.after_each[k];
      const snapshot = {
        state: { 1: 'learning', 2: 'review', 3: 'relearning' }[published.state],
        step: published.step,
        stability: published.stability,
        difficulty: published.difficulty,
        due: published.due,
        last_review: published.last_review,
      };
      const resumed = foldEvents(cardFromSnapshot(snapshot), ordered.slice(k + 1));
      expectSame(resumed, c.expected, `resumed after ${k + 1}`);
    }
  });
}

test('a card nobody has reviewed is new, and reviews begin it', () => {
  const fresh = cardFromSnapshot({ state: 'new', due: null, stability: 0.0, difficulty: 0.0, last_review: null, step: null });
  assert.deepEqual(fresh, newCard());
  assert.equal(foldEvents(fresh, []), fresh);
  const first = reviewCard(fresh, Rating.Good, Date.UTC(2026, 2, 2, 9));
  assert.equal(first.state, State.Learning);
  assert.equal(first.step, 1);
  assert.equal(first.due - first.lastReview, 10 * MIN);
});

test('a snapshot the computer could only read from the card row still resumes', () => {
  const card = cardFromSnapshot({
    state: 'review', due: '2026-03-10T09:00:00+00:00', stability: 5.5, difficulty: 4.2,
    last_review: '2026-03-05T09:00:00+00:00', step: null, reps: null, lapses: 1,
  });
  assert.equal(card.step, null);
  const next = reviewCard(card, Rating.Good, Date.UTC(2026, 2, 10, 9));
  assert.equal(next.state, State.Review);
  assert.ok(next.due > Date.UTC(2026, 2, 10, 9));
  const learning = cardFromSnapshot({ state: 'learning', due: '2026-03-10T09:00:00+00:00', stability: 1, difficulty: 5, last_review: '2026-03-10T08:50:00+00:00', step: null });
  assert.equal(learning.step, 0);
});

test('rating a card does not change the card it was given', () => {
  const card = reviewCard(newCard(), Rating.Good, Date.UTC(2026, 2, 2, 9));
  const copy = { ...card };
  reviewCard(card, Rating.Again, Date.UTC(2026, 2, 3, 9));
  assert.deepEqual(card, copy);
});

test('ratings outside 1 to 4 are refused', () => {
  for (const bad of [0, 5, 2.5, '3', null]) assert.throws(() => reviewCard(newCard(), bad, 0), RangeError);
});

test('the button previews are the intervals a rating really gives', () => {
  const at = Date.UTC(2026, 2, 2, 9);
  const card = foldEvents(newCard(), [
    { event_id: 'a', rating: 3, reviewed_at: '2026-03-02T09:00:00.000Z' },
    { event_id: 'b', rating: 3, reviewed_at: '2026-03-02T09:10:00.000Z' },
  ]);
  const later = at + 3 * DAY;
  const preview = previewIntervals(card, later);
  for (const rating of [1, 2, 3, 4]) assert.equal(preview[rating], reviewCard(card, rating, later).due - later);
  assert.equal(preview[1], 10 * MIN, 'Again goes to relearning');
  assert.ok(preview[2] < preview[3] && preview[3] < preview[4]);
});

test('intervals read as short labels', () => {
  assert.equal(formatInterval(20 * 1000), '<1 min');
  assert.equal(formatInterval(10 * MIN), '10 min');
  assert.equal(formatInterval(5 * 60 * MIN), '5 h');
  assert.equal(formatInterval(DAY), '1 day');
  assert.equal(formatInterval(9 * DAY), '9 days');
  assert.equal(formatInterval(60 * DAY), '2 months');
  assert.equal(formatInterval(800 * DAY), '2.2 years');
});
