/*
 * The phone's scheduler: py-fsrs 6.3.1's `Scheduler.review_card`, ported.
 *
 * The computer folds each card's review log with py-fsrs (nlp_pillars/fsrs_fold.py),
 * and a card is whatever that fold says. The phone has to arrive at the same card
 * when it schedules offline, so this is a port of that scheduler and not a different
 * FSRS library: ts-fsrs 5.4.2 was tried first, and on tests/fixtures/fsrs/sequences.json
 * it disagrees with py-fsrs about the due date in 38 of the 100 reviews. It counts days
 * as calendar days where py-fsrs counts 24-hour periods, it does not raise stability
 * for a same-day "Hard" the way py-fsrs does, it repeats the Hard interval of the first
 * step on later steps, and it rounds stability and difficulty to eight decimals after every
 * review. Any one of those moves a due date by a day.
 *
 * The same fixtures hold this file to py-fsrs exactly (tests/js/ and
 * cloud/reader/test/fsrs.test.mjs). Fuzz is off, as on the computer, so a history
 * has one outcome whoever replays it.
 *
 * Pure: no DOM, no network, no clock. A card is
 *   {state: 1|2|3 (Learning, Review, Relearning), step, stability, difficulty, due, lastReview}
 * with dates as epoch milliseconds. A card never reviewed is Learning with no stability,
 * which is how py-fsrs starts one.
 */

export const PARAMETERS = Object.freeze([
  0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001, 1.8722, 0.1666, 0.796,
  1.4835, 0.0614, 0.2629, 1.6483, 0.6014, 1.8729, 0.5425, 0.0912, 0.0658, 0.1542,
]);

export const Rating = Object.freeze({ Again: 1, Hard: 2, Good: 3, Easy: 4 });
export const State = Object.freeze({ Learning: 1, Review: 2, Relearning: 3 });

const MINUTE = 60 * 1000;
const DAY = 24 * 60 * MINUTE;
const LEARNING_STEPS = [1 * MINUTE, 10 * MINUTE];
const RELEARNING_STEPS = [10 * MINUTE];
const DESIRED_RETENTION = 0.9;
const MAXIMUM_INTERVAL = 36500;
const STABILITY_MIN = 0.001;
const MIN_DIFFICULTY = 1.0;
const MAX_DIFFICULTY = 10.0;

const P = PARAMETERS;
const DECAY = -P[20];
const FACTOR = 0.9 ** (1 / DECAY) - 1;

// Python's `math.e ** x` is a pow(), not an exp(); the two can differ in the last digit.
const ePow = (x) => Math.pow(Math.E, x);

/** Python's round(): halves go to the even neighbour. */
function roundHalfEven(x) {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

const clampStability = (s) => Math.max(s, STABILITY_MIN);
const clampDifficulty = (d) => Math.min(Math.max(d, MIN_DIFFICULTY), MAX_DIFFICULTY);

function initialStability(rating) {
  return clampStability(P[rating - 1]);
}

function initialDifficulty(rating, clamp) {
  const d = P[4] - ePow(P[5] * (rating - 1)) + 1;
  return clamp ? clampDifficulty(d) : d;
}

function nextDifficulty(difficulty, rating) {
  const arg1 = initialDifficulty(Rating.Easy, false);
  const delta = -(P[6] * (rating - 3));
  const arg2 = difficulty + ((10.0 - difficulty) * delta) / 9.0;
  return clampDifficulty(P[7] * arg1 + (1 - P[7]) * arg2);
}

function shortTermStability(stability, rating) {
  let increase = ePow(P[17] * (rating - 3 + P[18])) * Math.pow(stability, -P[19]);
  if (rating === Rating.Good || rating === Rating.Easy) increase = Math.max(increase, 1.0);
  return clampStability(stability * increase);
}

function nextForgetStability(difficulty, stability, retrievability) {
  const longTerm = P[11]
    * Math.pow(difficulty, -P[12])
    * (Math.pow(stability + 1, P[13]) - 1)
    * ePow((1 - retrievability) * P[14]);
  const shortTerm = stability / ePow(P[17] * P[18]);
  return Math.min(longTerm, shortTerm);
}

function nextRecallStability(difficulty, stability, retrievability, rating) {
  const hardPenalty = rating === Rating.Hard ? P[15] : 1;
  const easyBonus = rating === Rating.Easy ? P[16] : 1;
  return stability * (1
    + ePow(P[8])
    * (11 - difficulty)
    * Math.pow(stability, -P[9])
    * (ePow((1 - retrievability) * P[10]) - 1)
    * hardPenalty
    * easyBonus);
}

function nextStability(difficulty, stability, retrievability, rating) {
  const next = rating === Rating.Again
    ? nextForgetStability(difficulty, stability, retrievability)
    : nextRecallStability(difficulty, stability, retrievability, rating);
  return clampStability(next);
}

/** Whole 24-hour periods between two instants (Python's `timedelta.days`: floored). */
const daysBetween = (fromMs, toMs) => Math.floor((toMs - fromMs) / DAY);

function retrievability(card, atMs) {
  if (card.lastReview === null || card.stability === null) return 0;
  const elapsed = Math.max(0, daysBetween(card.lastReview, atMs));
  return Math.pow(1 + (FACTOR * elapsed) / card.stability, DECAY);
}

function nextIntervalDays(stability) {
  const raw = (stability / FACTOR) * (Math.pow(DESIRED_RETENTION, 1 / DECAY) - 1);
  return Math.min(Math.max(roundHalfEven(raw), 1), MAXIMUM_INTERVAL);
}

/** A card that has never been reviewed. */
export function newCard() {
  return { state: State.Learning, step: 0, stability: null, difficulty: null, due: null, lastReview: null };
}

export const isNew = (card) => card.lastReview === null;

/** How a Hard rating moves through the steps: the same step again, or between the first two. */
function hardInterval(steps, step) {
  if (step === 0 && steps.length === 1) return steps[0] * 1.5;
  if (step === 0 && steps.length >= 2) return (steps[0] + steps[1]) / 2.0;
  return steps[step];
}

/**
 * `card` after a review rated `rating` at `atMs`. The card passed in is not changed.
 * Mirrors Scheduler.review_card in py-fsrs, one branch at a time.
 */
export function reviewCard(card, rating, atMs) {
  if (![1, 2, 3, 4].includes(rating)) throw new RangeError(`rating must be 1 to 4, got ${rating}`);
  const next = { ...card };
  const days = card.lastReview !== null ? daysBetween(card.lastReview, atMs) : null;
  const sameDay = days !== null && days < 1;
  let interval; // milliseconds

  const graduate = () => {
    next.state = State.Review;
    next.step = null;
    interval = nextIntervalDays(next.stability) * DAY;
  };

  if (card.state === State.Learning || card.state === State.Relearning) {
    const steps = card.state === State.Learning ? LEARNING_STEPS : RELEARNING_STEPS;
    if (card.state === State.Learning && (card.stability === null || card.difficulty === null)) {
      next.stability = initialStability(rating);
      next.difficulty = initialDifficulty(rating, true);
    } else if (sameDay) {
      next.stability = shortTermStability(card.stability, rating);
      next.difficulty = nextDifficulty(card.difficulty, rating);
    } else {
      next.stability = nextStability(card.difficulty, card.stability, retrievability(card, atMs), rating);
      next.difficulty = nextDifficulty(card.difficulty, rating);
    }

    const step = card.step;
    if (steps.length === 0 || (step >= steps.length && rating !== Rating.Again)) {
      graduate();
    } else if (rating === Rating.Again) {
      next.step = 0;
      interval = steps[0];
    } else if (rating === Rating.Hard) {
      interval = hardInterval(steps, step);
    } else if (rating === Rating.Good) {
      if (step + 1 === steps.length) {
        graduate();
      } else {
        next.step = step + 1;
        interval = steps[next.step];
      }
    } else {
      graduate();
    }
  } else {
    if (sameDay) {
      next.stability = shortTermStability(card.stability, rating);
    } else {
      next.stability = nextStability(card.difficulty, card.stability, retrievability(card, atMs), rating);
    }
    next.difficulty = nextDifficulty(card.difficulty, rating);
    if (rating === Rating.Again) {
      next.state = State.Relearning;
      next.step = 0;
      interval = RELEARNING_STEPS[0];
    } else {
      interval = nextIntervalDays(next.stability) * DAY;
    }
  }

  next.due = atMs + interval;
  next.lastReview = atMs;
  return next;
}

// ---------------------------------------------------------------------------
// Events: the fold
// ---------------------------------------------------------------------------

/** Epoch ms of an ISO UTC timestamp (what events carry), or NaN. */
export const toMs = (iso) => Date.parse(iso);

/**
 * Events in fold order: by time, then by event id; an event id counted once.
 * The same order the computer uses (fsrs_fold.ordered), so every device agrees.
 */
export function orderEvents(events) {
  const unique = new Map();
  for (const event of events) if (!unique.has(event.event_id)) unique.set(event.event_id, event);
  return [...unique.values()].sort((a, b) => {
    const ta = toMs(a.reviewed_at);
    const tb = toMs(b.reviewed_at);
    if (ta !== tb) return ta - tb;
    return a.event_id < b.event_id ? -1 : a.event_id > b.event_id ? 1 : 0;
  });
}

/** `card` after `events` (any order, repeats allowed), starting from `card`. */
export function foldEvents(card, events) {
  let current = card;
  for (const event of orderEvents(events)) current = reviewCard(current, event.rating, toMs(event.reviewed_at));
  return current;
}

// ---------------------------------------------------------------------------
// The published snapshot
// ---------------------------------------------------------------------------

const STATE_BY_NAME = { learning: State.Learning, review: State.Review, relearning: State.Relearning };
export const STATE_NAME = { [State.Learning]: 'learning', [State.Review]: 'review', [State.Relearning]: 'relearning' };

/**
 * A card from a `schedule.json` entry (nlp_pillars/cloud/export.py::schedule_cards).
 * "new" is a card nobody has reviewed. An entry the computer could only read from the
 * card row has no `step`: a learning or relearning card then starts at its first.
 */
export function cardFromSnapshot(entry) {
  if (!entry || entry.state === 'new' || !STATE_BY_NAME[entry.state] || !entry.last_review) return newCard();
  const state = STATE_BY_NAME[entry.state];
  return {
    state,
    step: state === State.Review ? null : (Number.isInteger(entry.step) ? entry.step : 0),
    stability: entry.stability,
    difficulty: entry.difficulty,
    due: toMs(entry.due),
    lastReview: toMs(entry.last_review),
  };
}

/** The numbers two schedulers must agree on, for comparing a fold with a fixture. */
export function describe(card) {
  return {
    state: card.state,
    step: card.step,
    stability: card.stability,
    difficulty: card.difficulty,
    due_ms: card.due,
    last_review_ms: card.lastReview,
  };
}

// ---------------------------------------------------------------------------
// What a rating would do, for the buttons
// ---------------------------------------------------------------------------

/** `{1: ms, 2: ms, 3: ms, 4: ms}`: how long until the card is next due after each rating, at `atMs`. */
export function previewIntervals(card, atMs) {
  const out = {};
  for (const rating of [1, 2, 3, 4]) out[rating] = reviewCard(card, rating, atMs).due - atMs;
  return out;
}

/** "10 min", "3 days", "2 months": short enough for a button. */
export function formatInterval(ms) {
  const minutes = Math.round(ms / MINUTE);
  if (minutes < 1) return '<1 min';
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h`;
  const days = Math.round(ms / DAY);
  if (days < 30) return days === 1 ? '1 day' : `${days} days`;
  const months = Math.round(days / 30.4375);
  if (months < 12) return months === 1 ? '1 month' : `${months} months`;
  const years = Math.round((days / 365.25) * 10) / 10;
  return years === 1 ? '1 year' : `${years} years`;
}
