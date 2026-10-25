/*
 * What the cards look like on this device, and what to review next.
 *
 * Each card is the computer's published snapshot with the reviews made since folded on top:
 * this device's own (including ones still waiting to sync) and other devices' that the cloud
 * has heard about. Pure: no DOM, no network, no clock.
 *
 * The computer stays the authority. It folds every card's whole log, so when a late event
 * sorts before one the snapshot already holds, its result can differ from folding forward
 * from the snapshot as done here; the next snapshot replaces this view and corrects it.
 */
import { cardFromSnapshot, foldEvents, isNew } from './fsrs.js';

// Cards a first session introduces. Reviewing more is a decision, not a default.
export const NEW_PER_SESSION = 10;

/** One entry per card in the snapshot: `{card_id, paper_id, card, fresh}`. */
export function effectiveCards(scheduleCards, events) {
  const byCard = new Map();
  for (const event of events) {
    if (!byCard.has(event.card_id)) byCard.set(event.card_id, []);
    byCard.get(event.card_id).push(event);
  }
  return (scheduleCards || []).map((entry) => {
    const card = foldEvents(cardFromSnapshot(entry), byCard.get(entry.card_id) || []);
    return { card_id: entry.card_id, paper_id: entry.paper_id, card, fresh: isNew(card) };
  });
}

/** Cards that are due, longest overdue first. */
export function dueCards(states, nowMs) {
  return states.filter((s) => !s.fresh && s.card.due <= nowMs).sort((a, b) => a.card.due - b.card.due || (a.card_id < b.card_id ? -1 : 1));
}

/**
 * The cards to show now: everything due, then new cards up to `newLimit` for the session.
 * `introduced` is the set of new cards this session has already shown, so reviewing one does
 * not bring the next new card in without limit.
 */
export function buildQueue(states, nowMs, { newLimit = NEW_PER_SESSION, introduced = new Set() } = {}) {
  const due = dueCards(states, nowMs);
  const room = Math.max(0, newLimit - introduced.size);
  const fresh = states.filter((s) => s.fresh && !introduced.has(s.card_id)).sort((a, b) => (a.card_id < b.card_id ? -1 : 1));
  // A card of the session's introduced set that has been reviewed once is no longer fresh and
  // comes back through `due` when it is.
  return [...due, ...fresh.slice(0, room)];
}

export function summarize(states, nowMs) {
  const out = { total: states.length, due: 0, fresh: 0, upcoming: 0 };
  for (const s of states) {
    if (s.fresh) out.fresh += 1;
    else if (s.card.due <= nowMs) out.due += 1;
    else out.upcoming += 1;
  }
  return out;
}

/** When the next scheduled card becomes due (epoch ms), or null if none is waiting. */
export function nextDueAt(states, nowMs) {
  let soonest = null;
  for (const s of states) if (!s.fresh && s.card.due > nowMs && (soonest === null || s.card.due < soonest)) soonest = s.card.due;
  return soonest;
}

/**
 * The states as `schedule.json`-shaped cards (`{card_id, paper_id, state, due}`), so the Today
 * page's counting helpers work on what this device holds instead of on the published snapshot.
 */
export function asSnapshotCards(states) {
  return states.map((s) => ({
    card_id: s.card_id,
    paper_id: s.paper_id,
    state: s.fresh ? 'new' : 'review',
    due: s.fresh ? null : new Date(s.card.due).toISOString(),
  }));
}
