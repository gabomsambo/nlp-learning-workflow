/*
 * The reader's side of review sync: the outbox, the cloud and the published snapshot, together.
 *
 *   states()     every card as it stands on this device: the snapshot, plus the reviews made
 *                since (here, waiting to sync, and on other devices)
 *   rate()       write the review down first, then send it
 *   sync()       send what is waiting, then hear what other devices did; never throws
 *   status()     what the page says about it: reviews waiting, the session ended, ...
 *
 * A review is never held up by the network: `rate` records it and returns, and the send
 * happens behind it. If the send fails the review waits in the outbox and goes out the next
 * time `sync` runs (page open, `online`, page shown again).
 */
import { effectiveCards } from './schedule.js';

export function createReviews({ store, outbox, sync, now = Date.now }) {
  const listeners = new Set();
  let last = { status: 'nothing' };

  const changed = () => {
    for (const listener of listeners) {
      try { listener(); } catch (error) { /* a broken listener must not stop a review */ }
    }
  };

  /** The snapshot, with events it already holds dropped from the outbox so none is counted twice. */
  async function snapshot() {
    const schedule = await store.schedule();
    outbox.reconcile(Number.isInteger(schedule.through_seq) ? schedule.through_seq : 0);
    return schedule;
  }

  return {
    onChange(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },

    async states() {
      const schedule = await snapshot();
      return effectiveCards(schedule.cards, [...outbox.peerEvents(), ...outbox.events()]);
    },

    /** Record a review and start sending it. Resolves with the event once it is written down. */
    rate(cardId, rating, { responseMs = null, sessionId = null } = {}) {
      const event = outbox.record({ card_id: cardId, rating, response_time_ms: responseMs, session_id: sessionId });
      changed();
      this.flush();
      return event;
    },

    /** Send what is waiting. */
    async flush() {
      last = await sync.flush();
      changed();
      return last;
    },

    /**
     * A fresh snapshot if one can be had, then send, then listen. Every step may fail
     * without harm; the page keeps working from what it holds.
     */
    async sync() {
      try {
        await store.refresh();
      } catch (error) {
        // Offline, or nothing published yet: the snapshot we hold is still right to review from.
      }
      let schedule;
      try {
        schedule = await snapshot();
      } catch (error) {
        return last;
      }
      const sent = await sync.flush();
      const heard = await sync.pullPeers(Number.isInteger(schedule.through_seq) ? schedule.through_seq : 0);
      // The worst thing to tell the person wins: a ended session over a mere "offline".
      last = [sent, heard].find((r) => r.status === 'auth') || [sent, heard].find((r) => r.status === 'error' || r.status === 'offline') || sent;
      changed();
      return last;
    },

    status() {
      return {
        waiting: outbox.waiting(),
        rejected: outbox.rejected(),
        durable: outbox.durable(),
        last: last.status,
      };
    },

    clearRejected() {
      outbox.clearRejected();
      changed();
    },

    deviceId: () => outbox.deviceId(),
    now,
  };
}

/** A line for the page, or '' when there is nothing to say. */
export function statusLine({ waiting, rejected, durable, last }) {
  const parts = [];
  if (waiting > 0) {
    const reviews = `${waiting} review${waiting === 1 ? '' : 's'}`;
    if (last === 'auth') parts.push(`${reviews} waiting. Sign in again to send ${waiting === 1 ? 'it' : 'them'}.`);
    else if (last === 'offline') parts.push(`${reviews} waiting to sync. They will send when you are back online.`);
    else parts.push(`${reviews} waiting to sync.`);
  }
  if (rejected > 0) parts.push(`${rejected} review${rejected === 1 ? ' was' : 's were'} not accepted (made too long ago or with the wrong clock).`);
  if (!durable) parts.push('This browser is not keeping reviews between visits. Reviews made now are lost if the page closes before they sync.');
  return parts.join(' ');
}
