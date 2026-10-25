/*
 * The phone's record of its own reviews, and the other devices' reviews it has heard about.
 *
 * A review is written here first and sent afterwards, so closing the page, losing signal
 * or a failed request never loses one. It stays until the computer's schedule snapshot
 * includes it (`reconcile`), because until then the card on this device is "the snapshot
 * plus these events". The outbox is the only local data that cannot be fetched again.
 *
 * Storage is an interface, `{load(): object|null, save(object): void}`, and nothing else
 * here knows where it lives. The reader gives it localStorage today; the offline step swaps in
 * IndexedDB without touching this file. Every storage call is allowed to throw (private
 * windows, a full disk): the outbox then carries on in memory and says so through `durable()`.
 *
 * Pure apart from the storage and the injected clock and id source: node --test runs it.
 */

export const STORAGE_KEY = 'nlp-reader:reviews:v1';
// A runaway loop must not fill the phone. At about 150 bytes an event this is well over a year of
// reviews at a hundred a day, so reaching it means something is wrong, and it is refused loudly.
export const MAX_EVENTS = 5000;

export class OutboxFullError extends Error {}

export function memoryStorage(initial = null) {
  let held = initial;
  return { load: () => held, save: (value) => { held = JSON.parse(JSON.stringify(value)); } };
}

/** localStorage under one key. Throws nothing at construction; each call may throw. */
export function localStorageStorage(key = STORAGE_KEY, area = globalThis.localStorage) {
  return {
    load() {
      const raw = area.getItem(key);
      return raw === null ? null : JSON.parse(raw);
    },
    save(value) {
      area.setItem(key, JSON.stringify(value));
    },
  };
}

const isObject = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);

function randomId() {
  return globalThis.crypto.randomUUID();
}

function emptyState(deviceId) {
  return { v: 1, device_id: deviceId, outbox: [], peers: { through_seq: 0, events: [] }, rejected: 0 };
}

/** Keep only what has the right shape: a half-written or older state must never break the page. */
function sanitize(loaded, deviceId) {
  if (!isObject(loaded) || loaded.v !== 1) return emptyState(deviceId);
  const state = emptyState(typeof loaded.device_id === 'string' && loaded.device_id.length >= 8 ? loaded.device_id : deviceId);
  const keepEvent = (e) => isObject(e) && typeof e.event_id === 'string' && typeof e.card_id === 'string'
    && Number.isInteger(e.rating) && e.rating >= 1 && e.rating <= 4 && typeof e.reviewed_at === 'string';
  if (Array.isArray(loaded.outbox)) {
    for (const item of loaded.outbox.slice(0, MAX_EVENTS)) {
      if (isObject(item) && keepEvent(item.event)) {
        state.outbox.push({ event: item.event, ack_seq: Number.isInteger(item.ack_seq) ? item.ack_seq : null });
      }
    }
  }
  if (isObject(loaded.peers) && Array.isArray(loaded.peers.events)) {
    state.peers = {
      through_seq: Number.isInteger(loaded.peers.through_seq) ? loaded.peers.through_seq : 0,
      events: loaded.peers.events.filter((e) => keepEvent(e) && Number.isInteger(e.seq)).slice(0, MAX_EVENTS),
    };
  }
  state.rejected = Number.isInteger(loaded.rejected) && loaded.rejected > 0 ? loaded.rejected : 0;
  return state;
}

export function createOutbox(storage, { now = Date.now, newId = randomId } = {}) {
  let persisted = true;
  let state;
  try {
    state = sanitize(storage.load(), newId());
  } catch (error) {
    state = emptyState(newId());
    persisted = false;
  }

  function save() {
    try {
      storage.save(state);
      persisted = true;
    } catch (error) {
      persisted = false;
    }
  }
  save(); // a new device id is kept from the first moment

  return {
    deviceId: () => state.device_id,
    /** False when the last save failed: reviews are held in memory only until it works again. */
    durable: () => persisted,

    /**
     * Write a review down. `reviewed_at` is the device's clock to the millisecond: the same
     * precision every other device and the computer fold with.
     */
    record({ card_id: cardId, rating, response_time_ms: responseMs = null, session_id: sessionId = null }) {
      if (state.outbox.length >= MAX_EVENTS) throw new OutboxFullError('Too many reviews are waiting to sync.');
      const event = {
        event_id: newId(),
        card_id: cardId,
        rating,
        reviewed_at: new Date(now()).toISOString(),
        device_id: state.device_id,
      };
      if (Number.isInteger(responseMs) && responseMs >= 0) event.response_time_ms = Math.min(responseMs, 3600000);
      if (typeof sessionId === 'string') event.session_id = sessionId;
      state.outbox.push({ event, ack_seq: null });
      save();
      return event;
    },

    /** Events the cloud has not confirmed, oldest first. */
    pending: () => state.outbox.filter((item) => item.ack_seq === null).map((item) => item.event),
    /** Every review made here that the snapshot does not yet include. */
    events: () => state.outbox.map((item) => item.event),
    waiting: () => state.outbox.filter((item) => item.ack_seq === null).length,

    /**
     * Apply the cloud's answer to a POST: `results` is `[{event_id, status, seq?}]`.
     * accepted/duplicate: it is stored, remember its seq. rejected: it never will be, drop it
     * and count it (the person is told). Anything not mentioned stays pending.
     */
    applyResults(results) {
      const byId = new Map((results || []).filter((r) => r && typeof r.event_id === 'string').map((r) => [r.event_id, r]));
      let dropped = 0;
      state.outbox = state.outbox.filter((item) => {
        const result = byId.get(item.event.event_id);
        if (!result) return true;
        if (result.status === 'rejected') {
          dropped += 1;
          return false;
        }
        if ((result.status === 'accepted' || result.status === 'duplicate') && Number.isInteger(result.seq)) {
          item.ack_seq = result.seq;
        }
        return true;
      });
      state.rejected += dropped;
      save();
      return dropped;
    },

    rejected: () => state.rejected,
    clearRejected() {
      state.rejected = 0;
      save();
    },

    /** Other devices' reviews (and this one's) after the snapshot, as `GET /api/reviews` listed them. */
    setPeers({ events, throughSeq }) {
      state.peers = { through_seq: throughSeq, events: events.slice(0, MAX_EVENTS) };
      save();
    },
    peerEvents: () => state.peers.events,

    /**
     * The computer's snapshot now includes every event up to `throughSeq`, so those stop being
     * "the snapshot plus these": drop them rather than fold them in twice.
     */
    reconcile(throughSeq) {
      const before = state.outbox.length + state.peers.events.length;
      state.outbox = state.outbox.filter((item) => item.ack_seq === null || item.ack_seq > throughSeq);
      state.peers.events = state.peers.events.filter((e) => e.seq > throughSeq);
      if (state.outbox.length + state.peers.events.length !== before) save();
    },
  };
}
