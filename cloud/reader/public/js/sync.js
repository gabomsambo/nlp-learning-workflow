/*
 * Sending reviews to the cloud and hearing about other devices'.
 *
 *   flush()           POST the outbox's pending events, 50 at a time. Safe to call at any
 *                     time and as often as you like: an event is stored once however many times
 *                     it arrives, and only one flush runs at once.
 *   pullPeers(seq)    GET every event after the snapshot's `through_seq`.
 *
 * Neither throws. Each answers `{status, ...}` with one of
 *   ok / nothing    done (nothing = there was nothing to send)
 *   offline         no connection, or the cloud failed: try again later, nothing is lost
 *   auth            the Access session ended: sign in again, nothing is lost
 *   error           the cloud refused the request itself; the events stay and a later flush retries
 * The caller decides when to try again (on open, on `online`, when the page is shown).
 */

export const BATCH = 50;
const PAGE = 500;
const MAX_PAGES = 20;

const JSON_HEADERS = { 'content-type': 'application/json', accept: 'application/json' };

/** One request, classified. Behind Access an ended session answers with a redirect to the login page. */
async function call(fetchFn, path, init = {}) {
  let response;
  try {
    response = await fetchFn(path, { credentials: 'same-origin', redirect: 'manual', cache: 'no-store', ...init });
  } catch (error) {
    return { status: 'offline' };
  }
  if (response.type === 'opaqueredirect' || response.status === 401 || response.status === 403) return { status: 'auth' };
  if (response.status >= 500 || response.status === 429) return { status: 'offline' };
  if (!response.ok) return { status: 'error', http: response.status };
  try {
    return { status: 'ok', body: await response.json() };
  } catch (error) {
    return { status: 'error', http: response.status };
  }
}

export function createSync({ outbox, fetchFn = (...args) => globalThis.fetch(...args) }) {
  let running = null;

  async function sendAll() {
    let sent = 0;
    const rejectedBefore = outbox.rejected();
    for (;;) {
      const pending = outbox.pending();
      if (pending.length === 0) break;
      const batch = pending.slice(0, BATCH);
      const answer = await call(fetchFn, '/api/reviews', {
        method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ events: batch }),
      });
      if (answer.status !== 'ok') return { ...answer, sent, rejected: outbox.rejected() - rejectedBefore };
      const results = Array.isArray(answer.body && answer.body.results) ? answer.body.results : [];
      outbox.applyResults(results);
      sent += batch.length;
      // An answer that settled none of the batch would loop forever: stop and say so.
      const still = new Set(outbox.pending().map((e) => e.event_id));
      if (batch.every((e) => still.has(e.event_id))) {
        return { status: 'error', sent, rejected: outbox.rejected() - rejectedBefore };
      }
    }
    return { status: sent ? 'ok' : 'nothing', sent, rejected: outbox.rejected() - rejectedBefore };
  }

  return {
    flush() {
      if (!running) running = sendAll().finally(() => { running = null; });
      return running;
    },

    async pullPeers(sinceSeq) {
      const events = [];
      let after = sinceSeq;
      let tip = sinceSeq;
      for (let page = 0; page < MAX_PAGES; page += 1) {
        const answer = await call(fetchFn, `/api/reviews?since=${after}&limit=${PAGE}`, { headers: { accept: 'application/json' } });
        if (answer.status !== 'ok') return { status: answer.status, count: events.length };
        const body = answer.body || {};
        if (!Array.isArray(body.events) || !Number.isInteger(body.through_seq)) return { status: 'error', count: events.length };
        events.push(...body.events);
        tip = body.through_seq;
        if (!body.more || body.events.length === 0) break;
        after = body.events[body.events.length - 1].seq;
      }
      outbox.setPeers({ events, throughSeq: tip });
      return { status: 'ok', count: events.length };
    },
  };
}
