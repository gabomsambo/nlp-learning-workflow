/*
 * Loads the published bundle through the Worker. The manifest says what exists
 * and what each document must hash to; a document that does not match was
 * written after the manifest we hold, so we fetch the manifest again once and
 * then say so instead of showing a mixture of two publishes.
 *
 * Every manifest and document that checks out is also kept on the device (`kv`: IndexedDB in
 * the browser). While the connection and the login are good, a kept document whose hash is the
 * manifest's is used without asking the network. When either fails, what was kept is shown
 * instead and `status().mode` says why: "offline" (the network failed) or "signed-out" (Access
 * wants a login). Nothing is ever deleted because the network failed.
 */
import { sha256Hex } from './lib.js';
import { memoryKv } from './kv.js';

export class ReaderError extends Error {
  constructor(kind, message) {
    super(message);
    this.kind = kind; // "auth" | "empty" | "network" | "stale" | "unavailable"
  }
}

async function request(path) {
  let response;
  try {
    response = await fetch(path, { credentials: 'same-origin', redirect: 'manual', cache: 'no-cache' });
  } catch (error) {
    throw new ReaderError('network', 'The reader could not be reached.');
  }
  // Behind Access an expired session answers with a redirect to the login page.
  if (response.type === 'opaqueredirect' || response.status === 401 || response.status === 403) {
    throw new ReaderError('auth', 'Your session has ended.');
  }
  if (response.status === 404) throw new ReaderError('empty', 'Nothing has been published yet.');
  if (!response.ok) throw new ReaderError('network', `The reader answered ${response.status}.`);
  return response;
}

const MANIFEST_KEY = 'meta:manifest';
const DOC_PREFIX = 'doc:';
const FAILED_CONNECTION = new Set(['network', 'auth']);

export function createStore({ kv = null } = {}) {
  const keep = kv || memoryKv();
  let manifest = null;
  let mode = 'live'; // live | offline | signed-out
  const docs = new Map();
  const listeners = new Set();

  const setMode = (next) => {
    if (next === mode) return;
    mode = next;
    for (const listener of listeners) {
      try { listener(); } catch (error) { /* a broken listener must not break loading */ }
    }
  };
  const modeFor = (error) => (error.kind === 'auth' ? 'signed-out' : 'offline');

  // The device copy is a convenience: any failure reading or writing it means "no copy".
  const saved = async (key) => { try { return await keep.get(key); } catch (error) { return undefined; } };
  const save = async (key, value) => { try { await keep.set(key, value); } catch (error) { /* full, or private window */ } };

  async function prune(current) {
    try {
      const wanted = new Set((current.docs || []).map((d) => `${DOC_PREFIX}${d.key}`));
      for (const key of await keep.keys(DOC_PREFIX)) if (!wanted.has(key)) await keep.delete(key);
    } catch (error) { /* a leftover document is harmless */ }
  }

  async function loadManifest() {
    try {
      const response = await request('/api/manifest');
      manifest = await response.json();
    } catch (error) {
      if (error instanceof ReaderError && FAILED_CONNECTION.has(error.kind)) {
        const copy = await saved(MANIFEST_KEY);
        if (copy) {
          manifest = copy;
          docs.clear();
          setMode(modeFor(error));
          return manifest;
        }
      }
      throw error;
    }
    docs.clear();
    setMode('live');
    await save(MANIFEST_KEY, manifest);
    await prune(manifest);
    return manifest;
  }

  async function ensureManifest() {
    return manifest || loadManifest();
  }

  async function fetchVerified(key) {
    const entry = (manifest.docs || []).find((d) => d.key === key);
    const response = await request(`/api/doc/${encodeURIComponent(key)}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (entry && (await sha256Hex(bytes)) !== entry.sha256) return null;
    const value = JSON.parse(new TextDecoder().decode(bytes));
    if (entry) await save(`${DOC_PREFIX}${key}`, { sha256: entry.sha256, value });
    return value;
  }

  async function fetchCurrent(key) {
    let value = await fetchVerified(key);
    if (value === null) {
      await loadManifest();
      value = await fetchVerified(key);
      if (value === null) throw new ReaderError('stale', 'A publish is in progress. Try again in a moment.');
    }
    return value;
  }

  async function doc(key) {
    await ensureManifest();
    if (docs.has(key)) return docs.get(key);
    const copy = await saved(`${DOC_PREFIX}${key}`);
    const entry = (manifest.docs || []).find((d) => d.key === key);
    let value;
    if (copy && (mode !== 'live' || (entry && copy.sha256 === entry.sha256))) {
      value = copy.value; // already verified when it was kept
    } else {
      try {
        value = await fetchCurrent(key);
        setMode('live'); // it answered, so the connection and the login are good
      } catch (error) {
        if (!(error instanceof ReaderError) || !FAILED_CONNECTION.has(error.kind)) throw error;
        setMode(modeFor(error));
        if (!copy) {
          // Not kept, and no way to get it now: say that, not "could not be reached".
          throw new ReaderError(
            error.kind === 'auth' ? 'auth' : 'unavailable',
            error.kind === 'auth' ? error.message : 'This page is not saved on this device.',
          );
        }
        value = copy.value;
      }
    }
    docs.set(key, value);
    return value;
  }

  return {
    manifest: ensureManifest,
    refresh: loadManifest,
    doc,
    library: () => doc('library.json'),
    schedule: () => doc('schedule.json'),
    /** live / offline / signed-out, and when the bundle in hand was published. */
    status: () => ({ mode, publishedAt: manifest ? manifest.generated_at || null : null }),
    onStatus(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
