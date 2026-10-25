/*
 * Loads the published bundle through the Worker. The manifest says what exists
 * and what each document must hash to; a document that does not match was
 * written after the manifest we hold, so we fetch the manifest again once and
 * then say so instead of showing a mixture of two publishes.
 */
import { sha256Hex } from './lib.js';

export class ReaderError extends Error {
  constructor(kind, message) {
    super(message);
    this.kind = kind; // "auth" | "empty" | "network" | "stale"
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

export function createStore() {
  let manifest = null;
  const docs = new Map();

  async function loadManifest() {
    const response = await request('/api/manifest');
    manifest = await response.json();
    docs.clear();
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
    return JSON.parse(new TextDecoder().decode(bytes));
  }

  async function doc(key) {
    await ensureManifest();
    if (docs.has(key)) return docs.get(key);
    let value = await fetchVerified(key);
    if (value === null) {
      await loadManifest();
      value = await fetchVerified(key);
      if (value === null) throw new ReaderError('stale', 'A publish is in progress. Try again in a moment.');
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
  };
}
