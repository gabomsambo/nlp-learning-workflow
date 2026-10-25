/*
 * /ingest: how the desktop's publisher writes the bundle into R2.
 *
 * Nothing here is reachable by a person. A request needs a service-token JWT
 * for the ingest Access application (its own AUD, a Service Auth policy that
 * names one token), and the token's client id must be on INGEST_SERVICE_TOKENS.
 * A user session gets a 401 or 403 here whatever it carries.
 *
 * The Worker keeps the same promise the publisher makes: the manifest is last.
 * PUT /ingest/manifest is refused unless every object it lists is already in
 * the bucket with the hash it names, so a reader can never be handed a manifest
 * that points at something that is not there.
 *
 *   GET /ingest/inventory         what is stored: key, size, sha256, uploaded
 *   PUT /ingest/object/<key>      one document or one audio file
 *   PUT /ingest/manifest          the manifest, then garbage collection
 *   GET /ingest/reviews?after=N   the phone's review events, for the computer (reviews.js)
 *
 * No outbound request is made here. The bindings used are DOCS and REVIEWS.
 */

import { AccessError, authorize, verifyAccessJwt } from './access.js';
import { isDocKey } from './docs.js';
import { pull as pullReviews } from './reviews.js';

export const DEFAULT_MAX_BYTES = 5_000_000_000;
export const MAX_DOC_BYTES = 8 * 1024 * 1024;
export const MAX_AUDIO_BYTES = 40 * 1024 * 1024;
export const MAX_MANIFEST_ENTRIES = 5000;
// An object nothing references is kept this long. Tombstoned objects count from
// their deleted_at, anything else from the day it was uploaded.
export const GC_GRACE_MS = 7 * 24 * 60 * 60 * 1000;

const SHA256 = /^[0-9a-f]{64}$/;
const AUDIO_KEY = /^audio\/([0-9a-f]{64})\.mp3$/;

export class IngestError extends Error {
  constructor(status, message, extra = {}) {
    super(message);
    this.status = status;
    this.extra = extra;
  }
}

function reply(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
  });
}

export function isAudioKey(key) {
  return typeof key === 'string' && AUDIO_KEY.test(key);
}

/** The keys a publisher may write: what `cloud export` produces, nothing else. */
export function isIngestKey(key) {
  return (isDocKey(key) && key !== 'manifest.json') || isAudioKey(key);
}

/**
 * Store `buffer` under `key`, with R2 checking its SHA-256 against `sha256`.
 * R2 does the hashing, so the Worker spends no CPU on a 13 MB episode (the free
 * plan allows 10 ms per request) and a body that does not match is refused whole.
 */
async function storeVerified(env, key, buffer, sha256) {
  try {
    await env.DOCS.put(key, buffer, {
      sha256,
      httpMetadata: { contentType: contentTypeFor(key) },
      customMetadata: { sha256 },
    });
  } catch (error) {
    if (/checksum|digest|10037/i.test(String(error && error.message))) {
      throw new IngestError(400, 'body does not match x-content-sha256');
    }
    throw error;
  }
}

function looksLikeJsonObject(bytes) {
  let start = 0;
  let end = bytes.length - 1;
  while (start <= end && (bytes[start] === 0x20 || bytes[start] === 0x0a || bytes[start] === 0x0d || bytes[start] === 0x09)) start += 1;
  while (end >= start && (bytes[end] === 0x20 || bytes[end] === 0x0a || bytes[end] === 0x0d || bytes[end] === 0x09)) end -= 1;
  return start < end && bytes[start] === 0x7b && bytes[end] === 0x7d;
}

function looksLikeMp3(bytes) {
  if (bytes.length < 4) return false;
  if (bytes[0] === 0x49 && bytes[1] === 0x44 && bytes[2] === 0x33) return true; // ID3
  return bytes[0] === 0xff && (bytes[1] & 0xe0) === 0xe0; // frame sync
}

function maxBytes(env) {
  const configured = Number(env.INGEST_MAX_BYTES);
  return Number.isFinite(configured) && configured > 0 ? configured : DEFAULT_MAX_BYTES;
}

/** Every object in the bucket, one list call per thousand. */
export async function listAll(bucket) {
  const objects = [];
  let cursor;
  do {
    const page = await bucket.list({ cursor, limit: 1000, include: ['customMetadata'] });
    for (const item of page.objects) {
      objects.push({
        key: item.key,
        size: item.size,
        sha256: (item.customMetadata && item.customMetadata.sha256) || null,
        uploaded: item.uploaded instanceof Date ? item.uploaded.getTime() : Date.parse(item.uploaded),
      });
    }
    cursor = page.truncated ? page.cursor : undefined;
  } while (cursor);
  return objects;
}

function contentTypeFor(key) {
  return isAudioKey(key) ? 'audio/mpeg' : 'application/json; charset=utf-8';
}

/** Who may write. Throws AccessError; the caller maps it to a status. */
async function authenticate(request, env, { keyFor, now }) {
  if (!env.INGEST_AUD || !String(env.INGEST_SERVICE_TOKENS || '').trim()) throw new AccessError('misconfigured');
  const identity = await verifyAccessJwt(request.headers.get('cf-access-jwt-assertion'), {
    teamDomain: env.TEAM_DOMAIN,
    audience: env.INGEST_AUD,
    keyFor,
    now,
  });
  if (identity.kind !== 'service') throw new AccessError('not-allowed');
  // The reader's list parsing, fed a different variable.
  authorize(identity, { ALLOWED_SERVICE_TOKENS: env.INGEST_SERVICE_TOKENS });
  return identity;
}

async function putObject(request, env, key, listing) {
  if (!isIngestKey(key)) throw new IngestError(404, 'not a key the reader publishes');
  const claimed = (request.headers.get('x-content-sha256') || '').toLowerCase();
  if (!SHA256.test(claimed)) throw new IngestError(400, 'x-content-sha256 header is required');
  const audio = isAudioKey(key);
  if (audio && AUDIO_KEY.exec(key)[1] !== claimed) throw new IngestError(400, 'an audio key is the sha256 of its bytes');
  const limit = audio ? MAX_AUDIO_BYTES : MAX_DOC_BYTES;
  const declared = Number(request.headers.get('content-length'));
  if (Number.isFinite(declared) && declared > limit) throw new IngestError(413, `object is larger than ${limit} bytes`);

  const buffer = await request.arrayBuffer();
  if (buffer.byteLength === 0) throw new IngestError(400, 'empty body');
  if (buffer.byteLength > limit) throw new IngestError(413, `object is larger than ${limit} bytes`);
  // Shape checks only: parsing a 700 KB episode would cost more CPU than the plan allows.
  // The bytes are the publisher's, and the reader checks every document against the manifest's hash.
  if (audio) {
    if (!looksLikeMp3(new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 4)))) throw new IngestError(400, 'not an MP3');
  } else if (!looksLikeJsonObject(new Uint8Array(buffer))) {
    throw new IngestError(400, 'a document must be a JSON object');
  }

  const held = listing.find((item) => item.key === key);
  if (held && held.sha256 === claimed && held.size === buffer.byteLength) {
    return { key, sha256: claimed, bytes: buffer.byteLength, stored: false };
  }
  const total = listing.reduce((sum, item) => sum + item.size, 0) - (held ? held.size : 0) + buffer.byteLength;
  if (total > maxBytes(env)) {
    throw new IngestError(507, `the bucket would hold ${total} bytes, over the ${maxBytes(env)} byte cap`);
  }
  await storeVerified(env, key, buffer, claimed);
  return { key, sha256: claimed, bytes: buffer.byteLength, stored: true };
}

function entriesOf(manifest) {
  const problems = [];
  const wanted = new Map();
  const add = (kind, list) => {
    if (!Array.isArray(list)) {
      problems.push(`${kind} must be a list`);
      return;
    }
    for (const entry of list) {
      const ok = entry && typeof entry === 'object'
        && typeof entry.key === 'string' && SHA256.test(entry.sha256 || '')
        && Number.isInteger(entry.bytes) && entry.bytes >= 0;
      if (!ok) {
        problems.push(`${kind} has a malformed entry`);
      } else if (kind === 'docs' ? !isDocKey(entry.key) || entry.key === 'manifest.json' : !isAudioKey(entry.key)) {
        problems.push(`${kind} lists a key the reader does not serve: ${String(entry.key).slice(0, 80)}`);
      } else if (kind === 'audio' && AUDIO_KEY.exec(entry.key)[1] !== entry.sha256) {
        problems.push(`audio key ${entry.key} is not its own sha256`);
      } else {
        wanted.set(entry.key, entry);
      }
    }
  };
  add('docs', manifest.docs);
  add('audio', manifest.audio);
  return { wanted, problems };
}

/** Delete objects the manifest does not list, once they have been unreferenced for the grace period. */
export async function collectGarbage(bucket, manifest, listing, wanted, now) {
  const tombstoned = new Map();
  for (const item of Array.isArray(manifest.tombstones) ? manifest.tombstones : []) {
    const at = Date.parse(item && item.deleted_at);
    if (item && typeof item.key === 'string' && Number.isFinite(at)) tombstoned.set(item.key, at);
  }
  const doomed = listing
    .filter((item) => item.key !== 'manifest.json' && !wanted.has(item.key))
    .filter((item) => now - (tombstoned.has(item.key) ? tombstoned.get(item.key) : item.uploaded) >= GC_GRACE_MS)
    .map((item) => item.key);
  for (let i = 0; i < doomed.length; i += 1000) await bucket.delete(doomed.slice(i, i + 1000));
  return doomed;
}

async function putManifest(request, env, listing, now) {
  const buffer = await request.arrayBuffer();
  if (buffer.byteLength === 0 || buffer.byteLength > MAX_DOC_BYTES) throw new IngestError(413, 'manifest is empty or too large');
  const claimed = (request.headers.get('x-content-sha256') || '').toLowerCase();
  if (!SHA256.test(claimed)) throw new IngestError(400, 'x-content-sha256 header is required');

  let manifest;
  try {
    manifest = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(buffer));
  } catch (error) {
    throw new IngestError(400, 'manifest is not JSON');
  }
  if (manifest === null || typeof manifest !== 'object' || manifest.version !== 1) {
    throw new IngestError(400, 'manifest is not a version 1 manifest');
  }
  const { wanted, problems } = entriesOf(manifest);
  if (problems.length) throw new IngestError(400, 'manifest is not valid', { problems: problems.slice(0, 20) });
  if (wanted.size > MAX_MANIFEST_ENTRIES) throw new IngestError(400, 'manifest lists too many objects');

  const stored = new Map(listing.map((item) => [item.key, item]));
  const missing = [];
  for (const [key, entry] of wanted) {
    const held = stored.get(key);
    if (!held || held.sha256 !== entry.sha256 || held.size !== entry.bytes) missing.push(key);
  }
  if (missing.length) {
    throw new IngestError(409, 'manifest names objects that are not stored yet', { missing: missing.slice(0, 20), missing_count: missing.length });
  }
  const referenced = Array.from(wanted.values()).reduce((sum, entry) => sum + entry.bytes, 0);
  if (referenced > maxBytes(env)) throw new IngestError(507, `the bundle is ${referenced} bytes, over the ${maxBytes(env)} byte cap`);

  await storeVerified(env, 'manifest.json', buffer, claimed);
  let collected = [];
  try {
    collected = await collectGarbage(env.DOCS, manifest, listing, wanted, now);
  } catch (error) {
    // The manifest is already live. A failed sweep is retried by the next publish.
    console.warn(`gc failed: ${error && error.message}`);
  }
  return {
    sha256: claimed,
    docs: (manifest.docs || []).length,
    audio: (manifest.audio || []).length,
    deleted: collected.length,
    deleted_keys: collected.slice(0, 20),
  };
}

/**
 * Handle a request under /ingest. `keyFor` and `now` are the same injection
 * points the reader uses, so tests never touch the network.
 */
export async function handleIngest(request, env, { keyFor, now }) {
  try {
    await authenticate(request, env, { keyFor, now });
  } catch (error) {
    const code = error instanceof AccessError ? error.code : 'verify-failed';
    console.warn(`ingest denied: ${code}`);
    if (code === 'misconfigured') return reply(503, { error: 'ingest is not configured' });
    if (code === 'verify-failed') return reply(503, { error: 'sign-in could not be checked' });
    return reply(code === 'not-allowed' ? 403 : 401, { error: code === 'not-allowed' ? 'forbidden' : 'sign in required' });
  }

  const { pathname } = new URL(request.url);
  try {
    if (pathname === '/ingest/inventory') {
      if (request.method !== 'GET') return reply(405, { error: 'method not allowed' });
      const objects = await listAll(env.DOCS);
      return reply(200, {
        objects,
        bytes: objects.reduce((sum, item) => sum + item.size, 0),
        max_bytes: maxBytes(env),
      });
    }
    if (pathname === '/ingest/reviews') return await pullReviews(request, env);
    if (pathname === '/ingest/manifest') {
      if (request.method !== 'PUT') return reply(405, { error: 'method not allowed' });
      return reply(200, { ok: true, ...(await putManifest(request, env, await listAll(env.DOCS), now)) });
    }
    if (pathname.startsWith('/ingest/object/')) {
      if (request.method !== 'PUT') return reply(405, { error: 'method not allowed' });
      let key;
      try {
        key = decodeURIComponent(pathname.slice('/ingest/object/'.length));
      } catch (error) {
        throw new IngestError(404, 'not found');
      }
      const result = await putObject(request, env, key, await listAll(env.DOCS));
      return reply(result.stored ? 201 : 200, { ok: true, ...result });
    }
    return reply(404, { error: 'not found' });
  } catch (error) {
    if (error instanceof IngestError) return reply(error.status, { error: error.message, ...error.extra });
    console.error(`ingest failed: ${error && error.message}`);
    return reply(500, { error: 'ingest failed' });
  }
}
