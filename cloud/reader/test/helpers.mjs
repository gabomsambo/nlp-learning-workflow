// Test fixtures: a throwaway RSA key, a JWT signer, and fakes for R2 and the assets binding.
import { createHash } from 'node:crypto';

export const TEAM = 'soft-wave-31cb.cloudflareaccess.com';
export const AUD = 'a'.repeat(64);
export const KID = 'test-key-1';
export const NOW_MS = Date.UTC(2026, 9, 25, 12, 0, 0);
export const NOW_S = Math.floor(NOW_MS / 1000);

const SIGN = { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' };

export async function makeKeyPair() {
  const pair = await crypto.subtle.generateKey(
    { ...SIGN, modulusLength: 2048, publicExponent: new Uint8Array([1, 0, 1]) },
    true,
    ['sign', 'verify'],
  );
  const jwk = await crypto.subtle.exportKey('jwk', pair.publicKey);
  return { privateKey: pair.privateKey, jwk: { ...jwk, kid: KID, alg: 'RS256', use: 'sig' } };
}

function b64url(bytes) {
  return Buffer.from(bytes).toString('base64url');
}

const text = (value) => new TextEncoder().encode(typeof value === 'string' ? value : JSON.stringify(value));

export async function signJwt(privateKey, claims, header = {}) {
  const head = { alg: 'RS256', kid: KID, typ: 'JWT', ...header };
  const signingInput = `${b64url(text(head))}.${b64url(text(claims))}`;
  const signature = await crypto.subtle.sign(SIGN, privateKey, text(signingInput));
  return `${signingInput}.${b64url(new Uint8Array(signature))}`;
}

export function userClaims(overrides = {}) {
  return {
    aud: [AUD],
    iss: `https://${TEAM}`,
    email: 'Reader@Example.com',
    exp: NOW_S + 3600,
    iat: NOW_S - 10,
    nbf: NOW_S - 10,
    type: 'app',
    ...overrides,
  };
}

export function serviceClaims(overrides = {}) {
  const claims = userClaims({ common_name: 'abc123.access', ...overrides });
  delete claims.email;
  return { ...claims, ...overrides };
}

/**
 * Just enough of an R2 bucket: get() with an If-None-Match precondition or a
 * byte range, and head(). `reads` records every call so a test can say what was
 * (not) touched.
 */
export function fakeBucket(files) {
  const etagOf = (body) => `"${createHash('md5').update(body).digest('hex')}"`;
  const reads = [];
  return {
    reads,
    async head(key) {
      reads.push(['head', key]);
      if (!Object.hasOwn(files, key)) return null;
      return { key, size: Buffer.byteLength(files[key]), httpEtag: etagOf(files[key]) };
    },
    async get(key, options = {}) {
      reads.push(['get', key, options.range || null]);
      if (!Object.hasOwn(files, key)) return null;
      const whole = Buffer.from(files[key]);
      const httpEtag = etagOf(whole);
      const base = { key, size: whole.length, httpEtag };
      const sent = options.onlyIf && options.onlyIf.get ? options.onlyIf.get('if-none-match') : null;
      if (sent && sent === httpEtag) return base;
      const range = options.range;
      const body = range ? whole.subarray(range.offset, range.offset + range.length) : whole;
      return { ...base, body: new Blob([body]).stream(), range: range || undefined };
    },
  };
}

export function fakeAssets(files = { '/index.html': '<!doctype html><title>reader</title>' }) {
  return {
    async fetch(request) {
      const path = new URL(request.url).pathname;
      const hit = files[path === '/' ? '/index.html' : path];
      if (hit === undefined) return new Response('not found', { status: 404 });
      return new Response(hit, { headers: { 'content-type': 'text/html', 'cache-control': 'public, max-age=0' } });
    },
  };
}

export function makeEnv(overrides = {}) {
  return {
    TEAM_DOMAIN: TEAM,
    ACCESS_AUD: AUD,
    ALLOWED_EMAILS: 'reader@example.com, other@example.com',
    DOCS: fakeBucket({ 'manifest.json': '{"version":1}\n', 'library.json': '{"papers":[]}\n' }),
    ASSETS: fakeAssets(),
    ...overrides,
  };
}

export const INGEST_AUD = 'c'.repeat(64);
export const PUBLISHER = 'publisher123.access';

/** A writable R2 bucket: get, put, list (with custom metadata), delete. */
export function fakeStore(initial = {}) {
  const objects = new Map();
  const put = (key, body, options = {}, uploaded = Date.UTC(2026, 9, 25, 12, 0, 0)) => {
    const bytes = typeof body === 'string' ? new TextEncoder().encode(body) : new Uint8Array(body);
    objects.set(key, { bytes, customMetadata: options.customMetadata || {}, uploaded: new Date(uploaded) });
  };
  for (const [key, value] of Object.entries(initial)) put(key, value.body, { customMetadata: value.customMetadata }, value.uploaded);
  return {
    objects,
    seed: put,
    async put(key, body, options = {}) {
      // R2 checks the checksum it is given and refuses a body that does not match.
      if (options.sha256 && createHash('sha256').update(Buffer.from(body)).digest('hex') !== options.sha256) {
        throw new Error('put: The SHA-256 checksum you specified did not match what we received. (10037)');
      }
      put(key, body, options, NOW_MS);
    },
    async get(key) {
      const held = objects.get(key);
      if (!held) return null;
      return { key, size: held.bytes.length, httpEtag: `"${key}"`, body: new Blob([held.bytes]).stream() };
    },
    async list({ cursor, limit = 1000 } = {}) {
      const keys = Array.from(objects.keys()).sort();
      const start = cursor ? Number(cursor) : 0;
      const page = keys.slice(start, start + limit);
      return {
        objects: page.map((key) => ({
          key, size: objects.get(key).bytes.length, uploaded: objects.get(key).uploaded,
          customMetadata: objects.get(key).customMetadata,
        })),
        truncated: start + limit < keys.length,
        cursor: String(start + limit),
      };
    },
    async delete(keys) { for (const key of [].concat(keys)) objects.delete(key); },
  };
}

export function makeIngestEnv(overrides = {}) {
  return makeEnv({
    INGEST_AUD,
    INGEST_SERVICE_TOKENS: PUBLISHER,
    DOCS: fakeStore(),
    ...overrides,
  });
}
