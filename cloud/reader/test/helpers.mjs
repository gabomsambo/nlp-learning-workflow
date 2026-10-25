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

/** Just enough of an R2 bucket: get() with an If-None-Match precondition. */
export function fakeBucket(files) {
  const etagOf = (body) => `"${createHash('md5').update(body).digest('hex')}"`;
  return {
    async get(key, options = {}) {
      if (!Object.hasOwn(files, key)) return null;
      const body = files[key];
      const httpEtag = etagOf(body);
      const base = { key, size: Buffer.byteLength(body), httpEtag };
      const sent = options.onlyIf && options.onlyIf.get ? options.onlyIf.get('if-none-match') : null;
      if (sent && sent === httpEtag) return base;
      return { ...base, body: new Blob([body]).stream() };
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
