/*
 * Cloudflare Access JWT verification with WebCrypto, no dependencies.
 *
 * Access signs `Cf-Access-Jwt-Assertion` with RS256. A token is accepted only
 * when the signature verifies against one of the team's keys, the issuer is
 * this team, the audience is this application, and it has not expired. The
 * edge already refuses anonymous requests; checking here as well means a
 * misconfigured route fails closed instead of open.
 */

export class AccessError extends Error {
  constructor(code) {
    super(code);
    this.code = code;
  }
}

const decoder = new TextDecoder();

export function b64urlToBytes(text) {
  if (typeof text !== 'string' || !/^[A-Za-z0-9_-]*$/.test(text)) throw new AccessError('malformed');
  const padded = text.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (text.length % 4)) % 4);
  const raw = atob(padded);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) out[i] = raw.charCodeAt(i);
  return out;
}

function parseJson(text) {
  let value;
  try {
    value = JSON.parse(decoder.decode(b64urlToBytes(text)));
  } catch (error) {
    if (error instanceof AccessError) throw error;
    throw new AccessError('malformed');
  }
  if (value === null || typeof value !== 'object' || Array.isArray(value)) throw new AccessError('malformed');
  return value;
}

const VERIFY_ALGORITHM = { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' };

/**
 * Verify `token` and return `{kind, email|clientId, claims}`.
 * `kind` is "user" when the token carries an email, "service" when it carries a
 * service token's `common_name`. Anything else is rejected.
 * `keyFor(kid)` resolves a JWK or null. Throws AccessError with a short code.
 */
export async function verifyAccessJwt(token, { teamDomain, audience, keyFor, now = Date.now() }) {
  if (typeof token !== 'string' || token === '') throw new AccessError('missing');
  if (!teamDomain || !audience) throw new AccessError('misconfigured');
  const parts = token.split('.');
  if (parts.length !== 3) throw new AccessError('malformed');

  const header = parseJson(parts[0]);
  const claims = parseJson(parts[1]);
  if (header.alg !== 'RS256') throw new AccessError('bad-alg');
  if (typeof header.kid !== 'string') throw new AccessError('malformed');

  const jwk = await keyFor(header.kid);
  if (!jwk) throw new AccessError('unknown-key');
  const key = await crypto.subtle.importKey(
    'jwk',
    { kty: jwk.kty, n: jwk.n, e: jwk.e, alg: 'RS256', ext: true },
    VERIFY_ALGORITHM,
    false,
    ['verify'],
  );
  const signed = new TextEncoder().encode(`${parts[0]}.${parts[1]}`);
  const valid = await crypto.subtle.verify(VERIFY_ALGORITHM, key, b64urlToBytes(parts[2]), signed);
  if (!valid) throw new AccessError('bad-signature');

  const seconds = Math.floor(now / 1000);
  if (typeof claims.exp !== 'number' || claims.exp <= seconds) throw new AccessError('expired');
  if (typeof claims.nbf === 'number' && claims.nbf > seconds + 60) throw new AccessError('not-yet-valid');
  if (claims.iss !== `https://${teamDomain}`) throw new AccessError('bad-issuer');
  const audiences = Array.isArray(claims.aud) ? claims.aud : [claims.aud];
  if (!audiences.includes(audience)) throw new AccessError('bad-audience');

  if (typeof claims.email === 'string' && claims.email !== '') {
    return { kind: 'user', email: claims.email.toLowerCase(), claims };
  }
  if (typeof claims.common_name === 'string' && claims.common_name !== '') {
    return { kind: 'service', clientId: claims.common_name, claims };
  }
  throw new AccessError('no-identity');
}

function listFrom(value) {
  return String(value || '')
    .split(',')
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
}

/**
 * Who may read. Users must be on ALLOWED_EMAILS, which is required: with it
 * unset nobody is let in. A service token is refused unless its client id is on
 * ALLOWED_SERVICE_TOKENS, which is empty by default and exists only so a
 * deployment can be checked without a person's login.
 */
export function authorize(identity, env) {
  if (identity.kind === 'user') {
    const allowed = listFrom(env.ALLOWED_EMAILS);
    if (allowed.length === 0) throw new AccessError('misconfigured');
    if (!allowed.includes(identity.email)) throw new AccessError('not-allowed');
    return identity;
  }
  if (identity.kind === 'service' && listFrom(env.ALLOWED_SERVICE_TOKENS).includes(identity.clientId.toLowerCase())) {
    return identity;
  }
  throw new AccessError('not-allowed');
}
