import assert from 'node:assert/strict';
import { test } from 'node:test';

import { AccessError, authorize, b64urlToBytes, verifyAccessJwt } from '../src/access.js';
import { AUD, KID, NOW_MS, NOW_S, TEAM, makeKeyPair, serviceClaims, signJwt, userClaims } from './helpers.mjs';

const pair = await makeKeyPair();
const other = await makeKeyPair();
const keyFor = async (kid) => (kid === KID ? pair.jwk : null);
const options = { teamDomain: TEAM, audience: AUD, keyFor, now: NOW_MS };

async function codeOf(promise) {
  try {
    await promise;
  } catch (error) {
    assert.ok(error instanceof AccessError, `expected AccessError, got ${error}`);
    return error.code;
  }
  return 'accepted';
}

test('a valid user token is accepted and the email is lower-cased', async () => {
  const token = await signJwt(pair.privateKey, userClaims());
  const identity = await verifyAccessJwt(token, options);
  assert.equal(identity.kind, 'user');
  assert.equal(identity.email, 'reader@example.com');
});

test('the audience claim may be a single string', async () => {
  const token = await signJwt(pair.privateKey, userClaims({ aud: AUD }));
  assert.equal((await verifyAccessJwt(token, options)).kind, 'user');
});

test('an expired token is rejected', async () => {
  const token = await signJwt(pair.privateKey, userClaims({ exp: NOW_S - 1 }));
  assert.equal(await codeOf(verifyAccessJwt(token, options)), 'expired');
});

test('a token with no exp is rejected', async () => {
  const claims = userClaims();
  delete claims.exp;
  const token = await signJwt(pair.privateKey, claims);
  assert.equal(await codeOf(verifyAccessJwt(token, options)), 'expired');
});

test('a token for another application is rejected', async () => {
  const token = await signJwt(pair.privateKey, userClaims({ aud: ['b'.repeat(64)] }));
  assert.equal(await codeOf(verifyAccessJwt(token, options)), 'bad-audience');
});

test('a token from another team is rejected', async () => {
  const token = await signJwt(pair.privateKey, userClaims({ iss: 'https://evil.cloudflareaccess.com' }));
  assert.equal(await codeOf(verifyAccessJwt(token, options)), 'bad-issuer');
});

test('a missing header is rejected', async () => {
  assert.equal(await codeOf(verifyAccessJwt(null, options)), 'missing');
  assert.equal(await codeOf(verifyAccessJwt('', options)), 'missing');
});

test('garbage is rejected as malformed', async () => {
  assert.equal(await codeOf(verifyAccessJwt('not-a-jwt', options)), 'malformed');
  assert.equal(await codeOf(verifyAccessJwt('a.b.c', options)), 'malformed');
  assert.equal(await codeOf(verifyAccessJwt('a$.b.c', options)), 'malformed');
});

test('a token signed by another key is rejected', async () => {
  const token = await signJwt(other.privateKey, userClaims());
  assert.equal(await codeOf(verifyAccessJwt(token, options)), 'bad-signature');
});

test('a tampered payload is rejected', async () => {
  const token = await signJwt(pair.privateKey, userClaims({ email: 'reader@example.com' }));
  const [head, , signature] = token.split('.');
  const forged = Buffer.from(JSON.stringify(userClaims({ email: 'attacker@example.com' }))).toString('base64url');
  assert.equal(await codeOf(verifyAccessJwt(`${head}.${forged}.${signature}`, options)), 'bad-signature');
});

test('alg none and HS256 are rejected before any key is used', async () => {
  const none = await signJwt(pair.privateKey, userClaims(), { alg: 'none' });
  assert.equal(await codeOf(verifyAccessJwt(none, options)), 'bad-alg');
  const hs = await signJwt(pair.privateKey, userClaims(), { alg: 'HS256' });
  assert.equal(await codeOf(verifyAccessJwt(hs, options)), 'bad-alg');
});

test('an unknown key id is rejected', async () => {
  const token = await signJwt(pair.privateKey, userClaims(), { kid: 'rotated-away' });
  assert.equal(await codeOf(verifyAccessJwt(token, options)), 'unknown-key');
});

test('a token not valid yet is rejected', async () => {
  const token = await signJwt(pair.privateKey, userClaims({ nbf: NOW_S + 3600 }));
  assert.equal(await codeOf(verifyAccessJwt(token, options)), 'not-yet-valid');
});

test('a token with neither email nor service identity is rejected', async () => {
  const claims = userClaims();
  delete claims.email;
  const token = await signJwt(pair.privateKey, claims);
  assert.equal(await codeOf(verifyAccessJwt(token, options)), 'no-identity');
});

test('a service token verifies as a service identity', async () => {
  const token = await signJwt(pair.privateKey, serviceClaims());
  const identity = await verifyAccessJwt(token, options);
  assert.equal(identity.kind, 'service');
  assert.equal(identity.clientId, 'abc123.access');
});

test('missing configuration fails closed', async () => {
  const token = await signJwt(pair.privateKey, userClaims());
  assert.equal(await codeOf(verifyAccessJwt(token, { ...options, audience: undefined })), 'misconfigured');
});

test('base64url decoding rejects characters outside the alphabet', () => {
  assert.throws(() => b64urlToBytes('ab+/'), AccessError);
  assert.deepEqual([...b64urlToBytes('aGk')], [104, 105]);
});

test('authorize: only listed emails are let in, and the list is required', () => {
  const env = { ALLOWED_EMAILS: 'reader@example.com, other@example.com' };
  assert.equal(authorize({ kind: 'user', email: 'other@example.com' }, env).email, 'other@example.com');
  assert.throws(() => authorize({ kind: 'user', email: 'stranger@example.com' }, env), { code: 'not-allowed' });
  assert.throws(() => authorize({ kind: 'user', email: 'reader@example.com' }, {}), { code: 'misconfigured' });
});

test('authorize: a service token is refused unless it is listed', () => {
  const identity = { kind: 'service', clientId: 'abc123.access' };
  assert.throws(() => authorize(identity, { ALLOWED_EMAILS: 'reader@example.com' }), { code: 'not-allowed' });
  assert.equal(authorize(identity, { ALLOWED_SERVICE_TOKENS: 'ABC123.access' }).kind, 'service');
});
