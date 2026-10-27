import assert from 'node:assert/strict';
import { test } from 'node:test';

import { createHandler } from '../src/worker.js';
import { isDocKey } from '../src/docs.js';
import { isIngestKey } from '../src/ingest.js';
import { KID, NOW_MS, fakeBucket, makeEnv, makeKeyPair, serviceClaims, signJwt, userClaims } from './helpers.mjs';

const pair = await makeKeyPair();
const handle = createHandler({ keyFor: async (kid) => (kid === KID ? pair.jwk : null), now: () => NOW_MS });

async function get(path, { token, env = makeEnv(), method = 'GET', headers = {} } = {}) {
  const init = { method, headers: { ...headers } };
  if (token) init.headers['cf-access-jwt-assertion'] = token;
  return handle(new Request(`https://read.example.test${path}`, init), env);
}

const userToken = () => signJwt(pair.privateKey, userClaims());

test('a request with no Access token is refused before anything is read', async () => {
  let touched = 0;
  const env = makeEnv({
    DOCS: { get: async () => { touched += 1; return null; } },
    ASSETS: { fetch: async () => { touched += 1; return new Response('x'); } },
  });
  for (const path of ['/', '/api/manifest', '/api/doc/library.json', '/js/reader.js']) {
    const response = await get(path, { env });
    assert.equal(response.status, 401, path);
  }
  assert.equal(touched, 0);
});

test('an expired token is refused', async () => {
  const token = await signJwt(pair.privateKey, userClaims({ exp: Math.floor(NOW_MS / 1000) - 5 }));
  assert.equal((await get('/api/manifest', { token })).status, 401);
});

test('a token for another application is refused', async () => {
  const token = await signJwt(pair.privateKey, userClaims({ aud: ['b'.repeat(64)] }));
  assert.equal((await get('/', { token })).status, 401);
});

test('a service token is forbidden unless it is allow-listed', async () => {
  const token = await signJwt(pair.privateKey, serviceClaims());
  assert.equal((await get('/api/manifest', { token })).status, 403);
  const env = makeEnv({ ALLOWED_SERVICE_TOKENS: 'abc123.access' });
  assert.equal((await get('/api/manifest', { token, env })).status, 200);
});

test('a valid token for an unlisted email is forbidden', async () => {
  const token = await signJwt(pair.privateKey, userClaims({ email: 'stranger@example.com' }));
  assert.equal((await get('/api/manifest', { token })).status, 403);
});

test('with ALLOWED_EMAILS unset the reader answers 503, not open', async () => {
  const env = makeEnv({ ALLOWED_EMAILS: '' });
  assert.equal((await get('/api/manifest', { token: await userToken(), env })).status, 503);
});

test('the manifest is served with an ETag, and revalidates to 304', async () => {
  const token = await userToken();
  const first = await get('/api/manifest', { token });
  assert.equal(first.status, 200);
  assert.equal(await first.text(), '{"version":1}\n');
  assert.match(first.headers.get('content-type'), /^application\/json/);
  assert.equal(first.headers.get('cache-control'), 'private, no-cache');
  const etag = first.headers.get('etag');
  assert.ok(etag);

  const second = await get('/api/manifest', { token, headers: { 'if-none-match': etag } });
  assert.equal(second.status, 304);
  assert.equal(second.headers.get('etag'), etag);
  assert.equal(await second.text(), '');
});

test('a missing manifest is a 404 that says nothing is published', async () => {
  const env = makeEnv({ DOCS: fakeBucket({}) });
  const response = await get('/api/manifest', { token: await userToken(), env });
  assert.equal(response.status, 404);
  assert.match((await response.json()).error, /nothing published/);
});

test('documents are served by key, including an encoded path', async () => {
  const env = makeEnv({
    DOCS: fakeBucket({ 'papers/arxiv_3a1706.03762.json': '{"id":"arxiv:1706.03762"}\n' }),
  });
  const token = await userToken();
  const response = await get('/api/doc/papers%2Farxiv_3a1706.03762.json', { token, env });
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { id: 'arxiv:1706.03762' });
  assert.equal((await get('/api/doc/papers/arxiv_3a1706.03762.json', { token, env })).status, 200);
});

test('only published document names are reachable', async () => {
  const env = makeEnv({
    DOCS: fakeBucket({
      'audio/abc.mp3': 'bytes',
      'secrets.json': '{}',
      'papers/x.txt': 'x',
      'papers/../secrets.json': '{}',
    }),
  });
  const token = await userToken();
  for (const key of ['audio/abc.mp3', 'secrets.json', 'papers/x.txt', 'papers/../secrets.json', 'papers/', '..%2Fmanifest.json', '%']) {
    const response = await get(`/api/doc/${key}`, { token, env });
    assert.equal(response.status, 404, key);
  }
});

test('a playground is served by its uuid name and nothing looser', async () => {
  const id = 'abcdef12-3456-4789-8abc-def012345678';
  const env = makeEnv({
    DOCS: fakeBucket({ [`playgrounds/${id}.json`]: '{"id":"pg"}\n', 'playgrounds/notes.json': '{}' }),
  });
  const token = await userToken();
  assert.equal((await get(`/api/doc/playgrounds%2F${id}.json`, { token, env })).status, 200);
  assert.ok(isDocKey(`playgrounds/${id}.json`) && isIngestKey(`playgrounds/${id}.json`));
  for (const key of ['playgrounds/notes.json', `playgrounds/${id.toUpperCase()}.json`, `playgrounds/${id}.json.bak`, `playgrounds/../${id}.json`]) {
    assert.equal(isDocKey(key), false, key);
  }
  assert.equal((await get('/api/doc/playgrounds%2Fnotes.json', { token, env })).status, 404);
});

test('if the Access keys cannot be fetched the answer is 503, never an open door', async () => {
  const broken = createHandler({ keyFor: async () => { throw new Error('certs endpoint answered 500'); }, now: () => NOW_MS });
  const request = new Request('https://read.example.test/api/manifest', { headers: { 'cf-access-jwt-assertion': await userToken() } });
  const response = await broken(request, makeEnv());
  assert.equal(response.status, 503);
  assert.doesNotMatch(await response.text(), /certs/);
});

test('writes are refused', async () => {
  for (const method of ['POST', 'PUT', 'DELETE', 'PATCH']) {
    const response = await get('/api/manifest', { token: await userToken(), method });
    assert.equal(response.status, 405, method);
  }
});

test('unknown API paths are 404 and /api/me reports the caller', async () => {
  const token = await userToken();
  assert.equal((await get('/api/ingest', { token })).status, 404);
  const me = await (await get('/api/me', { token })).json();
  assert.deepEqual(me, { kind: 'user', email: 'reader@example.com' });
});

test('everything else is the app shell, with security headers and no shared caching', async () => {
  const response = await get('/', { token: await userToken() });
  assert.equal(response.status, 200);
  assert.match(await response.text(), /<title>reader<\/title>/);
  assert.match(response.headers.get('content-security-policy'), /default-src 'none'/);
  assert.match(response.headers.get('content-security-policy'), /frame-ancestors 'none'/);
  assert.equal(response.headers.get('x-content-type-options'), 'nosniff');
  assert.equal(response.headers.get('cache-control'), 'private, no-cache');
});

test('the Worker has no outbound path beyond the Access key store', async () => {
  // Static check on the source, the same guard PR 8 turns into CI.
  const { readFileSync, readdirSync } = await import('node:fs');
  const dir = new URL('../src/', import.meta.url);
  const calls = [];
  for (const name of readdirSync(dir)) {
    const source = readFileSync(new URL(name, dir), 'utf8');
    for (const match of source.matchAll(/\bfetch\s*\(/g)) calls.push(`${name}:${source.slice(0, match.index).split('\n').length}`);
  }
  // worker.js: env.ASSETS.fetch; certs.js: the key endpoint via the injected fetchImpl.
  for (const call of calls) assert.match(call, /^(worker|certs)\.js:/, call);
  const wrangler = readFileSync(new URL('../wrangler.toml', import.meta.url), 'utf8');
  assert.doesNotMatch(wrangler, /^\[ai\]|^\[\[(vectorize|queues|services|kv_namespaces)/m);
  // The one database is the review mailbox (reviews.js), and the bindings are exactly DOCS, REVIEWS and ASSETS.
  assert.equal((wrangler.match(/^\[\[d1_databases\]\]/gm) || []).length, 1);
  assert.match(wrangler, /\[\[d1_databases\]\]\s*\nbinding = "REVIEWS"/);
  assert.deepEqual([...wrangler.matchAll(/^binding = "(\w+)"/gm)].map((m) => m[1]).sort(), ['ASSETS', 'DOCS', 'REVIEWS']);
  assert.match(wrangler, /workers_dev = false/);
});
