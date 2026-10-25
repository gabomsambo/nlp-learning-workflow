import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';

import { createHandler } from '../src/worker.js';
import { GC_GRACE_MS, MAX_DOC_BYTES } from '../src/ingest.js';
import {
  INGEST_AUD, KID, NOW_MS, PUBLISHER, fakeStore, makeIngestEnv, makeKeyPair, serviceClaims, signJwt, userClaims,
} from './helpers.mjs';

const pair = await makeKeyPair();
const handle = createHandler({ keyFor: async (kid) => (kid === KID ? pair.jwk : null), now: () => NOW_MS });
const DAY = 24 * 60 * 60 * 1000;

const sha = (value) => createHash('sha256').update(value).digest('hex');
const mp3 = Buffer.concat([Buffer.from([0xff, 0xfb, 0x90, 0x44]), Buffer.from('audio-bytes')]);
const mp3Sha = sha(mp3);
const AUDIO_KEY = `audio/${mp3Sha}.mp3`;

const publisherToken = () => signJwt(pair.privateKey, serviceClaims({ aud: [INGEST_AUD], common_name: PUBLISHER }));

async function call(path, { method = 'PUT', body, token, env = makeIngestEnv(), headers = {} } = {}) {
  const init = { method, headers: { ...headers }, body };
  if (token !== undefined) init.headers['cf-access-jwt-assertion'] = token;
  return handle(new Request(`https://read.example.test${path}`, init), env);
}

async function putDoc(env, key, text) {
  const response = await call(`/ingest/object/${encodeURIComponent(key)}`, {
    env, token: await publisherToken(), body: text, headers: { 'x-content-sha256': sha(text) },
  });
  return response;
}

async function putManifest(env, body, token) {
  return call('/ingest/manifest', { env, token: token || await publisherToken(), body, headers: { 'x-content-sha256': sha(body) } });
}

function manifestFor(docs, audio = [], tombstones = []) {
  return JSON.stringify({
    version: 1,
    docs: docs.map(([key, text]) => ({ key, sha256: sha(text), bytes: Buffer.byteLength(text) })),
    audio,
    tombstones,
  });
}

test('every ingest route refuses a request with no token', async () => {
  const env = makeIngestEnv();
  for (const [method, path] of [['GET', '/ingest/inventory'], ['PUT', '/ingest/manifest'], ['PUT', '/ingest/object/library.json']]) {
    assert.equal((await call(path, { method, env, body: method === 'PUT' ? '{}' : undefined })).status, 401, path);
  }
  assert.equal(env.DOCS.objects.size, 0);
});

test('a person who is allowed to read cannot write', async () => {
  const env = makeIngestEnv();
  // Their reader token has the reader AUD, so it is not even valid here.
  const reader = await signJwt(pair.privateKey, userClaims());
  assert.equal((await call('/ingest/manifest', { env, token: reader, body: '{}' })).status, 401);
  // A user token minted for the ingest application is still refused: ingest takes service tokens only.
  const user = await signJwt(pair.privateKey, userClaims({ aud: [INGEST_AUD] }));
  assert.equal((await call('/ingest/inventory', { method: 'GET', env, token: user })).status, 403);
  assert.equal(env.DOCS.objects.size, 0);
});

test('a service token that is not the publisher is forbidden', async () => {
  const token = await signJwt(pair.privateKey, serviceClaims({ aud: [INGEST_AUD], common_name: 'someone-else.access' }));
  assert.equal((await call('/ingest/inventory', { method: 'GET', token })).status, 403);
});

test('ingest fails closed when it is not configured', async () => {
  const token = await publisherToken();
  for (const missing of ['INGEST_AUD', 'INGEST_SERVICE_TOKENS']) {
    const env = makeIngestEnv({ [missing]: '' });
    assert.equal((await call('/ingest/inventory', { method: 'GET', env, token })).status, 503, missing);
  }
});

test('the reader routes still refuse the publisher, and ingest is not a read path', async () => {
  const token = await publisherToken();
  const reader = await handle(new Request('https://read.example.test/api/manifest', { headers: { 'cf-access-jwt-assertion': token } }), makeIngestEnv());
  assert.equal(reader.status, 401); // wrong AUD for the reader
});

test('a document is stored with its hash, and the same upload again stores nothing', async () => {
  const env = makeIngestEnv();
  const doc = '{"papers":[]}\n';
  const first = await putDoc(env, 'library.json', doc);
  assert.equal(first.status, 201);
  assert.equal(env.DOCS.objects.get('library.json').customMetadata.sha256, sha(doc));
  const again = await putDoc(env, 'library.json', doc);
  assert.equal(again.status, 200);
  assert.equal((await again.json()).stored, false);
});

test('an object whose bytes do not match the claimed hash is rejected', async () => {
  const env = makeIngestEnv();
  const response = await call('/ingest/object/library.json', {
    env, token: await publisherToken(), body: '{"a":1}', headers: { 'x-content-sha256': sha('{"a":2}') },
  });
  assert.equal(response.status, 400);
  assert.equal(env.DOCS.objects.size, 0);
});

test('only keys the export produces can be written', async () => {
  const env = makeIngestEnv();
  for (const key of ['manifest.json', '../secret.json', 'papers/a/b.json', 'audio/short.mp3', 'index.html', 'papers/..json/x']) {
    const response = await putDoc(env, key, '{}');
    assert.ok([400, 404].includes(response.status), `${key} -> ${response.status}`);
  }
  assert.equal(env.DOCS.objects.size, 0);
});

test('a document must be a JSON object and an oversized one is refused', async () => {
  const env = makeIngestEnv();
  assert.equal((await putDoc(env, 'library.json', 'not json')).status, 400);
  assert.equal((await putDoc(env, 'library.json', '[1]')).status, 400);
  assert.equal((await putDoc(env, 'library.json', '  {"ok":true}\n')).status, 201);
  assert.equal((await putDoc(env, 'schedule.json', 'null')).status, 400);
  const big = JSON.stringify({ pad: 'x'.repeat(MAX_DOC_BYTES) });
  assert.equal((await putDoc(env, 'papers/big.json', big)).status, 413);
});

test('audio is stored under the hash of its bytes, and must be an MP3', async () => {
  const env = makeIngestEnv();
  const token = await publisherToken();
  const ok = await call(`/ingest/object/${encodeURIComponent(AUDIO_KEY)}`, { env, token, body: mp3, headers: { 'x-content-sha256': mp3Sha } });
  assert.equal(ok.status, 201);
  assert.equal(env.DOCS.objects.get(AUDIO_KEY).bytes.length, mp3.length);

  const other = Buffer.from('this is text, not audio');
  const wrongKey = await call(`/ingest/object/${encodeURIComponent(AUDIO_KEY)}`, { env, token, body: other, headers: { 'x-content-sha256': sha(other) } });
  assert.equal(wrongKey.status, 400);
  const notMp3Key = `audio/${sha(other)}.mp3`;
  const notMp3 = await call(`/ingest/object/${encodeURIComponent(notMp3Key)}`, { env, token, body: other, headers: { 'x-content-sha256': sha(other) } });
  assert.equal(notMp3.status, 400);
  assert.equal(env.DOCS.objects.has(notMp3Key), false);
});

test('the byte cap refuses an object that would go over it', async () => {
  const env = makeIngestEnv({ INGEST_MAX_BYTES: '100' });
  assert.equal((await putDoc(env, 'library.json', `{"a":"${'x'.repeat(60)}"}`)).status, 201);
  const refused = await putDoc(env, 'schedule.json', `{"a":"${'y'.repeat(60)}"}`);
  assert.equal(refused.status, 507);
  assert.equal(env.DOCS.objects.has('schedule.json'), false);
});

test('the manifest is refused until every object it names is stored', async () => {
  const env = makeIngestEnv();
  const token = await publisherToken();
  const lib = '{"papers":[]}\n';
  const paper = '{"id":"p1"}\n';
  await putDoc(env, 'library.json', lib);
  const body = manifestFor([['library.json', lib], ['papers/p1.json', paper]]);

  const early = await putManifest(env, body, token);
  assert.equal(early.status, 409);
  assert.deepEqual((await early.json()).missing, ['papers/p1.json']);
  assert.equal(env.DOCS.objects.has('manifest.json'), false);

  await putDoc(env, 'papers/p1.json', paper);
  const done = await putManifest(env, body, token);
  assert.equal(done.status, 200);
  assert.equal(env.DOCS.objects.get('manifest.json').customMetadata.sha256, sha(body));
});

test('a manifest naming a stale hash is refused even though the key exists', async () => {
  const env = makeIngestEnv();
  await putDoc(env, 'library.json', '{"v":1}');
  const response = await putManifest(env, manifestFor([['library.json', '{"v":2}']]));
  assert.equal(response.status, 409);
});

test('a malformed manifest is refused whole', async () => {
  const env = makeIngestEnv();
  const token = await publisherToken();
  assert.equal((await putManifest(env, 'nope', token)).status, 400);
  assert.equal((await putManifest(env, '{"version":2,"docs":[],"audio":[]}', token)).status, 400);
  const bad = JSON.stringify({ version: 1, docs: [{ key: '../x.json', sha256: 'a'.repeat(64), bytes: 1 }], audio: [] });
  const response = await putManifest(env, bad, token);
  assert.equal(response.status, 400);
  assert.ok((await response.json()).problems.length > 0);
});

test('garbage collection removes unreferenced objects only after the grace period', async () => {
  const lib = '{"papers":[]}\n';
  const store = fakeStore({
    'library.json': { body: lib, customMetadata: { sha256: sha(lib) }, uploaded: NOW_MS - DAY },
    // Deleted from the manifest eight days ago: past the grace period.
    'papers/gone.json': { body: '{}', customMetadata: { sha256: sha('{}') }, uploaded: NOW_MS - 40 * DAY },
    // Deleted from the manifest two days ago: kept, a device may still be catching up.
    'papers/recent.json': { body: '{}', customMetadata: { sha256: sha('{}') }, uploaded: NOW_MS - 40 * DAY },
    // Uploaded by an interrupted publish yesterday and not referenced yet: kept.
    'papers/inflight.json': { body: '{}', customMetadata: { sha256: sha('{}') }, uploaded: NOW_MS - DAY },
    // Never referenced, never tombstoned, old: swept.
    'papers/orphan.json': { body: '{}', customMetadata: { sha256: sha('{}') }, uploaded: NOW_MS - 9 * DAY },
  });
  const env = makeIngestEnv({ DOCS: store });
  const iso = (ms) => new Date(ms).toISOString().replace(/\.\d+Z$/, 'Z');
  const body = manifestFor([['library.json', lib]], [], [
    { key: 'papers/gone.json', deleted_at: iso(NOW_MS - GC_GRACE_MS - DAY) },
    { key: 'papers/recent.json', deleted_at: iso(NOW_MS - 2 * DAY) },
  ]);
  const response = await putManifest(env, body);
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.deepEqual(result.deleted_keys.sort(), ['papers/gone.json', 'papers/orphan.json']);
  assert.deepEqual(Array.from(store.objects.keys()).sort(), ['library.json', 'manifest.json', 'papers/inflight.json', 'papers/recent.json']);
});

test('the inventory reports keys, sizes and hashes', async () => {
  const env = makeIngestEnv();
  await putDoc(env, 'library.json', '{"a":1}');
  const response = await call('/ingest/inventory', { method: 'GET', env, token: await publisherToken() });
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.objects.length, 1);
  assert.equal(body.objects[0].sha256, sha('{"a":1}'));
  assert.equal(body.bytes, 7);
  assert.ok(body.max_bytes > 0);
});

test('methods other than the documented ones are refused', async () => {
  const token = await publisherToken();
  assert.equal((await call('/ingest/inventory', { method: 'PUT', token, body: '{}' })).status, 405);
  assert.equal((await call('/ingest/manifest', { method: 'GET', token })).status, 405);
  assert.equal((await call('/ingest/object/library.json', { method: 'DELETE', token })).status, 405);
  assert.equal((await call('/ingest/elsewhere', { method: 'GET', token })).status, 404);
});

test('a manifest without its hash header is refused, and so is one whose bytes differ from it', async () => {
  const env = makeIngestEnv();
  const token = await publisherToken();
  const body = manifestFor([]);
  assert.equal((await call('/ingest/manifest', { env, token, body })).status, 400);
  const wrong = await call('/ingest/manifest', { env, token, body, headers: { 'x-content-sha256': sha('other') } });
  assert.equal(wrong.status, 400);
  assert.equal(env.DOCS.objects.has('manifest.json'), false);
});
