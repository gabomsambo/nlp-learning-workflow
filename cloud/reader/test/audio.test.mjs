import assert from 'node:assert/strict';
import { test } from 'node:test';

import { isAudioKey, parseRange } from '../src/audio.js';
import { createHandler } from '../src/worker.js';
import { KID, NOW_MS, fakeBucket, makeEnv, makeKeyPair, signJwt, userClaims } from './helpers.mjs';

const pair = await makeKeyPair();
const handle = createHandler({ keyFor: async (kid) => (kid === KID ? pair.jwk : null), now: () => NOW_MS });

const NAME = `${'a1'.repeat(32)}.mp3`;
const BYTES = Buffer.from(Array.from({ length: 1000 }, (_, i) => i % 251));

const envWithAudio = (files = { [`audio/${NAME}`]: BYTES }) => makeEnv({ DOCS: fakeBucket(files) });

async function ask(path, { method = 'GET', headers = {}, env = envWithAudio(), signedIn = true } = {}) {
  const init = { method, headers: { ...headers } };
  if (signedIn) init.headers['cf-access-jwt-assertion'] = await signJwt(pair.privateKey, userClaims());
  return handle(new Request(`https://read.example.test${path}`, init), env);
}

const bytesOf = async (response) => Buffer.from(await response.arrayBuffer());

test('parseRange: the shapes a browser sends', () => {
  assert.deepEqual(parseRange('bytes=0-', 1000), { start: 0, end: 999 });
  assert.deepEqual(parseRange('bytes=0-1', 1000), { start: 0, end: 1 });
  assert.deepEqual(parseRange('bytes=500-599', 1000), { start: 500, end: 599 });
  assert.deepEqual(parseRange('bytes=900-5000', 1000), { start: 900, end: 999 }, 'end is clamped to the file');
  assert.deepEqual(parseRange('bytes=-100', 1000), { start: 900, end: 999 }, 'a suffix is the last N bytes');
  assert.deepEqual(parseRange('bytes=-5000', 1000), { start: 0, end: 999 });
  assert.deepEqual(parseRange(' Bytes = 10 - 20 ', 1000), { start: 10, end: 20 });
});

test('parseRange: past the end is unsatisfiable, nonsense is ignored', () => {
  assert.equal(parseRange('bytes=1000-', 1000), 'unsatisfiable');
  assert.equal(parseRange('bytes=5000-6000', 1000), 'unsatisfiable');
  assert.equal(parseRange('bytes=-0', 1000), 'unsatisfiable');
  assert.equal(parseRange('bytes=0-', 0), 'unsatisfiable');
  for (const header of [null, undefined, '', 'bytes=', 'bytes=-', 'items=0-5', 'bytes=0-5,10-20', 'bytes=9-3', 'bytes=a-b', 'bytes=0-99999999999999999999']) {
    assert.equal(parseRange(header, 1000), null, String(header));
  }
});

test('isAudioKey only admits what cloud export writes', () => {
  assert.ok(isAudioKey(`audio/${NAME}`));
  for (const key of ['audio/abc.mp3', `audio/${NAME}.json`, `audio/../${NAME}`, `papers/${NAME}`, `audio/${'A1'.repeat(32)}.mp3`, '', null, undefined]) {
    assert.ok(!isAudioKey(key), String(key));
  }
});

test('a request with no Access token never reaches the bucket', async () => {
  const env = envWithAudio();
  const response = await ask(`/api/audio/${NAME}`, { env, signedIn: false });
  assert.equal(response.status, 401);
  assert.deepEqual(env.DOCS.reads, []);
});

test('no Range: the whole file as a 200 that says ranges are welcome', async () => {
  const response = await ask(`/api/audio/${NAME}`);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('accept-ranges'), 'bytes');
  assert.equal(response.headers.get('content-type'), 'audio/mpeg');
  assert.equal(response.headers.get('content-length'), '1000');
  assert.equal(response.headers.get('content-range'), null);
  assert.match(response.headers.get('cache-control'), /^private, max-age=\d+, immutable$/);
  assert.ok(response.headers.get('etag'));
  assert.deepEqual(await bytesOf(response), BYTES);
});

test('Range: bytes=0-1 is the probe Safari sends first: 206 with two bytes', async () => {
  const response = await ask(`/api/audio/${NAME}`, { headers: { range: 'bytes=0-1' } });
  assert.equal(response.status, 206);
  assert.equal(response.headers.get('content-range'), 'bytes 0-1/1000');
  assert.equal(response.headers.get('content-length'), '2');
  assert.equal(response.headers.get('accept-ranges'), 'bytes');
  assert.deepEqual(await bytesOf(response), BYTES.subarray(0, 2));
});

test('a seek: a middle range, an open-ended range and a suffix return exactly those bytes', async () => {
  const cases = [
    ['bytes=250-499', 250, 499],
    ['bytes=600-', 600, 999],
    ['bytes=-40', 960, 999],
    ['bytes=990-9999', 990, 999],
  ];
  for (const [header, start, end] of cases) {
    const response = await ask(`/api/audio/${NAME}`, { headers: { range: header } });
    assert.equal(response.status, 206, header);
    assert.equal(response.headers.get('content-range'), `bytes ${start}-${end}/1000`, header);
    assert.equal(response.headers.get('content-length'), String(end - start + 1), header);
    assert.deepEqual(await bytesOf(response), BYTES.subarray(start, end + 1), header);
  }
});

test('only the requested bytes are read from the bucket', async () => {
  const env = envWithAudio();
  await ask(`/api/audio/${NAME}`, { env, headers: { range: 'bytes=100-199' } });
  assert.deepEqual(env.DOCS.reads.at(-1), ['get', `audio/${NAME}`, { offset: 100, length: 100 }]);
});

test('a range past the end is 416 and says how long the file is', async () => {
  const response = await ask(`/api/audio/${NAME}`, { headers: { range: 'bytes=1000-' } });
  assert.equal(response.status, 416);
  assert.equal(response.headers.get('content-range'), 'bytes */1000');
});

test('a Range header the server cannot read is ignored, not refused', async () => {
  for (const header of ['bytes=0-5,10-20', 'bytes=banana', 'items=0-5']) {
    const response = await ask(`/api/audio/${NAME}`, { headers: { range: header } });
    assert.equal(response.status, 200, header);
    assert.equal((await bytesOf(response)).length, 1000);
  }
});

test('If-Range: a range is honoured only while the client holds this exact object', async () => {
  const etag = (await ask(`/api/audio/${NAME}`, { method: 'HEAD' })).headers.get('etag');
  const same = await ask(`/api/audio/${NAME}`, { headers: { range: 'bytes=0-9', 'if-range': etag } });
  assert.equal(same.status, 206);
  const other = await ask(`/api/audio/${NAME}`, { headers: { range: 'bytes=0-9', 'if-range': '"something-else"' } });
  assert.equal(other.status, 200);
  assert.equal((await bytesOf(other)).length, 1000);
});

test('If-None-Match answers 304 without reading the body', async () => {
  const env = envWithAudio();
  const etag = (await ask(`/api/audio/${NAME}`, { env, method: 'HEAD' })).headers.get('etag');
  env.DOCS.reads.length = 0;
  const response = await ask(`/api/audio/${NAME}`, { env, headers: { 'if-none-match': etag } });
  assert.equal(response.status, 304);
  assert.equal(await response.text(), '');
  assert.ok(env.DOCS.reads.every(([op]) => op === 'head'));
});

test('HEAD reports the length and the status a GET would, with no body', async () => {
  const whole = await ask(`/api/audio/${NAME}`, { method: 'HEAD' });
  assert.equal(whole.status, 200);
  assert.equal(whole.headers.get('content-length'), '1000');
  assert.equal(whole.headers.get('accept-ranges'), 'bytes');
  assert.equal(await whole.text(), '');
  const part = await ask(`/api/audio/${NAME}`, { method: 'HEAD', headers: { range: 'bytes=10-19' } });
  assert.equal(part.status, 206);
  assert.equal(part.headers.get('content-range'), 'bytes 10-19/1000');
});

test('only audio/<sha256>.mp3 is reachable, and a missing file is 404', async () => {
  const files = { [`audio/${NAME}`]: BYTES, 'audio/notes.mp3': 'x', 'manifest.json': '{}' };
  const env = envWithAudio(files);
  for (const path of ['/api/audio/notes.mp3', '/api/audio/', '/api/audio/..%2Fmanifest.json', `/api/audio/${'b2'.repeat(32)}.mp3`, `/api/audio/${NAME}.bak`, `/api/audio/audio/${NAME}`]) {
    assert.equal((await ask(path, { env })).status, 404, path);
  }
});

test('audio is read-only like everything else, and the page may play audio from itself', async () => {
  assert.equal((await ask(`/api/audio/${NAME}`, { method: 'POST' })).status, 405);
  const response = await ask(`/api/audio/${NAME}`);
  assert.match(response.headers.get('content-security-policy'), /media-src 'self'/);
  assert.equal(response.headers.get('x-content-type-options'), 'nosniff');
});
