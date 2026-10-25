import assert from 'node:assert/strict';
import { test } from 'node:test';

import { certsUrl, createKeyStore } from '../src/certs.js';
import { TEAM } from './helpers.mjs';

function stubFetch(keys, counter) {
  return async (url) => {
    counter.urls.push(url);
    return new Response(JSON.stringify({ keys }), { status: 200 });
  };
}

test('the certs URL is built only for a team domain', () => {
  assert.equal(certsUrl(TEAM), `https://${TEAM}/cdn-cgi/access/certs`);
  for (const bad of ['evil.example.com', 'x.cloudflareaccess.com.evil.test', 'x.cloudflareaccess.com/path', '', undefined, '-.cloudflareaccess.com']) {
    assert.throws(() => certsUrl(bad), /TEAM_DOMAIN/, String(bad));
  }
});

test('keys are fetched once and then served from memory', async () => {
  const counter = { urls: [] };
  const store = createKeyStore(TEAM, stubFetch([{ kid: 'k1', kty: 'RSA', n: 'n', e: 'AQAB' }], counter));
  assert.equal((await store.keyFor('k1', 1000)).kid, 'k1');
  assert.equal((await store.keyFor('k1', 2000)).kid, 'k1');
  assert.equal(counter.urls.length, 1);
  assert.equal(counter.urls[0], `https://${TEAM}/cdn-cgi/access/certs`);
});

test('an unknown key id refetches at most once a minute', async () => {
  const counter = { urls: [] };
  const store = createKeyStore(TEAM, stubFetch([{ kid: 'k1', kty: 'RSA' }], counter));
  await store.keyFor('k1', 0);
  assert.equal(await store.keyFor('nope', 10_000), null);
  assert.equal(counter.urls.length, 1);
  assert.equal(await store.keyFor('nope', 61_000), null);
  assert.equal(counter.urls.length, 2);
});

test('keys older than an hour are fetched again', async () => {
  const counter = { urls: [] };
  const store = createKeyStore(TEAM, stubFetch([{ kid: 'k1', kty: 'RSA' }], counter));
  await store.keyFor('k1', 0);
  await store.keyFor('k1', 61 * 60 * 1000);
  assert.equal(counter.urls.length, 2);
});

test('a failing certs endpoint is an error, never an accepted token', async () => {
  const store = createKeyStore(TEAM, async () => new Response('nope', { status: 500 }));
  await assert.rejects(store.keyFor('k1'), /500/);
});
