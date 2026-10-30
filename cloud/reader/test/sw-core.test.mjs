// The service worker's pure rules: byte ranges, which requests it answers, cache names,
// and the shell version. Range parsing is held to the Worker's own (src/audio.js): the same
// header must mean the same bytes whether the network or the saved copy answers.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { test } from 'node:test';

import { parseRange as workerParseRange } from '../src/audio.js';
import { buildShell, shellScript } from '../scripts/build-shell.mjs';

const require = createRequire(import.meta.url);
const core = require('../public/sw-core.js');

const SIZE = 1000;
const HEADERS = [
  'bytes=0-0', 'bytes=0-', 'bytes=0-999', 'bytes=0-5000', 'bytes=500-', 'bytes=999-', 'bytes=1000-', 'bytes=2000-3000',
  'bytes=-100', 'bytes=-5000', 'bytes=-0', 'bytes=-', 'bytes=5-2', 'bytes=0-1,5-9', 'bytes = 10 - 20', 'BYTES=10-20',
  'items=0-5', 'bytes=a-b', '', 'bytes=9007199254740993-', undefined, null,
];

test('byte ranges mean what the Worker means, for every header it knows how to read', () => {
  for (const header of HEADERS) {
    assert.deepEqual(core.parseRange(header, SIZE), workerParseRange(header, SIZE), String(header));
  }
  assert.deepEqual(core.parseRange('bytes=0-5', 0), workerParseRange('bytes=0-5', 0));
  assert.deepEqual(core.parseRange('bytes=-5', 0), workerParseRange('bytes=-5', 0));
});

test('a range answer carries the status, length and Content-Range an audio element needs to seek', () => {
  assert.deepEqual(core.rangeAnswer(1000, 'bytes=100-199'), {
    status: 206, start: 100, end: 199, length: 100,
    headers: { 'accept-ranges': 'bytes', 'content-type': 'audio/mpeg', 'content-length': '100', 'content-range': 'bytes 100-199/1000' },
  });
  const open = core.rangeAnswer(1000, 'bytes=900-');
  assert.equal(open.status, 206);
  assert.equal(open.headers['content-range'], 'bytes 900-999/1000');
  const suffix = core.rangeAnswer(1000, 'bytes=-10');
  assert.equal(suffix.headers['content-range'], 'bytes 990-999/1000');
  // No Range header, or one that is ignored: the whole file, as a plain 200 that still advertises ranges.
  for (const header of [null, 'garbage', 'bytes=0-1,5-9']) {
    const whole = core.rangeAnswer(1000, header);
    assert.equal(whole.status, 200);
    assert.equal(whole.headers['content-length'], '1000');
    assert.equal(whole.headers['accept-ranges'], 'bytes');
    assert.equal(whole.headers['content-range'], undefined);
  }
  const past = core.rangeAnswer(1000, 'bytes=1000-');
  assert.equal(past.status, 416);
  assert.equal(past.headers['content-range'], 'bytes */1000');
  assert.equal(past.length, 0);
});

const req = (path, { method = 'GET', mode = 'cors', origin = 'https://read.example.test' } = {}) => ({ url: new URL(path, origin), method, mode });
const SHELL = new Set(['/', '/js/reader.js', '/reader.css']);
const ORIGIN = 'https://read.example.test';
const AUDIO = `/api/audio/${'a'.repeat(64)}.mp3`;

test('the worker answers saved audio, page loads and shell files, and nothing else', () => {
  assert.equal(core.classify(req(AUDIO), ORIGIN, SHELL), 'audio');
  assert.equal(core.classify(req(AUDIO, { method: 'HEAD' }), ORIGIN, SHELL), 'audio');
  assert.equal(core.classify(req(AUDIO, { method: 'POST' }), ORIGIN, SHELL), 'pass');
  assert.equal(core.classify(req('/api/audio/short.mp3'), ORIGIN, SHELL), 'pass');
  assert.equal(core.classify(req('/', { mode: 'navigate' }), ORIGIN, SHELL), 'navigate');
  assert.equal(core.classify(req('/some/other/page?x=1', { mode: 'navigate' }), ORIGIN, SHELL), 'navigate');
  assert.equal(core.classify(req('/js/reader.js'), ORIGIN, SHELL), 'shell');
  assert.equal(core.classify(req('/js/unknown.js'), ORIGIN, SHELL), 'pass');
  // Documents, reviews and the publisher's door are never the worker's business.
  for (const path of ['/api/manifest', '/api/doc/library.json', '/api/reviews', '/api/me', '/ingest/manifest']) {
    assert.equal(core.classify(req(path), ORIGIN, SHELL), 'pass', path);
  }
  assert.equal(core.classify(req('/api/reviews', { method: 'POST' }), ORIGIN, SHELL), 'pass');
  assert.equal(core.classify(req('/js/reader.js', { method: 'POST' }), ORIGIN, SHELL), 'pass');
  assert.equal(core.classify(req('/js/reader.js', { origin: 'https://elsewhere.test' }), ORIGIN, SHELL), 'pass');
});

test('"sign in again" is a navigation the worker leaves alone, so it can reach Access', () => {
  assert.equal(core.classify(req('/?signin=1', { mode: 'navigate' }), ORIGIN, SHELL), 'pass');
  // Only a navigation: a stray ?signin on a script request is still the shell's.
  assert.equal(core.classify(req('/js/reader.js?signin=1'), ORIGIN, SHELL), 'shell');
});

test('only the previous shells are stale, never the saved audio', () => {
  const names = ['nlp-shell-aaa', 'nlp-shell-bbb', core.AUDIO_CACHE, 'something-else'];
  assert.deepEqual(core.staleShellCaches(names, 'bbb'), ['nlp-shell-aaa']);
  assert.deepEqual(core.staleShellCaches(names, 'zzz'), ['nlp-shell-aaa', 'nlp-shell-bbb']);
  assert.equal(core.shellCacheName('v1'), 'nlp-shell-v1');
});

const response = (type, { ok = true, redirected = false, status = 200, redirect = false } = {}) => ({
  ok, redirected, status, type: redirect ? 'opaqueredirect' : 'basic', headers: new Headers({ 'content-type': type }),
});

test('a login redirect or login page is never accepted as app code', () => {
  assert.equal(core.usableShellResponse('/js/reader.js', response('text/javascript')), true);
  assert.equal(core.usableShellResponse('/', response('text/html; charset=utf-8')), true);
  assert.equal(core.usableShellResponse('/js/reader.js', response('text/javascript', { redirect: true, ok: false, status: 0 })), false);
  assert.equal(core.usableShellResponse('/js/reader.js', response('text/javascript', { redirected: true })), false);
  assert.equal(core.usableShellResponse('/js/reader.js', response('text/html')), false); // the Access page at 200
  assert.equal(core.usableShellResponse('/', response('application/json')), false);
  assert.equal(core.usableShellResponse('/js/reader.js', response('text/javascript', { ok: false, status: 500 })), false);
  assert.equal(core.usableShellResponse('/js/reader.js', null), false);
});

// ---- the shell version -----------------------------------------------------------

const file = (path, text) => ({ path, bytes: Buffer.from(text) });
const SITE = [
  file('index.html', '<html>'), file('js/reader.js', 'a'), file('reader.css', 'b'), file('sw.js', 'worker'),
  file('sw-core.js', 'core'), file('shell.js', 'old'), file('static/.DS_Store', 'junk'), file('manifest.webmanifest', '{}'),
];

test('the shell lists what the page loads, with the page itself at "/", and not the worker', () => {
  const shell = buildShell(SITE);
  assert.deepEqual(shell.files, ['/', '/js/reader.js', '/manifest.webmanifest', '/reader.css']);
  assert.match(shell.version, /^[0-9a-f]{12}$/);
  assert.match(shellScript(shell), /^\/\/ Generated.*\nself\.NLP_SHELL = \{"version":"[0-9a-f]{12}"/);
});

test('the version changes when any file does, and only then', () => {
  const base = buildShell(SITE).version;
  assert.equal(buildShell([...SITE].reverse()).version, base, 'order does not matter');
  assert.equal(buildShell(SITE.map((f) => (f.path === 'sw.js' ? file('sw.js', 'changed') : f))).version, base, 'the worker is not part of its own shell');
  assert.notEqual(buildShell(SITE.map((f) => (f.path === 'js/reader.js' ? file('js/reader.js', 'a2') : f))).version, base);
  assert.notEqual(buildShell([...SITE, file('js/new.js', 'x')]).version, base);
  assert.notEqual(buildShell(SITE.filter((f) => f.path !== 'reader.css')).version, base);
});
