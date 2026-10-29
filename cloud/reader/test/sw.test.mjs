// sw.js itself, run in a vm with a fake Cache API, fetch and event system: install, update,
// answering from the shell, and playing and seeking a saved episode with no network.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { test } from 'node:test';

const read = (name) => readFileSync(new URL(`../public/${name}`, import.meta.url), 'utf8');
const ORIGIN = 'https://read.example.test';
const AUDIO_PATH = `/api/audio/${'b'.repeat(64)}.mp3`;
const FILES = ['/', '/js/reader.js', '/reader.css'];

const pathOf = (input) => new URL(typeof input === 'string' ? input : input.url, ORIGIN).pathname;

/** The slice of the Cache API the worker uses. A body is kept as a blob so it can be read twice. */
function fakeCaches() {
  const stores = new Map();
  const cacheFor = (name) => {
    if (!stores.has(name)) stores.set(name, new Map());
    const map = stores.get(name);
    return {
      async put(input, res) {
        map.set(pathOf(input), { blob: await res.blob(), status: res.status, headers: [...res.headers] });
      },
      async match(input) {
        const hit = map.get(pathOf(input));
        return hit ? new Response(hit.blob, { status: hit.status, headers: hit.headers }) : undefined;
      },
    };
  };
  return {
    stores,
    open: async (name) => cacheFor(name),
    keys: async () => [...stores.keys()],
    delete: async (name) => stores.delete(name),
  };
}

/** Load sw.js the way a browser does: importScripts of the two files beside it, then events. */
function startWorker({ version = 'v1', files = FILES, network, caches = fakeCaches() }) {
  const handlers = {};
  const calls = { skipWaiting: 0, claim: 0 };
  // A browser resolves `new Request('/x')` against the worker's own address; node has no such base.
  class WorkerRequest extends Request {
    constructor(input, init) { super(typeof input === 'string' ? new URL(input, ORIGIN).href : input, init); }
  }
  const sandbox = {
    Response, Request: WorkerRequest, URL, Headers, Blob, Promise, Error, Set, Map, console,
    caches,
    fetch: (input) => network(input instanceof Request ? input : new Request(new URL(input, ORIGIN))),
    location: { origin: ORIGIN },
    clients: { claim: async () => { calls.claim += 1; } },
    skipWaiting: () => { calls.skipWaiting += 1; },
    addEventListener: (type, fn) => { handlers[type] = fn; },
  };
  sandbox.self = sandbox;
  const context = vm.createContext(sandbox);
  sandbox.importScripts = (...names) => {
    for (const name of names) {
      const text = name === '/shell.js'
        ? `self.NLP_SHELL = ${JSON.stringify({ version, files })};`
        : read(name.slice(1));
      vm.runInContext(text, context, { filename: name });
    }
  };
  vm.runInContext(read('sw.js'), context, { filename: 'sw.js' });

  /** Fire an event; returns what the worker handed to waitUntil / respondWith, or undefined when it let go. */
  const fire = async (type, init = {}) => {
    let held;
    const event = { ...init, waitUntil: (p) => { held = p; }, respondWith: (p) => { held = p; } };
    handlers[type](event);
    return held === undefined ? undefined : held;
  };
  const request = (path, { method = 'GET', mode = 'cors', headers = {} } = {}) => {
    const req = new Request(new URL(path, ORIGIN), { method, headers });
    return new Proxy(req, { get: (target, key) => (key === 'mode' ? mode : (typeof target[key] === 'function' ? target[key].bind(target) : target[key])) });
  };
  return { fire, request, caches, calls, handlers };
}

const ok = (path) => new Response(path === '/' ? '<html></html>' : `// ${path}`, {
  status: 200, headers: { 'content-type': path === '/' ? 'text/html; charset=utf-8' : path.endsWith('.css') ? 'text/css' : 'text/javascript' },
});
const servesShell = async (input) => ok(new URL(input.url).pathname);

test('install caches every shell file under the version, and activate drops older shells but never saved audio', async () => {
  const caches = fakeCaches();
  await (await caches.open('nlp-shell-old')).put('/', ok('/'));
  await (await caches.open('nlp-audio-v1')).put(AUDIO_PATH, new Response('mp3'));
  const sw = startWorker({ network: servesShell, caches });
  await sw.fire('install');
  assert.deepEqual([...caches.stores.get('nlp-shell-v1').keys()].sort(), [...FILES].sort());
  await sw.fire('activate');
  assert.deepEqual([...caches.stores.keys()].sort(), ['nlp-audio-v1', 'nlp-shell-v1']);
  assert.equal(sw.calls.claim, 1);
  assert.equal(sw.calls.skipWaiting, 0, 'a new version waits for the page to ask');
  sw.handlers.message({ data: { type: 'SKIP_WAITING' } });
  assert.equal(sw.calls.skipWaiting, 1);
  sw.handlers.message({ data: { type: 'other' } });
  assert.equal(sw.calls.skipWaiting, 1);
});

test('install fails, and leaves no half shell, when the login redirect stands in for a file', async () => {
  const caches = fakeCaches();
  const network = async (input) => {
    const path = new URL(input.url).pathname;
    if (path === '/js/reader.js') return { ok: false, status: 0, type: 'opaqueredirect', headers: new Headers(), redirected: false };
    return ok(path);
  };
  const sw = startWorker({ network, caches });
  await assert.rejects(sw.fire('install'), /reader\.js/);
  assert.equal(caches.stores.has('nlp-shell-v1'), false);
});

test('install fails when the login page comes back as a script', async () => {
  const network = async (input) => (new URL(input.url).pathname === '/reader.css'
    ? new Response('<html>Sign in</html>', { status: 200, headers: { 'content-type': 'text/html' } })
    : ok(new URL(input.url).pathname));
  const sw = startWorker({ network });
  await assert.rejects(sw.fire('install'), /reader\.css/);
});

test('page loads and shell files come from the cache with the network gone', async () => {
  const sw = startWorker({ network: servesShell });
  await sw.fire('install');
  const offline = startWorker({ network: async () => { throw new TypeError('offline'); }, caches: sw.caches });
  const page = await offline.fire('fetch', { request: offline.request('/anything?x=1', { mode: 'navigate' }) });
  assert.equal(await (await page).text(), '<html></html>');
  const script = await offline.fire('fetch', { request: offline.request('/js/reader.js') });
  assert.equal(await (await script).text(), '// /js/reader.js');
});

test('what the worker does not own is left to the browser', async () => {
  const sw = startWorker({ network: servesShell });
  for (const [path, init] of [
    ['/api/manifest', {}], ['/api/doc/library.json', {}], ['/api/reviews', { method: 'POST' }],
    ['/ingest/manifest', { method: 'PUT' }], ['/?signin=1', { mode: 'navigate' }], ['/js/not-in-shell.js', {}],
  ]) {
    assert.equal(await sw.fire('fetch', { request: sw.request(path, init) }), undefined, path);
  }
});

// ---- a saved episode ------------------------------------------------------------

const BYTES = Uint8Array.from({ length: 5000 }, (_, i) => i % 251);

async function withSavedEpisode(network) {
  const caches = fakeCaches();
  await (await caches.open('nlp-audio-v1')).put(AUDIO_PATH, new Response(new Blob([BYTES]), {
    status: 200, headers: { 'content-type': 'audio/mpeg' },
  }));
  return startWorker({ network, caches });
}

const noNetwork = async () => { throw new TypeError('offline'); };
const ask = async (sw, headers = {}, method = 'GET') => {
  const res = await sw.fire('fetch', { request: sw.request(AUDIO_PATH, { headers, method }) });
  return res;
};

test('a saved episode plays and seeks with no network: 206 with the right bytes', async () => {
  const sw = await withSavedEpisode(noNetwork);
  const whole = await ask(sw);
  const wholeResponse = await whole;
  assert.equal(wholeResponse.status, 200);
  assert.equal(wholeResponse.headers.get('accept-ranges'), 'bytes');
  assert.equal(wholeResponse.headers.get('content-type'), 'audio/mpeg');
  assert.equal(wholeResponse.headers.get('content-length'), '5000');
  assert.deepEqual(new Uint8Array(await wholeResponse.arrayBuffer()), BYTES);

  const middle = await (await ask(sw, { range: 'bytes=1000-1999' }));
  assert.equal(middle.status, 206);
  assert.equal(middle.headers.get('content-range'), 'bytes 1000-1999/5000');
  assert.equal(middle.headers.get('content-length'), '1000');
  assert.deepEqual(new Uint8Array(await middle.arrayBuffer()), BYTES.slice(1000, 2000));

  const tail = await (await ask(sw, { range: 'bytes=4900-' }));
  assert.equal(tail.headers.get('content-range'), 'bytes 4900-4999/5000');
  assert.deepEqual(new Uint8Array(await tail.arrayBuffer()), BYTES.slice(4900));

  const suffix = await (await ask(sw, { range: 'bytes=-10' }));
  assert.deepEqual(new Uint8Array(await suffix.arrayBuffer()), BYTES.slice(4990));

  const past = await (await ask(sw, { range: 'bytes=5000-' }));
  assert.equal(past.status, 416);
  assert.equal(past.headers.get('content-range'), 'bytes */5000');

  const head = await (await ask(sw, { range: 'bytes=0-9' }, 'HEAD'));
  assert.equal(head.status, 206);
  assert.equal(head.headers.get('content-length'), '10');
  assert.equal((await head.arrayBuffer()).byteLength, 0);
});

test('a saved episode is served from the device even when the network is up', async () => {
  let asked = 0;
  const sw = await withSavedEpisode(async () => { asked += 1; return new Response('from the network'); });
  const res = await ask(sw, { range: 'bytes=0-9' });
  assert.equal((await res).status, 206);
  assert.equal(asked, 0);
});

test('an episode that is not saved goes to the network, and says so when there is none', async () => {
  const seen = [];
  const sw = startWorker({ network: async (input) => { seen.push(input.headers.get('range')); return new Response('x', { status: 206 }); } });
  const res = await ask(sw, { range: 'bytes=0-9' });
  assert.equal((await res).status, 206);
  assert.deepEqual(seen, ['bytes=0-9']);

  const down = startWorker({ network: noNetwork });
  const failed = await (await ask(down));
  assert.equal(failed.status, 503);
  assert.match(await failed.text(), /not saved on this device/);
});
