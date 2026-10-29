/*
 * The reader's service worker: the app opens with no connection, and an episode saved on
 * the phone plays and seeks with none.
 *
 *   shell    every file the page loads is precached under a version (shell.js, written by
 *            scripts/build-shell.mjs from the file contents) and answered from there. A new
 *            version installs beside the old one and waits; the page offers "Reload to update"
 *            so code never changes under an open episode.
 *   audio    /api/audio/<sha256>.mp3 comes from the saved copy when there is one, with Range
 *            answered from it (an <audio> element cannot seek without 206). Otherwise the
 *            network, as if there were no worker.
 *   the rest documents, reviews and /ingest are not touched here. The page keeps documents in
 *            IndexedDB, where it can check each against the manifest's hash.
 *
 * Nothing is cached from a response that was not the file asked for: an ended Access session
 * answers with a redirect or the login page, and installing that as the app would lock the
 * phone out of it. See sw-core.js for the rules.
 */
/* global NLPSwCore, NLP_SHELL */
importScripts('/sw-core.js', '/shell.js');

const core = self.NLPSwCore;
const SHELL = self.NLP_SHELL;
const SHELL_CACHE = core.shellCacheName(SHELL.version);
const SHELL_PATHS = new Set(SHELL.files);

async function precache() {
  const cache = await caches.open(SHELL_CACHE);
  try {
    await Promise.all(SHELL.files.map(async (path) => {
      const response = await fetch(new Request(path, { cache: 'reload', credentials: 'same-origin', redirect: 'manual' }));
      if (!core.usableShellResponse(path, response)) throw new Error(`${path} is not the file (status ${response.status})`);
      await cache.put(path, response);
    }));
  } catch (error) {
    // Half a shell is worse than none: leave the installed version in charge.
    await caches.delete(SHELL_CACHE);
    throw error;
  }
}

self.addEventListener('install', (event) => {
  event.waitUntil(precache());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const name of core.staleShellCaches(await caches.keys(), SHELL.version)) await caches.delete(name);
    await self.clients.claim();
  })());
});

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') self.skipWaiting();
});

async function fromShell(path, request) {
  const cache = await caches.open(SHELL_CACHE);
  return (await cache.match(path)) || fetch(request);
}

async function savedEpisode(request) {
  const cache = await caches.open(core.AUDIO_CACHE);
  const saved = await cache.match(new URL(request.url).pathname);
  if (!saved) {
    try {
      return await fetch(request);
    } catch (error) {
      return new Response('This episode is not saved on this device.', {
        status: 503, headers: { 'content-type': 'text/plain; charset=utf-8', 'cache-control': 'no-store' },
      });
    }
  }
  // A blob slices without reading the whole file, so a seek in a 26 MB episode costs what it reads.
  const blob = await saved.blob();
  const answer = core.rangeAnswer(blob.size, request.headers.get('range'));
  const body = request.method === 'HEAD' || answer.status === 416 ? null : blob.slice(answer.start, answer.end + 1);
  return new Response(body, { status: answer.status, headers: answer.headers });
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  const kind = core.classify(
    { url: new URL(request.url), method: request.method, mode: request.mode },
    self.location.origin,
    SHELL_PATHS,
  );
  if (kind === 'audio') event.respondWith(savedEpisode(request));
  else if (kind === 'navigate') event.respondWith(fromShell('/', request));
  else if (kind === 'shell') event.respondWith(fromShell(new URL(request.url).pathname, request));
});
