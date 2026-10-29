/*
 * Keeping things on the phone: the newest episodes (audio and the documents that go with them),
 * the ones the person saved by hand, and the service worker that lets the app open with no
 * connection.
 *
 *   sync()       fetch the base documents, work out which episodes to hold (offline-core.js),
 *                delete the audio that is no longer wanted, download what is missing.
 *   save(id)     keep this episode on the device until it is removed
 *   remove(id)   stop keeping it (it stays if it is one of the newest)
 *
 * Audio goes into the Cache API under its own URL, `/api/audio/<sha256>.mp3`, as one complete
 * 200 response; sw.js answers Range requests from it. A download is only stored once the bytes
 * hash to the name they were fetched under, so a login page or half a file is never kept.
 * Documents are kept by the store (data.js) as they are read.
 *
 * Nothing here throws to the page: a download that fails is a state ("Not saved", with a
 * reason), and the next sync tries again.
 */
import { BASE_DOCS, episodeDocKeys, planEpisodes, playableEpisodes } from './offline-core.js';
import { audioSrc, sha256Hex } from './lib.js';

export const KEEP_NEWEST = 3;
export const AUDIO_CACHE = 'nlp-audio-v1'; // sw-core.js names the same cache

const SAVED_KEY = 'meta:saved';
const PINS_KEY = 'meta:pins';
const SETTINGS_KEY = 'meta:settings';

export class DownloadError extends Error {
  constructor(kind, message) {
    super(message || kind);
    this.kind = kind; // auth | network | corrupt | full
  }
}

function kindOf(error) {
  if (error instanceof DownloadError) return error.kind;
  if (error && error.kind === 'auth') return 'auth';
  if (error && (error.name === 'QuotaExceededError' || error.code === 22)) return 'full';
  return 'network';
}

const pathOf = (audioKey) => audioSrc(audioKey);

export function createOffline({
  store,
  kv,
  cacheStorage = globalThis.caches,
  fetchFn = (...args) => globalThis.fetch(...args),
  isOnline = () => globalThis.navigator?.onLine !== false,
  storageManager = globalThis.navigator?.storage,
  // False when the person asked the browser to save data: then only episodes saved by hand are downloaded.
  autoAllowed = () => true,
}) {
  const listeners = new Set();
  const progress = new Map(); // episode id -> 0..1 while it downloads
  const failed = new Map(); // episode id -> kind of the last failure
  let running = null;

  const changed = () => {
    for (const listener of listeners) {
      try { listener(); } catch (error) { /* a broken listener must not stop a download */ }
    }
  };

  const read = async (key, fallback) => {
    try {
      const value = await kv.get(key);
      return value === undefined ? fallback : value;
    } catch (error) {
      return fallback;
    }
  };
  const write = async (key, value) => {
    try { await kv.set(key, value); } catch (error) { /* the copy is a convenience */ }
  };

  const audioCache = async () => (cacheStorage ? cacheStorage.open(AUDIO_CACHE) : null);

  async function cachedPaths(cache) {
    if (!cache) return new Set();
    return new Set((await cache.keys()).map((request) => new URL(request.url, 'https://reader.invalid').pathname));
  }

  async function downloadAudio(audioKey, onProgress) {
    const path = pathOf(audioKey);
    const cache = await audioCache();
    if (!path || !cache) throw new DownloadError('network', 'No audio to download.');
    let response;
    try {
      // no-store: the browser must not keep a second 26 MB copy in its own HTTP cache.
      response = await fetchFn(path, { credentials: 'same-origin', redirect: 'manual', cache: 'no-store' });
    } catch (error) {
      throw new DownloadError('network');
    }
    if (response.type === 'opaqueredirect' || response.status === 401 || response.status === 403) throw new DownloadError('auth');
    if (response.status !== 200) throw new DownloadError('network', `The reader answered ${response.status}.`);

    const total = Number(response.headers.get('content-length')) || 0;
    const chunks = [];
    let loaded = 0;
    try {
      if (response.body && response.body.getReader) {
        const reader = response.body.getReader();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          loaded += value.length;
          if (total) onProgress(Math.min(1, loaded / total));
        }
      } else {
        const bytes = new Uint8Array(await response.arrayBuffer());
        chunks.push(bytes);
        loaded = bytes.length;
      }
    } catch (error) {
      throw new DownloadError('network');
    }
    const blob = new Blob(chunks, { type: 'audio/mpeg' });
    const expected = path.slice('/api/audio/'.length, -'.mp3'.length);
    if (blob.size !== loaded || (await sha256Hex(new Uint8Array(await blob.arrayBuffer()))) !== expected) {
      throw new DownloadError('corrupt');
    }
    try {
      await cache.put(path, new Response(blob, {
        status: 200,
        headers: { 'content-type': 'audio/mpeg', 'content-length': String(blob.size), 'accept-ranges': 'bytes' },
      }));
    } catch (error) {
      throw new DownloadError(kindOf(error), error && error.message);
    }
    return blob.size;
  }

  /** The documents an episode is read from, so it opens with no connection. */
  async function fetchDocs(entry) {
    const library = await store.library();
    const episodeDoc = await store.doc(entry.doc);
    const paperEntry = (library.papers || []).find((p) => p.id === entry.paper_id);
    const paperDoc = paperEntry ? await store.doc(paperEntry.doc) : null;
    await Promise.all(episodeDocKeys(entry, episodeDoc, paperDoc, library).map((key) => store.doc(key)));
  }

  async function fetchEpisode(entry, saved) {
    progress.set(entry.id, 0);
    failed.delete(entry.id);
    changed();
    try {
      await fetchDocs(entry);
      const bytes = await downloadAudio(entry.audio_key, (fraction) => {
        progress.set(entry.id, fraction);
        changed();
      });
      saved[entry.id] = { audio_key: entry.audio_key, bytes, saved_at: new Date().toISOString() };
      await write(SAVED_KEY, saved);
    } catch (error) {
      failed.set(entry.id, kindOf(error));
    } finally {
      progress.delete(entry.id);
      changed();
    }
  }

  async function autoKeep() {
    return (await read(SETTINGS_KEY, {})).auto !== false;
  }

  async function persist() {
    try {
      if (!storageManager || !storageManager.persist) return null;
      if (await storageManager.persisted()) return true;
      return await storageManager.persist();
    } catch (error) {
      return null;
    }
  }

  async function runSync() {
    let library;
    try {
      library = await store.library();
    } catch (error) {
      return { status: 'unavailable' };
    }
    const episodes = playableEpisodes(library);
    const cache = await audioCache();
    const present = await cachedPaths(cache);

    let saved = await read(SAVED_KEY, {});
    // The system may have taken the audio back (storage pressure): a record without its file is not a copy.
    saved = Object.fromEntries(Object.entries(saved).filter(([, r]) => present.has(pathOf(r.audio_key))));
    const pins = (await read(PINS_KEY, [])).filter((id) => episodes.some((e) => e.id === id));
    const plan = planEpisodes({ episodes, saved, pinned: pins, keep: (await autoKeep()) ? KEEP_NEWEST : 0 });

    for (const id of plan.remove) {
      if (cache) await cache.delete(pathOf(saved[id].audio_key));
      delete saved[id];
    }
    // Audio nothing claims (an interrupted run, an older version) is dead weight.
    const claimed = new Set(Object.values(saved).map((r) => pathOf(r.audio_key)));
    if (cache) for (const path of present) if (!claimed.has(path)) await cache.delete(path);
    await write(SAVED_KEY, saved);
    await write(PINS_KEY, pins);
    changed();

    if (isOnline()) {
      for (const key of BASE_DOCS) {
        try { await store.doc(key); } catch (error) { /* offline, or signed out: the next sync retries */ }
      }
      const byId = new Map(episodes.map((e) => [e.id, e]));
      const wanted = autoAllowed() ? plan.download : plan.download.filter((id) => pins.includes(id));
      for (const id of wanted) await fetchEpisode(byId.get(id), saved);
      // Worth asking once there is something to lose.
      if (Object.keys(saved).length > 0) await persist();
    }
    return { status: 'ok', downloaded: plan.download.filter((id) => saved[id]).length, removed: plan.remove.length };
  }

  /** One run at a time; a second caller gets the first's result. Never throws. */
  function sync() {
    if (!running) {
      running = runSync().catch(() => ({ status: 'error' })).finally(() => { running = null; changed(); });
    }
    return running;
  }

  return {
    onChange(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },

    sync,

    /** Every episode with audio, newest first, and what the phone holds of it. */
    async episodes() {
      const library = await store.library();
      const saved = await read(SAVED_KEY, {});
      const pins = new Set(await read(PINS_KEY, []));
      return playableEpisodes(library).map((episode) => {
        const record = saved[episode.id];
        const have = !!record && record.audio_key === episode.audio_key;
        let status = 'none';
        if (progress.has(episode.id)) status = 'saving';
        else if (have) status = 'saved';
        else if (failed.has(episode.id)) status = 'error';
        return {
          id: episode.id,
          title: episode.title,
          created_at: episode.created_at,
          duration_seconds: episode.duration_seconds,
          status,
          pinned: pins.has(episode.id),
          bytes: have ? record.bytes : null,
          progress: progress.has(episode.id) ? progress.get(episode.id) : null,
          problem: failed.get(episode.id) || null,
        };
      });
    },

    async save(id) {
      const pins = new Set(await read(PINS_KEY, []));
      pins.add(id);
      await write(PINS_KEY, [...pins]);
      failed.delete(id);
      changed();
      return sync();
    },

    async remove(id) {
      const pins = new Set(await read(PINS_KEY, []));
      pins.delete(id);
      await write(PINS_KEY, [...pins]);
      return sync();
    },

    autoKeep,
    async setAutoKeep(on) {
      await write(SETTINGS_KEY, { ...(await read(SETTINGS_KEY, {})), auto: !!on });
      return sync();
    },

    /** Whether the browser has promised to keep this site's data. null = cannot say. Never asks. */
    async persisted() {
      try {
        return storageManager && storageManager.persisted ? await storageManager.persisted() : null;
      } catch (error) {
        return null;
      }
    },

    /** Ask the browser not to clear this site's data under storage pressure. null = cannot say. */
    persist,

    async usage() {
      try {
        if (!storageManager || !storageManager.estimate) return null;
        const { usage, quota } = await storageManager.estimate();
        return { usage, quota };
      } catch (error) {
        return null;
      }
    },
  };
}

/**
 * Register the service worker and report when a new version is waiting. `applyUpdate` asks it to
 * take over and reloads once it has, so the code never changes under an open episode.
 */
export function registerWorker({
  nav = globalThis.navigator,
  reload = () => globalThis.location.reload(),
  onChange = () => {},
} = {}) {
  const state = { supported: !!(nav && nav.serviceWorker), registered: false, updateReady: false, error: null };
  if (!state.supported) return { state: () => state, applyUpdate() {}, check() {} };
  let registration = null;

  const waiting = () => {
    const ready = !!(registration && registration.waiting && nav.serviceWorker.controller);
    if (ready !== state.updateReady) {
      state.updateReady = ready;
      onChange();
    }
  };
  const watch = (worker) => {
    worker.addEventListener('statechange', () => { if (worker.state === 'installed') waiting(); });
  };

  nav.serviceWorker.register('/sw.js', { scope: '/', updateViaCache: 'none' }).then((made) => {
    registration = made;
    state.registered = true;
    if (made.installing) watch(made.installing);
    made.addEventListener('updatefound', () => { if (made.installing) watch(made.installing); });
    waiting();
    onChange();
  }).catch((error) => {
    state.error = error && error.message ? error.message : String(error);
    onChange();
  });

  let reloading = false;
  nav.serviceWorker.addEventListener('controllerchange', () => {
    // The first install also changes the controller (clients.claim); only an update needs a reload.
    if (state.updateReady && !reloading) {
      reloading = true;
      reload();
    }
  });

  return {
    state: () => state,
    applyUpdate() {
      if (registration && registration.waiting) registration.waiting.postMessage({ type: 'SKIP_WAITING' });
    },
    check() {
      if (registration) registration.update().catch(() => {});
    },
  };
}
