/*
 * The device's copy of the published documents: a small async key-value store.
 *
 *   get(key) / set(key, value) / delete(key) / keys(prefix)
 *
 * IndexedDB in the browser, a Map in the tests and wherever IndexedDB is missing or refuses
 * (private windows). Every caller treats a failure here as "nothing saved": the reader then
 * works exactly as it did before there was an offline copy.
 */

const DB_NAME = 'nlp-reader';
const STORE = 'kv';

export function memoryKv(initial = {}) {
  const map = new Map(Object.entries(initial));
  return {
    kind: 'memory',
    async get(key) { return map.has(key) ? structuredClone(map.get(key)) : undefined; },
    async set(key, value) { map.set(key, structuredClone(value)); },
    async delete(key) { map.delete(key); },
    async keys(prefix = '') { return [...map.keys()].filter((k) => k.startsWith(prefix)).sort(); },
  };
}

export function idbKv(indexedDB = globalThis.indexedDB) {
  let opened = null;
  const open = () => {
    if (!opened) {
      opened = new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, 1);
        request.onupgradeneeded = () => request.result.createObjectStore(STORE);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
        request.onblocked = () => reject(new Error('IndexedDB is blocked'));
      });
      opened.catch(() => { opened = null; }); // a later call may succeed
    }
    return opened;
  };

  function run(mode, work) {
    return open().then((db) => new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE, mode);
      const request = work(transaction.objectStore(STORE));
      transaction.oncomplete = () => resolve(request ? request.result : undefined);
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error || new Error('IndexedDB transaction aborted'));
    }));
  }

  return {
    kind: 'indexeddb',
    get: (key) => run('readonly', (s) => s.get(key)),
    set: (key, value) => run('readwrite', (s) => s.put(value, key)),
    delete: (key) => run('readwrite', (s) => s.delete(key)),
    keys: (prefix = '') => run('readonly', (s) => s.getAllKeys(IDBKeyRange.bound(prefix, `${prefix}￿`))),
  };
}

/** IndexedDB if this browser has it, else memory (nothing survives a reload, nothing breaks). */
export function openKv() {
  try {
    if (globalThis.indexedDB) return idbKv();
  } catch (error) { /* fall through */ }
  return memoryKv();
}
