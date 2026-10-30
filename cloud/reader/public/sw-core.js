/*
 * The pure parts of the service worker: which requests it answers, how a byte range is cut
 * out of a saved episode, which caches belong to which version. No caches, no fetch, no
 * events in here, so node --test can run it.
 *
 * A classic script on purpose. sw.js loads it with importScripts (a service worker started
 * as a module is not available in every Safari), and the tests load it with require().
 */
(function (root) {
  'use strict';

  var SHELL_PREFIX = 'nlp-shell-';
  // Saved episodes outlive a shell update: the name of the bytes is their hash, so they never go stale.
  var AUDIO_CACHE = 'nlp-audio-v1';
  var AUDIO_PATH = /^\/api\/audio\/[0-9a-f]{64}\.mp3$/;

  function shellCacheName(version) {
    return SHELL_PREFIX + version;
  }

  /** Shell caches that are not the current one. The audio cache is never listed. */
  function staleShellCaches(names, version) {
    var current = shellCacheName(version);
    return names.filter(function (name) { return name.indexOf(SHELL_PREFIX) === 0 && name !== current; });
  }

  /**
   * One byte range of an object `size` bytes long, from a `Range` header. The same rules as
   * the Worker's src/audio.js (a test holds the two together): null = ignore the header and
   * send everything, 'unsatisfiable' = past the end, else {start, end} (inclusive, clamped).
   */
  function parseRange(header, size) {
    if (typeof header !== 'string') return null;
    var match = /^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$/i.exec(header);
    if (!match) return null;
    var first = match[1];
    var last = match[2];
    if (first === '' && last === '') return null;
    if (first === '') {
      var count = Number(last);
      if (!Number.isSafeInteger(count)) return null;
      if (count === 0 || size === 0) return 'unsatisfiable';
      return { start: Math.max(0, size - count), end: size - 1 };
    }
    var start = Number(first);
    if (!Number.isSafeInteger(start)) return null;
    if (start >= size) return 'unsatisfiable';
    var end = last === '' ? size - 1 : Number(last);
    if (!Number.isSafeInteger(end)) return null;
    if (end < start) return null;
    return { start: start, end: Math.min(end, size - 1) };
  }

  /** The status line and headers for answering `rangeHeader` from a saved file of `size` bytes. */
  function rangeAnswer(size, rangeHeader) {
    var range = parseRange(rangeHeader, size);
    if (range === 'unsatisfiable') {
      return { status: 416, start: 0, end: -1, length: 0, headers: { 'content-range': 'bytes */' + size, 'accept-ranges': 'bytes' } };
    }
    var start = range ? range.start : 0;
    var end = range ? range.end : size - 1;
    var headers = {
      'accept-ranges': 'bytes',
      'content-type': 'audio/mpeg',
      'content-length': String(Math.max(0, end - start + 1)),
    };
    if (range) headers['content-range'] = 'bytes ' + start + '-' + end + '/' + size;
    return { status: range ? 206 : 200, start: start, end: end, length: Math.max(0, end - start + 1), headers: headers };
  }

  /**
   * What the worker does with a request:
   *   audio     a saved episode, answered from the cache when it is there
   *   navigate  a page load: the cached app shell
   *   shell     one of the shell's own files
   *   pass      not ours; the browser handles it as if there were no worker
   * `shell` is the set of paths the shell precaches. Everything under /api/ and /ingest passes,
   * apart from audio: documents are kept by the page (IndexedDB), where a hash can be checked.
   */
  function classify(request, origin, shell) {
    var url = request.url;
    if (url.origin !== origin) return 'pass';
    var method = request.method;
    var path = url.pathname;
    if (AUDIO_PATH.test(path)) return method === 'GET' || method === 'HEAD' ? 'audio' : 'pass';
    if (method !== 'GET') return 'pass';
    if (path === '/ingest' || path.indexOf('/ingest/') === 0 || path.indexOf('/api/') === 0) return 'pass';
    if (request.mode === 'navigate') {
      // "Sign in again" lands here: it must reach Access, which a cached page never would.
      return url.searchParams.has('signin') ? 'pass' : 'navigate';
    }
    return shell.has(path) ? 'shell' : 'pass';
  }

  /**
   * Is a response to a shell fetch the file we asked for? An ended Access session answers with a
   * redirect, or with the login page as HTML: neither may be stored as app code.
   */
  function usableShellResponse(path, response) {
    if (!response || response.type === 'opaqueredirect' || response.redirected || !response.ok) return false;
    var type = String((response.headers && response.headers.get('content-type')) || '').toLowerCase();
    var wantsHtml = path === '/' || /\.html?$/.test(path);
    return wantsHtml ? type.indexOf('text/html') === 0 : type.indexOf('text/html') !== 0;
  }

  var api = {
    SHELL_PREFIX: SHELL_PREFIX,
    AUDIO_CACHE: AUDIO_CACHE,
    AUDIO_PATH: AUDIO_PATH,
    shellCacheName: shellCacheName,
    staleShellCaches: staleShellCaches,
    parseRange: parseRange,
    rangeAnswer: rangeAnswer,
    classify: classify,
    usableShellResponse: usableShellResponse,
  };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.NLPSwCore = api;
})(typeof self !== 'undefined' ? self : this);
