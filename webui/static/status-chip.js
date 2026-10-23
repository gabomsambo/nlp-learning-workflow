/*
 * Shell status chip. Reads GET /api/system/status, caches the payload for a
 * minute, and links to the Status section. It does not follow a pipeline run:
 * that poller rewrites the page URL and the stored run id.
 */
(function (global) {
  'use strict';

  var STATUS_URL = '/api/system/status';
  var CACHE_KEY = 'nlp:systemStatus';
  var TTL_MS = 60000;
  var inflight = null;

  function chipLabel(payload) {
    if (!payload || !Array.isArray(payload.checks)) return 'Status';
    var down = 0;
    for (var i = 0; i < payload.checks.length; i++) {
      if (payload.checks[i] && payload.checks[i].state !== 'ok') down += 1;
    }
    if (!down) return 'All up';
    return down === 1 ? '1 down' : down + ' down';
  }

  function chipState(payload) {
    if (!payload || !Array.isArray(payload.checks)) return 'unknown';
    for (var i = 0; i < payload.checks.length; i++) {
      if (payload.checks[i] && payload.checks[i].state !== 'ok') return 'down';
    }
    return payload.checks.length ? 'ok' : 'unknown';
  }

  function shouldRefresh(cachedAt, now, ttl) {
    if (!cachedAt) return true;
    var age = now - cachedAt;
    if (!isFinite(age) || age < 0) return true;
    return age >= (ttl == null ? TTL_MS : ttl);
  }

  function readCache(now) {
    try {
      var raw = global.sessionStorage.getItem(CACHE_KEY);
      if (!raw) return null;
      var parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== 'object' || !parsed.payload) return null;
      if (shouldRefresh(parsed.at, now || Date.now(), TTL_MS)) return null;
      return parsed;
    } catch (e) {
      return null;
    }
  }

  function writeCache(payload) {
    try {
      global.sessionStorage.setItem(CACHE_KEY, JSON.stringify({ at: Date.now(), payload: payload }));
    } catch (e) { /* the chip still paints this response */ }
  }

  function paint(payload) {
    if (typeof document === 'undefined') return;
    var label = chipLabel(payload);
    var state = chipState(payload);
    document.querySelectorAll('.status-chip-shell').forEach(function (link) {
      var text = link.querySelector('.status-chip-label');
      if (text) text.textContent = label;
      var dot = link.querySelector('.status-dot');
      if (dot) dot.setAttribute('data-state', state);
      link.setAttribute('aria-label', label + '. Open system status');
    });
  }

  function load(options) {
    var force = !!(options && options.force);
    if (!force) {
      var cached = readCache(Date.now());
      if (cached) {
        paint(cached.payload);
        return Promise.resolve(cached.payload);
      }
      if (inflight) return inflight;
    }
    var request = fetch(STATUS_URL, { headers: { Accept: 'application/json' }, cache: 'no-store' })
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(function (payload) {
        writeCache(payload);
        paint(payload);
        return payload;
      });
    if (force) return request;
    inflight = request.finally(function () {
      if (inflight === request) inflight = null;
    });
    return inflight;
  }

  function boot() {
    load();
    if (typeof document === 'undefined') return;
    document.addEventListener('visibilitychange', function () {
      if (document.visibilityState !== 'visible') return;
      var cached = null;
      try {
        var raw = global.sessionStorage.getItem(CACHE_KEY);
        cached = raw ? JSON.parse(raw) : null;
      } catch (e) {
        cached = null;
      }
      if (!cached || shouldRefresh(cached.at, Date.now(), TTL_MS)) load();
    });
  }

  global.NlpStatus = {
    load: load,
    chipLabel: chipLabel,
    chipState: chipState,
    shouldRefresh: shouldRefresh,
  };

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
    else boot();
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      chipLabel: chipLabel,
      chipState: chipState,
      shouldRefresh: shouldRefresh,
    };
  }
})(typeof window !== 'undefined' ? window : globalThis);
