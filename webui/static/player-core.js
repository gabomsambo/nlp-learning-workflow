/*
 * The one player.
 *
 * Every page has exactly one <audio> (the bar's, #mini-player-audio) and one
 * controller wrapped around it. The bar, the Listen page's own transport and
 * the read-along bar all call this controller and all render from its
 * snapshots, so a press on any of them is what the others show a moment later.
 * Nothing else creates or drives an <audio>.
 *
 * A full page load is the end of that document's <audio>, so the position is
 * kept in storage (nlp:player) and handed to the next page's controller, which
 * seeks to it as soon as the metadata is in. The next page also tries to
 * resume playing when the previous one was playing a moment ago; a browser
 * that refuses leaves the bar paused at the exact position.
 *
 * This file is the pure part plus the controller. It takes the audio element
 * and the storage as arguments so node --test can drive it with fakes.
 */
(function (global) {
  'use strict';

  var PLAYER_KEY = 'nlp:player';
  var RATES = [0.75, 1, 1.25, 1.5, 1.75, 2];
  // A page that was playing when it was left asks the next one to carry on, but
  // only for this long: a stale stamp must never start sound minutes later.
  var RESUME_WINDOW_MS = 15000;
  var PERSIST_EVERY_MS = 1000;
  var MEDIA_EVENTS = [
    'play', 'pause', 'seeked', 'ratechange', 'ended', 'emptied',
    'loadedmetadata', 'durationchange', 'timeupdate',
  ];

  function formatClock(seconds) {
    if (!isFinite(seconds) || seconds < 0) return '0:00';
    seconds = Math.floor(seconds);
    var h = Math.floor(seconds / 3600);
    var m = Math.floor((seconds % 3600) / 60);
    var s = seconds % 60;
    function pad(n) { return (n < 10 ? '0' : '') + n; }
    if (h) return h + ':' + pad(m) + ':' + pad(s);
    return m + ':' + pad(s);
  }

  function minuteSecondPhrase(seconds) {
    seconds = Math.floor(Math.max(0, isFinite(seconds) ? seconds : 0));
    var h = Math.floor(seconds / 3600);
    var m = Math.floor((seconds % 3600) / 60);
    var s = seconds % 60;
    if (h) return h + ' hours ' + m + ' minutes ' + s;
    return m + ' minutes ' + s;
  }

  function formatValueText(current, duration) {
    if (!isFinite(duration) || duration <= 0) return minuteSecondPhrase(current);
    return minuteSecondPhrase(current) + ' of ' + minuteSecondPhrase(duration);
  }

  function clampSeek(time, delta, duration) {
    var next = (isFinite(time) ? time : 0) + delta;
    if (next < 0) next = 0;
    if (isFinite(duration) && duration > 0 && next > duration) next = duration;
    return next;
  }

  function normalizeRate(rate) {
    var n = Number(rate);
    for (var i = 0; i < RATES.length; i++) {
      if (Math.abs(RATES[i] - n) < 0.01) return RATES[i];
    }
    return 1;
  }

  function isSafeAudioSrc(src) {
    if (typeof src !== 'string' || !src) return false;
    if (src.indexOf('..') !== -1 || src.indexOf('\\') !== -1) return false;
    if (src.indexOf('\n') !== -1 || src.indexOf('\r') !== -1) return false;
    return src.indexOf('/api/podcast/audio/') === 0;
  }

  function isSafeAppHref(href) {
    if (typeof href !== 'string' || href.charAt(0) !== '/') return false;
    if (href.indexOf('//') !== -1 || href.indexOf('..') !== -1) return false;
    if (href.indexOf('\\') !== -1) return false;
    return true;
  }

  function toStoredSrc(src, base) {
    if (typeof src !== 'string' || !src) return '';
    if (src.indexOf('//') === -1 && src.charAt(0) === '/') {
      var hash = src.indexOf('#');
      return hash === -1 ? src : src.slice(0, hash);
    }
    try {
      var url = new URL(src, base || 'http://localhost');
      if (url.protocol !== 'http:' && url.protocol !== 'https:') return '';
      return url.pathname + url.search;
    } catch (e) {
      return '';
    }
  }

  function cleanTitle(title) {
    if (typeof title !== 'string') return 'Episode';
    var text = title.replace(/\s+/g, ' ').trim();
    if (!text) return 'Episode';
    return text.slice(0, 180);
  }

  function safeId(scriptId) {
    return typeof scriptId === 'string' && /^[A-Za-z0-9-]+$/.test(scriptId) ? scriptId : '';
  }

  function fallbackHref(scriptId) {
    var id = safeId(scriptId);
    return id ? '/podcast/?script=' + encodeURIComponent(id) : '/podcast/';
  }

  /*
   * Where "open this episode" goes: its own page, in read-along. The Listen
   * page reads `view=readalong` as "I came to follow along" and lands at the
   * spot being spoken, whatever view it last remembered.
   */
  function readAlongHref(scriptId, fallback) {
    var id = safeId(scriptId);
    if (id) return '/podcast/?script=' + encodeURIComponent(id) + '&view=readalong';
    return isSafeAppHref(fallback) ? fallback : '/podcast/';
  }

  function parsePlayerState(raw) {
    if (!raw) return null;
    var data;
    try {
      data = typeof raw === 'string' ? JSON.parse(raw) : raw;
    } catch (e) {
      return null;
    }
    if (!data || typeof data !== 'object') return null;
    var src = toStoredSrc(data.src, 'http://localhost');
    if (!isSafeAudioSrc(src)) return null;
    var time = Number(data.time);
    if (!isFinite(time) || time < 0) time = 0;
    if (time > 48 * 3600) time = 48 * 3600;
    var scriptId = typeof data.script_id === 'string' ? data.script_id.slice(0, 80) : '';
    var resumeUntil = Number(data.resume_until);
    return {
      script_id: scriptId,
      src: src,
      time: time,
      rate: normalizeRate(data.rate),
      title: cleanTitle(data.title),
      href: isSafeAppHref(data.href) ? data.href : fallbackHref(scriptId),
      resume_until: isFinite(resumeUntil) && resumeUntil > 0 ? resumeUntil : 0,
    };
  }

  /*
   * What opening an episode's page should do about the player.
   *
   * `attached`: the episode on screen is the one the player holds, so the page
   * lands where the audio is (and follows it). Anything else leaves the player
   * alone: looking at a different episode never takes over the one playing.
   * The match is by script id, falling back to the audio path for state saved
   * before it carried an id.
   */
  function landingFor(snapshot, ref) {
    var none = { attached: false, time: 0, playing: false };
    var track = snapshot && snapshot.track;
    if (!track || !ref) return none;
    var sameId = !!(ref.script_id && track.script_id && track.script_id === ref.script_id);
    var sameSrc = !!(ref.src && track.src === ref.src);
    if (!sameId && !sameSrc) return none;
    return { attached: true, time: snapshot.time || 0, playing: !!snapshot.playing };
  }

  function nowMs() { return Date.now(); }

  /*
   * options.audio    the one <audio>
   * options.storage  getItem/setItem/removeItem (localStorage); may be absent
   * options.base     page URL, for turning currentSrc into a stored path
   * options.now      clock, for tests
   */
  function createPlayer(options) {
    options = options || {};
    var audio = options.audio;
    var storage = options.storage || null;
    var base = options.base || 'http://localhost';
    var now = options.now || nowMs;
    var track = null;
    var rate = 1;
    var lastTime = 0;
    var pendingSeek = null;
    var lastPersist = 0;
    var listeners = [];
    var bound = false;

    function audioSrc() {
      if (!audio) return '';
      return toStoredSrc(audio.currentSrc || audio.src || '', base);
    }

    // Metadata for the loaded track is in, so currentTime means something.
    function ready() {
      return !!(audio && track && audio.readyState >= 1 && audioSrc() === track.src);
    }

    function duration() {
      return ready() && isFinite(audio.duration) && audio.duration > 0 ? audio.duration : 0;
    }

    function time() {
      if (!track) return 0;
      if (ready() && isFinite(audio.currentTime)) return audio.currentTime;
      return pendingSeek != null ? pendingSeek : lastTime;
    }

    function snapshot() {
      var paused = !audio || !track || !!audio.paused;
      return {
        track: track ? {
          script_id: track.script_id,
          src: track.src,
          title: track.title,
          href: track.href,
        } : null,
        time: time(),
        duration: duration(),
        paused: paused,
        playing: !paused,
        rate: rate,
        ready: ready(),
      };
    }

    function emit(reason) {
      var snap = snapshot();
      listeners.slice().forEach(function (fn) {
        try { fn(snap, reason); } catch (e) { /* one bad view must not stop the others */ }
      });
    }

    function applyRate() {
      if (!audio) return;
      try {
        audio.defaultPlaybackRate = rate;
        audio.playbackRate = rate;
      } catch (e) { /* a media element that refuses keeps its own rate */ }
    }

    function write(resumeUntil) {
      if (!storage || !track) return;
      var state = {
        script_id: track.script_id,
        src: track.src,
        time: time(),
        rate: rate,
        title: track.title,
        href: track.href,
      };
      if (resumeUntil) state.resume_until = resumeUntil;
      try { storage.setItem(PLAYER_KEY, JSON.stringify(state)); } catch (e) { /* private window */ }
    }

    function persist(force) {
      if (!track) return;
      var stamp = now();
      if (!force && stamp - lastPersist < PERSIST_EVERY_MS) return;
      lastPersist = stamp;
      write(0);
    }

    function applyPendingSeek() {
      if (pendingSeek == null || !ready()) return;
      var target = pendingSeek;
      pendingSeek = null;
      if (isFinite(audio.duration) && audio.duration > 0 && target > audio.duration) target = audio.duration;
      try { audio.currentTime = target; } catch (e) { /* retried by the next metadata event */ }
      lastTime = target;
    }

    function onMedia(event) {
      var name = event && event.type;
      if (!track) return;
      if (name === 'loadedmetadata' || name === 'durationchange') {
        applyPendingSeek();
        applyRate();
      }
      if (name === 'ratechange' && ready()) {
        var next = normalizeRate(audio.playbackRate);
        if (next !== rate) rate = next;
      }
      if (ready() && isFinite(audio.currentTime) && pendingSeek == null) lastTime = audio.currentTime;
      persist(name !== 'timeupdate');
      emit(name);
    }

    function bind() {
      if (bound || !audio || !audio.addEventListener) return;
      bound = true;
      MEDIA_EVENTS.forEach(function (name) { audio.addEventListener(name, onMedia); });
    }

    function tryPlay() {
      if (!audio || !track) return null;
      var pending = null;
      try { pending = audio.play(); } catch (e) { return null; }
      if (pending && typeof pending.catch === 'function') pending.catch(function () { /* autoplay refused */ });
      return pending;
    }

    function setSource(src, at, preload) {
      pendingSeek = at > 0 ? at : 0;
      lastTime = pendingSeek;
      audio.preload = preload;
      audio.src = src;
      applyRate();
    }

    /*
     * Put an episode in the player. The same episode keeps its place (an
     * explicit opts.at moves it); a different one replaces the track and
     * starts from opts.at or 0. opts.play starts it.
     */
    function load(meta, opts) {
      meta = meta || {};
      opts = opts || {};
      var src = toStoredSrc(meta.src || '', base);
      if (!audio || !isSafeAudioSrc(src)) return null;
      var scriptId = typeof meta.script_id === 'string' ? meta.script_id.slice(0, 80) : '';
      if (track && track.src === src) {
        if (scriptId) track.script_id = scriptId;
        if (meta.title) track.title = cleanTitle(meta.title);
        track.href = isSafeAppHref(meta.href) ? meta.href : fallbackHref(track.script_id);
        if (meta.rate != null) {
          rate = normalizeRate(meta.rate);
          applyRate();
        }
        if (opts.at != null) seekTo(opts.at);
      } else {
        if (meta.rate != null) rate = normalizeRate(meta.rate);
        track = {
          script_id: scriptId,
          src: src,
          title: cleanTitle(meta.title),
          href: isSafeAppHref(meta.href) ? meta.href : fallbackHref(scriptId),
        };
        setSource(src, opts.at != null ? Math.max(0, Number(opts.at) || 0) : 0, opts.play ? 'auto' : 'metadata');
      }
      bind();
      persist(true);
      emit('load');
      if (opts.play) tryPlay();
      return snapshot();
    }

    function play() { return tryPlay(); }

    function pause() {
      if (!audio || !track) return;
      try { audio.pause(); } catch (e) { /* nothing is playing */ }
    }

    function toggle() {
      if (!audio || !track) return;
      if (audio.paused) tryPlay();
      else pause();
    }

    function seekTo(seconds) {
      if (!audio || !track) return;
      var target = Number(seconds);
      if (!isFinite(target) || target < 0) target = 0;
      var length = duration();
      if (length > 0 && target > length) target = length;
      lastTime = target;
      if (ready()) {
        pendingSeek = null;
        try { audio.currentTime = target; } catch (e) { pendingSeek = target; }
      } else {
        pendingSeek = target;
      }
      persist(true);
      emit('seek');
    }

    function seekBy(delta) {
      if (!track) return;
      seekTo(clampSeek(time(), delta, duration()));
    }

    function setRate(next) {
      rate = normalizeRate(next);
      applyRate();
      persist(true);
      emit('rate');
    }

    /* Bring the last saved episode back, paused, at its saved place. */
    function restore() {
      if (!audio || !storage) return null;
      var saved = null;
      try { saved = parsePlayerState(storage.getItem(PLAYER_KEY)); } catch (e) { saved = null; }
      if (!saved) return null;
      track = {
        script_id: saved.script_id,
        src: saved.src,
        title: saved.title,
        href: saved.href,
      };
      rate = saved.rate;
      setSource(saved.src, saved.time, 'metadata');
      bind();
      emit('restore');
      if (saved.resume_until && now() < saved.resume_until) tryPlay();
      return snapshot();
    }

    /*
     * The page is going away. Save the place exactly; if sound is playing,
     * leave the stamp that asks the next page to carry on.
     */
    function suspend(navigating) {
      if (!track) return;
      lastPersist = now();
      write(navigating && audio && !audio.paused ? now() + RESUME_WINDOW_MS : 0);
    }

    function subscribe(fn) {
      listeners.push(fn);
      return function () {
        listeners = listeners.filter(function (item) { return item !== fn; });
      };
    }

    return {
      load: load,
      play: play,
      pause: pause,
      toggle: toggle,
      seekTo: seekTo,
      seekBy: seekBy,
      setRate: setRate,
      restore: restore,
      suspend: suspend,
      snapshot: snapshot,
      subscribe: subscribe,
      isCurrent: function (ref) { return landingFor(snapshot(), ref).attached; },
    };
  }

  var api = {
    PLAYER_KEY: PLAYER_KEY,
    RATES: RATES,
    RESUME_WINDOW_MS: RESUME_WINDOW_MS,
    formatClock: formatClock,
    formatValueText: formatValueText,
    clampSeek: clampSeek,
    normalizeRate: normalizeRate,
    isSafeAudioSrc: isSafeAudioSrc,
    isSafeAppHref: isSafeAppHref,
    toStoredSrc: toStoredSrc,
    cleanTitle: cleanTitle,
    fallbackHref: fallbackHref,
    readAlongHref: readAlongHref,
    parsePlayerState: parsePlayerState,
    landingFor: landingFor,
    createPlayer: createPlayer,
  };

  global.NlpPlayerCore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
