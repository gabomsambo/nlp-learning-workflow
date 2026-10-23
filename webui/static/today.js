/*
 * Today: streak dots, plan ticks, the service-status hook, and the in-progress
 * card.
 *
 * The card reads GET /api/pipeline-runs/active and paints with
 * RunProgress.preview. It must not call RunProgress.attach — attach writes
 * ?run= and localStorage, which belong to the page that started the run.
 *
 * Ticks come from real events. A review today (local calendar), a lesson whose
 * nlp:readProgress:<paper id> reached the end, or an episode played past 90%.
 * There is no checkbox.
 */
(function (global) {
  var READ_PREFIX = 'nlp:readProgress:';
  var HEARD_PREFIX = 'nlp:heard:';
  var LETTERS = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];
  var POLL_MS = 4000;

  function pad(n) {
    return (n < 10 ? '0' : '') + n;
  }

  function browserLocalDate(ms) {
    var d = new Date(ms);
    return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate());
  }

  function addDays(key, delta) {
    var parts = key.split('-');
    var d = new Date(Date.UTC(Number(parts[0]), Number(parts[1]) - 1, Number(parts[2])));
    d.setUTCDate(d.getUTCDate() + delta);
    return d.getUTCFullYear() + '-' + pad(d.getUTCMonth() + 1) + '-' + pad(d.getUTCDate());
  }

  function weekdayIndex(key) {
    var parts = key.split('-');
    var d = new Date(Date.UTC(Number(parts[0]), Number(parts[1]) - 1, Number(parts[2])));
    return (d.getUTCDay() + 6) % 7;
  }

  function streakFromTimestamps(timestamps, nowMs, toLocalDate) {
    var local = toLocalDate || browserLocalDate;
    var seen = {};
    (timestamps || []).forEach(function (raw) {
      var ms = typeof raw === 'number' ? raw : Date.parse(raw);
      if (isNaN(ms)) return;
      seen[local(ms)] = true;
    });
    var today = local(nowMs);
    var monday = addDays(today, -weekdayIndex(today));
    var dots = [];
    var i;
    for (i = 0; i < 7; i++) {
      var day = addDays(monday, i);
      dots.push({
        label: LETTERS[i],
        iso: day,
        reviewed: !!seen[day],
        isToday: day === today,
        isFuture: day > today,
      });
    }
    var cursor = seen[today] ? today : addDays(today, -1);
    var count = 0;
    if (seen[cursor]) {
      while (seen[cursor]) {
        count += 1;
        cursor = addDays(cursor, -1);
      }
    }
    return { count: count, dots: dots };
  }

  function heardPastNinety(time, duration) {
    time = Number(time);
    duration = Number(duration);
    if (!isFinite(time) || !isFinite(duration) || duration <= 0 || time < 0) return false;
    return time / duration >= 0.9;
  }

  function readRatio(paperId) {
    if (!paperId) return null;
    var raw;
    try { raw = global.localStorage.getItem(READ_PREFIX + paperId); } catch (e) { return null; }
    if (raw == null || raw === '') return null;
    var value = raw;
    try { value = JSON.parse(raw); } catch (e) { value = raw; }
    if (typeof value === 'number' && isFinite(value)) return value;
    if (value && typeof value === 'object') {
      if (value.complete === true) return 1;
      if (typeof value.ratio === 'number') return value.ratio;
      if (typeof value.depth === 'number') return value.depth;
    }
    var n = Number(raw);
    return isFinite(n) ? n : null;
  }

  function readDone(ratio) {
    return typeof ratio === 'number' && isFinite(ratio) && ratio >= 0.97;
  }

  function statusChipText(status, body) {
    if (status !== 200 || !body || typeof body !== 'object') return null;
    var summary = typeof body.summary === 'string' ? body.summary.trim() : '';
    if (summary) return { ok: body.ok !== false, text: summary.slice(0, 160) };
    var message = typeof body.message === 'string' ? body.message.trim() : '';
    if (message) return { ok: body.ok !== false, text: message.slice(0, 160) };
    if (Array.isArray(body.services)) {
      var down = null;
      for (var i = 0; i < body.services.length; i++) {
        if (body.services[i] && body.services[i].ok === false) {
          down = body.services[i];
          break;
        }
      }
      if (!down && body.ok !== false) return { ok: true, text: 'All services up' };
      if (down) {
        var name = String(down.label || down.name || 'A service').trim() || 'A service';
        return { ok: false, text: name + ' needs attention' };
      }
    }
    if (body.ok === true) return { ok: true, text: 'All services up' };
    if (body.ok === false) return { ok: false, text: 'A service needs attention' };
    return null;
  }

  function setDone(row, done) {
    if (!row) return;
    row.classList.toggle('is-done', !!done);
    var state = row.querySelector('.plan-state');
    if (state) state.textContent = done ? 'Done' : 'Not done yet';
  }

  function applyStreak(root, model) {
    var count = root.querySelector('#streak-count');
    var unit = root.querySelector('#streak-unit');
    var list = root.querySelector('#streak-dots');
    if (count) count.textContent = String(model.count);
    if (unit) unit.textContent = model.count === 1 ? 'day' : 'days';
    if (list) {
      list.replaceChildren();
      model.dots.forEach(function (dot) {
        var li = document.createElement('li');
        li.className = 'streak-dot'
          + (dot.reviewed ? ' is-done' : '')
          + (dot.isToday ? ' is-today' : '')
          + (dot.isFuture ? ' is-future' : '');
        var span = document.createElement('span');
        span.textContent = dot.label;
        li.appendChild(span);
        list.appendChild(li);
      });
    }
    var doneLabels = model.dots.filter(function (dot) { return dot.reviewed; }).map(function (dot) { return dot.label; });
    var head = model.count === 1 ? '1 day in a row' : model.count + ' days in a row';
    root.setAttribute('aria-label', doneLabels.length
      ? head + '. Reviewed ' + doneLabels.join(', ') + ' this week.'
      : head + '. No reviews yet this week.');
  }

  function showReadProgress(paperId) {
    var slot = document.getElementById('read-progress');
    var ratio = readRatio(paperId);
    if (ratio == null) return;
    if (readDone(ratio)) {
      setDone(document.getElementById('plan-next'), true);
      if (slot) slot.textContent = 'Read on this browser';
      return;
    }
    if (ratio > 0 && slot) {
      slot.textContent = ' · ' + Math.round(ratio * 100) + '% read on this browser';
    }
  }

  function rememberedHeard(scriptId) {
    try { return global.localStorage.getItem(HEARD_PREFIX + scriptId) === '1'; } catch (e) { return false; }
  }

  function rememberHeard(scriptId) {
    try { global.localStorage.setItem(HEARD_PREFIX + scriptId, '1'); } catch (e) {}
  }

  function episodeProgress(scriptId, durationHint) {
    if (rememberedHeard(scriptId)) return true;
    var stored = global.NlpPlayer && typeof global.NlpPlayer.read === 'function'
      ? global.NlpPlayer.read() : null;
    if (!stored || stored.script_id !== scriptId) return false;
    var audio = document.getElementById('mini-player-audio');
    var duration = Number(durationHint);
    var time = stored.time;
    if (audio && isFinite(audio.duration) && audio.duration > 0) duration = audio.duration;
    if (audio && isFinite(audio.currentTime)) time = audio.currentTime;
    if (!heardPastNinety(time, duration)) return false;
    rememberHeard(scriptId);
    return true;
  }

  function runElements() {
    return {
      status: document.getElementById('today-run-status'),
      stages: document.getElementById('today-run-stages'),
      progress: document.getElementById('today-run-bar'),
      cancel: document.getElementById('today-run-cancel'),
    };
  }

  function paintRun(run) {
    if (!global.RunProgress || typeof global.RunProgress.preview !== 'function') return;
    global.RunProgress.preview(run && run.id ? run : null, runElements());
  }

  function readPayload() {
    var node = document.getElementById('today-data');
    if (!node) return {};
    try { return JSON.parse(node.textContent || '{}'); } catch (e) { return {}; }
  }

  function wirePlayButtons() {
    document.querySelectorAll('[data-play-episode]').forEach(function (button) {
      button.addEventListener('click', function () {
        if (!global.NlpPlayer || typeof global.NlpPlayer.play !== 'function') return;
        global.NlpPlayer.play({
          src: button.getAttribute('data-src'),
          script_id: button.getAttribute('data-script-id') || '',
          title: button.getAttribute('data-title') || '',
          href: button.getAttribute('data-href') || '',
        });
      });
    });
  }

  function wireHeard(scriptId, durationHint) {
    var audio = document.getElementById('mini-player-audio');
    if (!audio || !scriptId) return;
    var check = function () {
      if (episodeProgress(scriptId, durationHint)) {
        setDone(document.getElementById('plan-listen'), true);
      }
    };
    audio.addEventListener('timeupdate', check);
    audio.addEventListener('ended', check);
    check();
  }

  function loadStatus() {
    var slot = document.getElementById('service-status-slot');
    if (!slot) return;
    var endpoint = slot.getAttribute('data-status-endpoint');
    if (!endpoint) return;
    fetch(endpoint).then(function (res) {
      if (res.status !== 200) return null;
      return res.json().then(function (body) { return statusChipText(200, body); });
    }).then(function (chip) {
      if (!chip || !slot) return;
      slot.hidden = false;
      slot.textContent = chip.text;
      slot.classList.toggle('is-down', !chip.ok);
    }).catch(function () {});
  }

  function boot() {
    var data = readPayload();
    if (Array.isArray(data.reviews)) {
      var model = streakFromTimestamps(data.reviews, Date.now());
      var streak = document.getElementById('streak');
      if (streak) applyStreak(streak, model);
      var today = browserLocalDate(Date.now());
      var didReview = data.reviews.some(function (iso) {
        var ms = Date.parse(iso);
        return !isNaN(ms) && browserLocalDate(ms) === today;
      });
      setDone(document.getElementById('plan-review'), didReview);
    }
    if (data.upNextPaperId) showReadProgress(data.upNextPaperId);
    if (data.episode && data.episode.scriptId) {
      wireHeard(data.episode.scriptId, data.episode.duration);
    }
    wirePlayButtons();
    loadStatus();
    paintRun(data.active);
    var cancel = document.getElementById('today-run-cancel');
    if (cancel) {
      cancel.addEventListener('click', function () {
        var id = cancel.dataset.runId;
        if (!id || !global.RunProgress || typeof global.RunProgress.cancel !== 'function') return;
        cancel.disabled = true;
        global.RunProgress.cancel(id).catch(function () { cancel.disabled = false; });
      });
    }
    var timer = null;
    var stopped = false;
    function poll() {
      if (stopped) return;
      fetch('/api/pipeline-runs/active').then(function (res) {
        if (!res.ok) throw new Error('status');
        return res.json();
      }).then(function (run) {
        paintRun(run);
      }).catch(function () {}).then(function () {
        if (!stopped) timer = global.setTimeout(poll, POLL_MS);
      });
    }
    poll();
    document.addEventListener('visibilitychange', function () {
      if (document.visibilityState === 'hidden') {
        stopped = true;
        if (timer) global.clearTimeout(timer);
      } else if (stopped) {
        stopped = false;
        poll();
      }
    });
  }

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
    else boot();
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      streakFromTimestamps: streakFromTimestamps,
      addDays: addDays,
      heardPastNinety: heardPastNinety,
      readRatio: readRatio,
      readDone: readDone,
      statusChipText: statusChipText,
    };
  }
})(typeof window !== 'undefined' ? window : globalThis);
