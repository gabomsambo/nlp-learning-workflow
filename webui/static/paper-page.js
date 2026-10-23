/*
 * Paper page behaviour that is not the HTML itself.
 *
 * Answers use the same localStorage key as the quick-peek modal
 * (nlp:quizAnswersVisible) and the hidden attribute, so a choice on one
 * surface is the choice on the other. Reading progress is per browser only
 * (nlp:readProgress:<paper id>) and is labelled as such in the lesson panel.
 */
(function (global) {
  'use strict';

  var ANSWERS_KEY = 'nlp:quizAnswersVisible';
  var PROGRESS_PREFIX = 'nlp:readProgress:';

  function progressKey(paperId) {
    return PROGRESS_PREFIX + paperId;
  }

  function progressLabel(percent) {
    if (!percent) return 'On this browser · not started';
    return 'On this browser · ' + percent + '% of the lesson';
  }

  function depthPercent(top, height, viewport) {
    if (!height || height <= 0) return null;
    var seen = Math.min(height, Math.max(0, viewport - top));
    return Math.max(0, Math.min(100, Math.round((100 * seen) / height)));
  }

  function mergeProgress(previous, next) {
    if (next === null || typeof next === 'undefined' || isNaN(next)) return previous || 0;
    var prev = Number(previous);
    if (!prev || prev < 0) prev = 0;
    var value = Number(next);
    if (!value || value < 0) value = 0;
    return Math.max(Math.min(100, prev), Math.min(100, value));
  }

  function toggleLabel(show) {
    return show ? 'Hide answers' : 'Show answers';
  }

  function boot() {
    if (typeof document === 'undefined') return;
    var root = document.querySelector('.paper-page');
    if (!root) return;
    var paperId = root.getAttribute('data-paper-id') || '';

    var toggle = document.getElementById('quiz-answers-toggle');
    if (toggle) {
      toggle.addEventListener('click', function () {
        var show = toggle.getAttribute('aria-expanded') !== 'true';
        document.querySelectorAll('[data-quiz-answer]').forEach(function (node) {
          node.hidden = !show;
        });
        toggle.textContent = toggleLabel(show);
        toggle.setAttribute('aria-expanded', show ? 'true' : 'false');
        try {
          global.localStorage.setItem(ANSWERS_KEY, show ? 'true' : 'false');
        } catch (e) { /* the toggle still applies for this view */ }
      });
    }

    var refresh = document.getElementById('refresh-metadata');
    var refreshStatus = document.getElementById('metadata-refresh-status');
    if (refresh) {
      refresh.addEventListener('click', function () {
        var id = refresh.getAttribute('data-paper-id');
        if (!id || refresh.disabled) return;
        refresh.disabled = true;
        if (refreshStatus) {
          refreshStatus.hidden = false;
          refreshStatus.textContent = 'Refreshing metadata…';
          refreshStatus.className = 'refresh-status is-info';
        }
        fetch('/api/papers/' + encodeURIComponent(id) + '/refresh-metadata', { method: 'POST' })
          .then(function (response) {
            return response.json().then(function (body) {
              if (!response.ok) {
                var detail = body && body.detail;
                throw new Error(typeof detail === 'string' ? detail : 'HTTP ' + response.status);
              }
              return body;
            });
          })
          .then(function (body) {
            if (body && body.updated) {
              global.location.reload();
              return;
            }
            if (refreshStatus) {
              refreshStatus.hidden = false;
              refreshStatus.textContent = (body && body.message) || 'Metadata refresh finished.';
              refreshStatus.className = 'refresh-status is-info';
            }
            refresh.disabled = false;
          })
          .catch(function (error) {
            if (refreshStatus) {
              refreshStatus.hidden = false;
              refreshStatus.textContent = error.message;
              refreshStatus.className = 'refresh-status is-error';
            }
            refresh.disabled = false;
          });
      });
    }

    var label = document.getElementById('read-label');
    var bar = document.getElementById('read-bar');
    var lesson = document.getElementById('lesson');
    if (!label || !lesson || !paperId) return;
    var stored = 0;
    try {
      stored = Number(global.localStorage.getItem(progressKey(paperId))) || 0;
    } catch (e) { stored = 0; }
    stored = mergeProgress(0, stored);

    function paint(percent) {
      label.textContent = progressLabel(percent);
      if (bar) bar.style.width = percent + '%';
    }
    paint(stored);

    function measure() {
      var rect = lesson.getBoundingClientRect();
      var next = depthPercent(rect.top, lesson.offsetHeight, global.innerHeight || 0);
      var merged = mergeProgress(stored, next);
      if (merged === stored) return;
      stored = merged;
      paint(stored);
      try {
        global.localStorage.setItem(progressKey(paperId), String(stored));
      } catch (e) { /* labelled as this browser; a private window just won't keep it */ }
    }
    global.addEventListener('scroll', measure, { passive: true });
    global.addEventListener('hashchange', measure);
    measure();
  }

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
    else boot();
  }

  var api = {
    progressKey: progressKey,
    progressLabel: progressLabel,
    depthPercent: depthPercent,
    mergeProgress: mergeProgress,
    toggleLabel: toggleLabel,
    ANSWERS_KEY: ANSWERS_KEY,
  };
  global.PaperPage = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
