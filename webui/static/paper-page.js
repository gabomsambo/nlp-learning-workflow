/*
 * Paper page behaviour that is not the HTML itself.
 *
 * Answers use the same localStorage key as the quick-peek modal
 * (nlp:quizAnswersVisible) and the hidden attribute, so a choice on one
 * surface is the choice on the other. Reading progress is per browser only
 * (nlp:readProgress:<paper id>) and is labelled as such in the lesson panel.
 *
 * The Episode panel also shows each episode's fact check. The claim logic is
 * factcheck.js (shared with Listen); this file only fetches and draws it.
 */
(function (global) {
  'use strict';

  var Factcheck = global.Factcheck ||
    (typeof require === 'function' ? require('./factcheck.js') : {});
  var TITLE_MAX = 140;
  var QUOTE_MAX = 160;

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

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function readJson(response) {
    return response.json().catch(function () { return {}; });
  }

  function loadEpisodeFactcheck(scriptId) {
    var base = '/api/podcast/' + encodeURIComponent(scriptId);
    var check = fetch(base + '/factcheck').then(function (res) {
      return readJson(res).then(function (data) { return { status: res.status, data: data }; });
    }).catch(function (err) {
      return { status: 0, data: { detail: err.message || 'Could not load fact check.' } };
    });
    var run = fetch('/api/pipeline-runs?kind=podcast_factcheck&script_id=' +
      encodeURIComponent(scriptId) + '&limit=1').then(function (res) {
      return readJson(res).then(function (data) {
        if (!res.ok) return { ok: false, detail: (data && data.detail) || 'Could not read fact-check run status.' };
        return { ok: true, run: Array.isArray(data) && data.length ? data[0] : null };
      });
    }).catch(function (err) {
      return { ok: false, detail: err.message || 'Could not read fact-check run status.' };
    });
    var hash = fetch(base).then(function (res) {
      return readJson(res).then(function (data) {
        var value = res.ok && data && data.script ? data.script.script_sha256 : '';
        return { ok: !!value, value: value || '' };
      });
    }).catch(function () { return { ok: false, value: '' }; });
    return Promise.all([check, run, hash]).then(function (parts) {
      return Factcheck.panelState({ check: parts[0], run: parts[1], hash: parts[2] });
    });
  }

  function chipText(state) {
    if (state.kind === 'done') {
      var high = Factcheck.highSeverityItems(state.claims).length;
      var total = state.claims.length;
      if (high) return 'Fact check: ' + high + ' high-severity · ' + total + ' claims checked';
      return 'Fact check: no high-severity flags · ' + total + ' claims checked';
    }
    return {
      running: 'Fact check running',
      none: 'Not fact-checked yet',
      failed: 'Fact check failed',
      stale: 'Fact check is stale',
      migration: 'Fact check unavailable',
      unavailable: 'Fact check status unavailable',
    }[state.kind] || 'Fact check status unavailable';
  }

  function paintFactcheck(box, scriptId, state) {
    box.replaceChildren();
    box.setAttribute('data-factcheck-state', state.kind);
    var heading = el('p', 'episode-factcheck-title', 'Fact check');
    box.appendChild(heading);
    if (state.kind !== 'done') {
      var status = el('p', 'episode-factcheck-status' + (state.tone ? ' is-' + state.tone : ''), state.message);
      box.appendChild(status);
      if (state.detail) box.appendChild(el('p', 'episode-factcheck-detail', state.detail));
      if (state.kind !== 'migration') {
        var open = el('a', 'episode-factcheck-link', 'Open in Listen');
        open.setAttribute('href', Factcheck.scriptLineHref(scriptId));
        box.appendChild(open);
      }
      return;
    }
    box.appendChild(el('p', 'episode-factcheck-summary', Factcheck.factcheckSummary(state.claims)));
    state.caveats.forEach(function (note) {
      box.appendChild(el('p', 'episode-factcheck-status is-warn', note));
    });
    var items = Factcheck.highSeverityItems(state.claims);
    if (!items.length) {
      box.appendChild(el('p', 'episode-factcheck-status',
        'No high-severity flags. Lower-severity flags, opinions and supported claims are counted above.'));
      return;
    }
    var list = el('ul', 'episode-factcheck-items');
    items.forEach(function (claim) {
      var presentation = Factcheck.claimPresentation(claim);
      var item = el('li', 'episode-factcheck-item is-' + presentation.tone);
      item.appendChild(el('strong', '', presentation.icon + ' ' + presentation.label));
      if (Number.isInteger(claim.line_index)) item.appendChild(el('span', 'factcheck-issue-line', 'L' + claim.line_index));
      var said = Factcheck.truncateText(claim.span_text || claim.claim, TITLE_MAX);
      if (said) item.appendChild(el('q', '', said));
      var evidence = Array.isArray(claim.evidence) ? claim.evidence : [];
      if (evidence.length && evidence[0].quote) {
        item.appendChild(el('blockquote', '', 'Paper: \u201c' + Factcheck.truncateText(evidence[0].quote, QUOTE_MAX) + '\u201d'));
      } else {
        item.appendChild(el('p', 'factcheck-evidence-missing', 'Evidence quote not located.'));
      }
      var link = el('a', 'episode-factcheck-link', 'Open in script');
      link.setAttribute('href', Factcheck.scriptLineHref(scriptId, claim.line_index));
      item.appendChild(link);
      list.appendChild(item);
    });
    box.appendChild(list);
    box.appendChild(el('p', 'list-row-meta', 'High-severity items only. The full list is in Listen.'));
  }

  function bootFactchecks() {
    document.querySelectorAll('[data-factcheck-script]').forEach(function (box) {
      var scriptId = box.getAttribute('data-factcheck-script');
      if (!scriptId) return;
      box.replaceChildren(el('p', 'episode-factcheck-status', 'Loading fact check\u2026'));
      loadEpisodeFactcheck(scriptId).then(function (state) {
        paintFactcheck(box, scriptId, state);
        document.querySelectorAll('[data-factcheck-chip]').forEach(function (chip) {
          if (chip.getAttribute('data-factcheck-chip') !== scriptId) return;
          chip.textContent = chipText(state);
          chip.hidden = false;
        });
      });
    });
  }

  function boot() {
    if (typeof document === 'undefined') return;
    var root = document.querySelector('.paper-page');
    if (!root) return;
    var paperId = root.getAttribute('data-paper-id') || '';

    bootFactchecks();

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
    chipText: chipText,
    ANSWERS_KEY: ANSWERS_KEY,
  };
  global.PaperPage = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
