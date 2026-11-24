/*
 * Daily run page: save the settings, start a run now.
 *
 * Saving is a PUT /api/daily-run; the scheduler container re-reads the row
 * within a minute. Turning the run ON asks first, with the estimated daily
 * cost, because from then on it spends money every day. "Run now" is a POST
 * that spend-guard.js intercepts with its own confirm dialog and budget.
 * Text goes in with textContent only.
 */
(function (global) {
  'use strict';

  function money(value) {
    var n = Number(value);
    return isFinite(n) ? '$' + n.toFixed(2) : '$?';
  }

  /** The question asked before the run is turned on. */
  function enableMessage(runTime, timezone, estimateUsd, capUsd) {
    return 'Turn on the daily run? It will run every day at ' + runTime + ' (' + timezone + ') ' +
      'and spend about ' + money(estimateUsd) + ' a day on Claude calls, never more than the ' +
      money(capUsd) + ' daily cap.';
  }

  /** A readable message from an error response body (FastAPI detail or ours). */
  function errorText(body, status) {
    if (body && typeof body.detail === 'string') return body.detail;
    if (body && Array.isArray(body.detail) && body.detail.length) {
      return body.detail.map(function (d) { return d && d.msg ? d.msg : String(d); }).join('; ');
    }
    if (body && typeof body.error === 'string') return body.error;
    return 'Request failed (HTTP ' + status + ').';
  }

  /** Summary line after Run now answered 202. */
  function runNowSummary(payload) {
    var runs = (payload && payload.runs) || [];
    var started = runs.filter(function (r) { return r.run_id; }).length;
    var skipped = runs.length - started;
    var text = 'Started ' + started + ' run' + (started === 1 ? '' : 's') + '.';
    if (skipped) text += ' ' + skipped + ' not started; see Last run.';
    return text;
  }

  function readJson(response) {
    return response.json().catch(function () { return null; });
  }

  function init(doc) {
    var form = doc.getElementById('daily-run-form');
    if (!form) return;
    var data = {};
    try { data = JSON.parse(doc.getElementById('daily-run-data').textContent || '{}'); } catch (e) {}
    var errorEl = doc.getElementById('dr-error');
    var statusEl = doc.getElementById('dr-status');
    var saveBtn = doc.getElementById('dr-save');
    var runBtn = doc.getElementById('daily-run-now');

    function showError(text) {
      errorEl.textContent = text;
      errorEl.hidden = !text;
    }

    function settings() {
      var papers = {};
      Array.prototype.forEach.call(form.querySelectorAll('.dr-papers'), function (input) {
        papers[input.getAttribute('data-pillar-id')] = Number(input.value);
      });
      var pillarIds = Array.prototype.filter.call(
        form.querySelectorAll('.dr-pillar'), function (box) { return box.checked; }
      ).map(function (box) { return box.value; });
      return {
        enabled: doc.getElementById('dr-enabled').checked,
        run_time: doc.getElementById('dr-time').value,
        timezone: doc.getElementById('dr-timezone').value.trim(),
        pillar_ids: pillarIds,
        papers_per_day: papers
      };
    }

    form.addEventListener('submit', function (event) {
      event.preventDefault();
      showError('');
      var body = settings();
      if (body.enabled && !data.enabled &&
          !global.confirm(enableMessage(body.run_time, body.timezone, data.estimate_usd, data.cap_usd))) {
        statusEl.textContent = 'Not saved.';
        return;
      }
      saveBtn.disabled = true;
      statusEl.textContent = 'Saving…';
      global.fetch('/api/daily-run', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      }).then(function (response) {
        return readJson(response).then(function (json) {
          if (!response.ok) throw new Error(errorText(json, response.status));
          statusEl.textContent = 'Saved.';
          global.location.reload();
        });
      }).catch(function (err) {
        statusEl.textContent = '';
        showError(err.message || String(err));
        saveBtn.disabled = false;
      });
    });

    if (runBtn) {
      runBtn.addEventListener('click', function () {
        showError('');
        runBtn.disabled = true;
        statusEl.textContent = 'Starting…';
        global.fetch('/api/daily-run/run-now', { method: 'POST' }).then(function (response) {
          return readJson(response).then(function (json) {
            if (response.status !== 202) throw new Error(errorText(json, response.status));
            statusEl.textContent = runNowSummary(json);
            global.setTimeout(function () { global.location.reload(); }, 1200);
          });
        }).catch(function (err) {
          statusEl.textContent = '';
          showError(err.message || String(err));
          runBtn.disabled = false;
        });
      });
    }
  }

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', function () { init(document); });
    } else {
      init(document);
    }
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { enableMessage: enableMessage, errorText: errorText, runNowSummary: runNowSummary };
  }
})(typeof window !== 'undefined' ? window : globalThis);
