/*
 * Poll a pipeline run and render its stages.
 *
 * Shared by pipeline.html and discovery.html — both start long runs and both need
 * the same reattach, backoff and cleanup behaviour, and this is too much logic to
 * keep in two inline <script> blocks in sync.
 *
 * Deliberate choices, each of which has a failure mode behind it:
 *
 *  - Recursive setTimeout, never setInterval. setInterval queues the next tick
 *    whether or not the previous response arrived, so a slow server produces
 *    overlapping requests that land out of order and can render stage 4 after
 *    stage 7. Scheduling only after a response is self-limiting.
 *  - Backoff on errors only. A healthy eleven-stage run wants steady one-second
 *    updates; creeping to fifteen seconds because the run is long would be wrong.
 *  - AbortController on stop, so a late response cannot repaint a panel the user
 *    has already moved on from.
 *  - 404 clears the stored run id. Otherwise a run id left in localStorage from a
 *    wiped database polls a ghost forever.
 *  - pagehide rather than beforeunload — beforeunload can disqualify the page from
 *    the back/forward cache, and this cleanup is only cosmetic anyway: the run keeps
 *    going server-side, which is the entire point.
 */
(function (global) {
  'use strict';

  var TERMINAL = ['succeeded', 'failed', 'cancelled', 'interrupted'];
  var POLL_MS = 1000;
  var MAX_BACKOFF_MS = 15000;
  var STORAGE_KEY = 'nlp:activeRun';

  function isTerminal(status) {
    return TERMINAL.indexOf(status) !== -1;
  }

  /* ---------------------------------------------------------------- polling */

  function pollRun(runId, handlers) {
    var delay = POLL_MS;
    var stopped = false;
    var timer = null;
    var controller = new AbortController();

    function stop() {
      if (stopped) return;
      stopped = true;
      clearTimeout(timer);
      controller.abort();
    }

    async function tick() {
      if (stopped) return;
      try {
        var res = await fetch('/api/pipeline-runs/' + encodeURIComponent(runId), {
          signal: controller.signal,
        });

        if (res.status === 404) {
          // The run does not exist — a stale id. Stop rather than retry forever.
          stop();
          if (handlers.onGone) handlers.onGone(runId);
          return;
        }
        if (!res.ok) throw new Error('HTTP ' + res.status);

        var run = await res.json();
        if (handlers.onUpdate) handlers.onUpdate(run);

        if (isTerminal(run.status)) {
          stop();
          if (handlers.onDone) handlers.onDone(run);
          return;
        }
        delay = POLL_MS; // healthy: back to a fast poll
      } catch (err) {
        if (err.name === 'AbortError') return; // deliberate stop, not a failure
        delay = Math.min(delay * 2, MAX_BACKOFF_MS);
        if (handlers.onError) handlers.onError(err);
      }
      timer = setTimeout(tick, delay); // scheduled AFTER the response, never before
    }

    tick();
    return stop;
  }

  /* -------------------------------------------------------------- rendering */

  var STAGE_ICON = {
    pending: '○',
    running: '◐',
    completed: '●',
    failed: '✕',
    skipped: '–',
  };
  var STAGE_COLOR = {
    pending: 'var(--pico-muted-color, #888)',
    running: 'var(--pico-primary, #0a84ff)',
    completed: 'var(--pico-ins-color, #2a9d4a)',
    failed: 'var(--pico-del-color, #d33)',
    skipped: 'var(--pico-muted-color, #888)',
  };

  function duration(stage) {
    if (!stage.started_at) return '';
    var end = stage.finished_at ? new Date(stage.finished_at) : new Date();
    var secs = (end - new Date(stage.started_at)) / 1000;
    if (!isFinite(secs) || secs < 0) return '';
    return secs < 60 ? secs.toFixed(1) + 's'
                     : Math.floor(secs / 60) + 'm' + Math.round(secs % 60) + 's';
  }

  function renderStages(container, run) {
    if (!container) return;
    var rows = (run.stages || []).map(function (s) {
      var icon = STAGE_ICON[s.status] || '○';
      var color = STAGE_COLOR[s.status] || 'inherit';
      var detail = s.detail ? ' <small style="opacity:.7">' + escapeHtml(s.detail) + '</small>' : '';
      var dur = duration(s);
      var durHtml = dur ? '<small style="opacity:.55;float:right">' + dur + '</small>' : '';
      var weight = s.status === 'running' ? 'font-weight:600;' : '';
      return '<li style="list-style:none;color:' + color + ';' + weight + '">' +
             icon + ' ' + escapeHtml(s.name) + detail + durHtml + '</li>';
    }).join('');
    container.innerHTML = '<ul style="padding-left:0;margin:.5rem 0">' + rows + '</ul>';
  }

  function escapeHtml(text) {
    var d = document.createElement('div');
    d.textContent = String(text == null ? '' : text);
    return d.innerHTML;
  }

  function summarise(run) {
    if (run.status === 'running' || run.status === 'pending') {
      return run.current_stage ? 'Running — ' + run.current_stage : 'Starting…';
    }
    if (run.status === 'succeeded') {
      return 'Done — ' + run.papers_processed + ' paper(s) processed' +
             (run.papers_failed ? ', ' + run.papers_failed + ' failed' : '');
    }
    if (run.status === 'interrupted') {
      return 'Interrupted — the server stopped before this run finished';
    }
    if (run.status === 'cancelled') return 'Cancelled';
    return 'Failed' + (run.error ? ' — ' + run.error : '');
  }

  function statusColor(status) {
    if (status === 'succeeded') return 'green';
    if (status === 'failed') return 'crimson';
    if (status === 'interrupted' || status === 'cancelled') return 'darkorange';
    return '';
  }

  /* ------------------------------------------------------------ attach/track */

  var activeStop = null;
  var activeRunId = null;
  var activeEls = null;        // remembered so a hidden->visible resume can rebuild
  var activeOnFinished = null;

  /**
   * Follow a run: render into `els`, remember it across reloads, clean up after.
   * els = { status: HTMLElement, stages: HTMLElement }
   */
  function attach(runId, els, onFinished) {
    detach();
    activeRunId = runId;
    activeEls = els;
    activeOnFinished = onFinished;
    try { localStorage.setItem(STORAGE_KEY, runId); } catch (e) { /* private mode */ }
    // ?run= makes the in-flight run survive a reload and be shareable.
    try {
      var url = new URL(window.location.href);
      url.searchParams.set('run', runId);
      history.replaceState({}, '', url);
    } catch (e) { /* non-fatal */ }

    activeStop = pollRun(runId, {
      onUpdate: function (run) {
        if (els.status) {
          els.status.textContent = summarise(run);
          els.status.style.color = statusColor(run.status);
        }
        renderStages(els.stages, run);
      },
      onDone: function (run) {
        clearStored();
        if (onFinished) onFinished(run);
      },
      onGone: function () {
        clearStored();
        if (els.status) els.status.textContent = '';
        if (els.stages) els.stages.innerHTML = '';
      },
      onError: function () {
        if (els.status) els.status.textContent = 'Lost contact with the server, retrying…';
      },
    });
  }

  function detach() {
    if (activeStop) activeStop();
    activeStop = null;
    activeRunId = null;
    activeEls = null;
    activeOnFinished = null;
  }

  function clearStored() {
    activeStop = null;
    activeRunId = null;
    activeEls = null;
    activeOnFinished = null;
    try { localStorage.removeItem(STORAGE_KEY); } catch (e) { /* ignore */ }
    try {
      var url = new URL(window.location.href);
      url.searchParams.delete('run');
      history.replaceState({}, '', url);
    } catch (e) { /* ignore */ }
  }

  /**
   * On page load, reattach to a run already in flight.
   * ?run= wins over localStorage: an explicit URL is a stronger signal than a
   * leftover. Falls back to asking the server, so opening the page fresh on another
   * device still finds the run.
   */
  async function findActive(pillarId) {
    try {
      var q = pillarId ? '?pillar_id=' + encodeURIComponent(pillarId) : '';
      var res = await fetch('/api/pipeline-runs/active' + q);
      if (!res.ok) return null;
      var run = await res.json();
      return run && run.id ? run : null;
    } catch (e) {
      return null; // nothing in flight, or offline
    }
  }

  /**
   * Reattach to a run already in flight. Returns the run object when one was found
   * by asking the server, the run id when it came from the URL or storage, or null.
   *
   * NOTE the server lookup is deliberately NOT filtered by pillar. Filtering by
   * whatever the pillar dropdown happens to show means a run started for another
   * pillar is invisible after a reload, which is exactly when you most want to see
   * it. For a single-user tool "the active run" is well defined, so callers get told
   * which pillar it belongs to and can sync their own UI.
   */
  async function reattach(els, onFinished) {
    var fromUrl = new URLSearchParams(window.location.search).get('run');
    var stored = null;
    try { stored = localStorage.getItem(STORAGE_KEY); } catch (e) { /* ignore */ }
    var runId = fromUrl || stored;

    if (runId) {
      attach(runId, els, onFinished);
      return runId;
    }

    var run = await findActive();
    if (run) {
      attach(run.id, els, onFinished);
      return run;
    }
    return null;
  }

  // Stop polling while hidden and resume on return, rather than fighting the
  // browser's background-tab throttling.
  document.addEventListener('visibilitychange', function () {
    if (document.visibilityState === 'hidden' && activeStop) {
      activeStop();
      activeStop = null;          // keep activeRunId/activeEls so we can resume
    } else if (
      document.visibilityState === 'visible' && activeRunId && !activeStop && activeEls
    ) {
      // Re-attaching fires an immediate fetch, so the panel catches up at once
      // rather than after one poll interval.
      attach(activeRunId, activeEls, activeOnFinished);
    }
  });
  window.addEventListener('pagehide', function () { if (activeStop) activeStop(); });

  global.RunProgress = {
    attach: attach,
    detach: detach,
    reattach: reattach,
    findActive: findActive,
    isTerminal: isTerminal,
    escapeHtml: escapeHtml,
  };
})(window);
