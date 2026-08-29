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
 *    https://developer.mozilla.org/en-US/docs/Web/API/Window/setInterval
 *  - Backoff on errors only. A healthy eleven-stage run wants steady one-second
 *    updates; creeping to fifteen seconds because the run is long would be wrong.
 *  - AbortController on stop, so a late response cannot repaint a panel the user
 *    has already moved on from.
 *  - 404 clears the stored run id, but 5xx does NOT. The server now distinguishes
 *    "no such run" (404) from "could not reach the database" (503); treating the
 *    latter as "gone" is how a single blip used to detach the UI from a live run.
 *  - onGone still calls onFinished. It did not, and onFinished is the only thing
 *    that re-enables the page's button, so a stale run id left the button disabled
 *    on "Running…" forever with a blank panel and no way out but a reload.
 *  - pagehide rather than beforeunload — beforeunload can disqualify the page from
 *    the back/forward cache, and this cleanup is only cosmetic anyway: the run keeps
 *    going server-side, which is the entire point.
 *  - Rows are built with createElement + textContent, never innerHTML. The
 *    escapeHtml() helper below escapes `&<>` but NOT quotes (it serializes a text
 *    node, and the HTML spec only escapes quotes in *attribute* mode), so
 *    `title="${escapeHtml(x)}"` is still injectable. Measured: a stage detail of
 *    `a" onmouseover="…` produced a live onmouseover attribute. Do not reintroduce
 *    template-literal HTML here.
 *    https://cheatsheetseries.owasp.org/cheatsheets/DOM_based_XSS_Prevention_Cheat_Sheet.html
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

  /**
   * Call a caller's handler without letting it break the poll.
   *
   * The handlers render; the poll is transport. Running them inside tick()'s try
   * meant any rendering bug was caught by the transport catch and reported as
   * "Lost contact with the server — retrying", which is a lie, and — because the
   * throw jumped over the isTerminal check — left the page polling a finished run
   * for as long as it stayed open. Measured on a discovery run whose stored
   * candidates were the wrong shape: a TypeError in the candidates table produced
   * exactly that, with no console error to say otherwise.
   *
   * Same reasoning as the orchestrator's on_stage sink (AGENTS.md, "Long runs are
   * background jobs"): losing the progress display is bad, losing the run with it
   * is worse. Logged rather than swallowed — a render bug should still be findable.
   */
  function safely(fn, arg, what) {
    if (!fn) return;
    try {
      fn(arg);
    } catch (err) {
      // `global` is the IIFE's parameter (window in a browser, globalThis under
      // node --test), not a bare `window` — which would itself throw in node.
      if (global.console && global.console.error) {
        global.console.error('run-progress: ' + what + ' handler failed', err);
      }
    }
  }

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
          // The run genuinely does not exist — a stale id. Stop rather than retry
          // forever. Note this is now ONLY sent for a real absence: a database
          // failure answers 503 and falls through to the retry path below.
          stop();
          safely(handlers.onGone, runId, 'onGone');
          return;
        }
        if (!res.ok) throw new Error('HTTP ' + res.status);

        var run = await res.json();
        safely(handlers.onUpdate, run, 'onUpdate');

        if (isTerminal(run.status)) {
          stop();
          safely(handlers.onDone, run, 'onDone');
          return;
        }
        delay = POLL_MS; // healthy: back to a fast poll
      } catch (err) {
        if (err.name === 'AbortError') return; // deliberate stop, not a failure
        delay = Math.min(delay * 2, MAX_BACKOFF_MS);
        safely(handlers.onError, err, 'onError');
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
    unknown: '⚠',
  };
  var STAGE_COLOR = {
    pending: 'var(--pico-muted-color, #888)',
    running: 'var(--pico-primary, #0a84ff)',
    completed: 'var(--pico-ins-color, #2a9d4a)',
    failed: 'var(--pico-del-color, #d33)',
    skipped: 'var(--pico-muted-color, #888)',
    unknown: 'var(--pico-del-color, #d33)',
  };

  //: Stage names are StageName enum values from nlp_pillars/schemas.py — machine
  //: slugs. They were rendered raw, so the user read "pop_queue" and
  //: "Running — pop_queue". Anything not in this map falls back to the raw name, so
  //: a new stage degrades to readable-ish rather than blank.
  var STAGE_LABEL = {
    discovery: 'Choosing search queries',
    search: 'Searching for papers',
    enqueue: 'Adding to the queue',
    pop_queue: 'Taking papers off the queue',
    process: 'Processing papers',
    ingest: 'Downloading and reading the PDF',
    summarize: 'Summarizing',
    synthesize: 'Writing the lesson',
    quiz: 'Building quiz cards',
    persist: 'Saving to the database',
    vectors: 'Indexing for search',
    // run_discovery_with_selection. Named for what the user gets out of each step,
    // not for the function that does it.
    discover_context: 'Reading your recent papers',
    discover_queries: 'Writing search queries',
    discover_vectors: 'Searching your library by meaning',
    discover_arxiv: 'Searching arXiv',
    discover_semantic_scholar: 'Searching Semantic Scholar',
    discover_citations: 'Following citations from your recent papers',
    discover_rank: 'Ranking and removing duplicates',
  };

  function stageLabel(name) {
    return STAGE_LABEL[name] || name;
  }

  function duration(stage) {
    if (!stage.started_at) return '';
    var end = stage.finished_at ? new Date(stage.finished_at) : new Date();
    var secs = (end - new Date(stage.started_at)) / 1000;
    if (!isFinite(secs) || secs < 0) return '';
    return secs < 60 ? secs.toFixed(1) + 's'
                     : Math.floor(secs / 60) + 'm' + Math.round(secs % 60) + 's';
  }

  /**
   * A stage's status as it should be DISPLAYED, which is not always what is stored.
   *
   * Nothing rewrites child stage rows when a run dies: the interrupted sweep updates
   * only the parent row, and a paper that fails leaves whichever stage it was in
   * marked running until some later paper re-enters that stage. So a finished run can
   * carry a stage still saying "running", which renders as a bold spinner on a run
   * that ended minutes ago. Show that honestly as unknown instead of pretending work
   * is in flight.
   */
  function displayStatus(stage, run) {
    if (stage.status === 'running' && isTerminal(run.status)) {
      return run.status === 'succeeded' ? 'completed' : 'unknown';
    }
    return stage.status;
  }

  /**
   * How many stages are done, counted the way they are DISPLAYED.
   *
   * Must go through displayStatus for the same reason renderStages does, or the bar
   * and the list disagree: db.update_pipeline_run_stage() swallows its own failures,
   * so a completed stage's write can be silently lost and the row left at 'running'.
   * The list then shows eleven green rows (displayStatus resolves them) while a bar
   * counting raw status sticks at 10/11 and never fills.
   *
   * Skipped counts as done — quiz is skipped when disabled, and a bar that can never
   * reach the end because of a deliberate skip reads as a stall.
   */
  function countCompleted(run) {
    return (run.stages || []).filter(function (s) {
      var status = displayStatus(s, run);
      return status === 'completed' || status === 'skipped';
    }).length;
  }

  function renderStages(container, run) {
    if (!container) return;
    var frag = document.createDocumentFragment();

    (run.stages || []).forEach(function (s) {
      var status = displayStatus(s, run);
      var li = document.createElement('li');
      li.style.listStyle = 'none';
      li.style.color = STAGE_COLOR[status] || 'inherit';
      if (status === 'running') li.style.fontWeight = '600';

      // Icon and label. textContent everywhere — see the header note on escapeHtml.
      li.appendChild(document.createTextNode((STAGE_ICON[status] || '○') + ' '));

      var name = document.createElement('span');
      name.textContent = stageLabel(s.name);
      li.appendChild(name);

      if (s.detail) {
        var detail = document.createElement('small');
        detail.style.opacity = '.7';
        detail.textContent = ' ' + s.detail;
        li.appendChild(detail);
      }
      if (status === 'unknown') {
        var note = document.createElement('small');
        note.style.opacity = '.7';
        note.textContent = ' (never finished)';
        li.appendChild(note);
      }

      var dur = duration(s);
      if (dur) {
        var d = document.createElement('small');
        d.style.opacity = '.55';
        d.style.float = 'right';
        d.textContent = dur;
        li.appendChild(d);
      }
      frag.appendChild(li);
    });

    var ul = document.createElement('ul');
    ul.style.paddingLeft = '0';
    ul.style.margin = '.5rem 0';
    ul.appendChild(frag);
    container.replaceChildren(ul);
  }

  /**
   * Update the <progress> bar, if the page gave us one.
   *
   * Indeterminate while pending, because "0 of 11" reads as stalled when the truth is
   * "not started". removeAttribute is the only way back to indeterminate — setting
   * value to null coerces to 0 and stays determinate.
   * https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/progress
   */
  function renderProgress(el, run) {
    if (!el) return;
    var total = (run.stages || []).length;
    var done = countCompleted(run);

    if (!total || run.status === 'pending') {
      el.removeAttribute('value');
    } else {
      el.max = total;
      el.value = Math.min(done, total);
    }

    // Pico funnels every vendor pseudo-element through one custom property, so one
    // line recolours the bar in all engines. https://picocss.com/docs/progress
    var color = 'var(--pico-primary-background)';
    if (run.status === 'failed') color = 'var(--pico-del-color)';
    else if (run.status === 'succeeded') color = 'var(--pico-ins-color)';
    else if (run.status === 'cancelled' || run.status === 'interrupted') {
      color = 'var(--pico-muted-color)';
    }
    el.style.setProperty('--pico-progress-color', color);
    el.hidden = false;
  }

  function escapeHtml(text) {
    // Kept for callers that still build body text. NOT safe for attribute contexts:
    // it serializes a text node, which escapes & < > and leaves both quote
    // characters untouched. See the header comment.
    var d = document.createElement('div');
    d.textContent = String(text == null ? '' : text);
    return d.innerHTML;
  }

  /**
   * How many candidates a finished discovery run found.
   *
   * Read from the stored payload rather than from papers_processed, which a discovery
   * run leaves at 0 on purpose: it processed no papers, it found some. Conflating the
   * two would put "10 papers processed" on a run that ingested nothing.
   */
  function candidateCount(run) {
    var result = run && run.result;
    return result && Array.isArray(result.candidates) ? result.candidates.length : 0;
  }

  function summarise(run) {
    var total = (run.stages || []).length;
    var done = countCompleted(run);

    if (run.status === 'running' || run.status === 'pending') {
      if (!run.current_stage) return 'Starting…';
      return 'Step ' + Math.min(done + 1, total) + ' of ' + total + ' — ' +
             stageLabel(run.current_stage);
    }
    if (run.kind === 'discover' && run.status === 'succeeded') {
      // A discovery run counts candidates, not processed papers, and it reports the
      // sources that failed even when it succeeded: "ten papers, but arXiv was
      // rate-limited" is a different claim from "ten papers", and only the user can
      // decide whether that is worth a retry.
      var found = candidateCount(run);
      var msg = found
        ? 'Done — ' + found + ' candidate paper(s) found'
        : 'Done — no papers matched';
      if (run.error) msg += ' — ' + run.error;
      return msg;
    }
    if (run.status === 'succeeded') {
      // A run that found nothing is recorded as succeeded (it is not a failure), but
      // "Done — 0 paper(s) processed" reads like something went wrong. Say what
      // actually happened.
      if (!run.papers_processed && !run.papers_failed) {
        return 'Done — no new papers to process';
      }
      var msg = 'Done — ' + run.papers_processed + ' paper(s) processed';
      if (run.papers_failed) {
        msg += ', ' + run.papers_failed + ' failed';
        if (run.error) msg += ' — ' + run.error;
      }
      return msg;
    }
    if (run.status === 'interrupted') {
      return 'Interrupted — the server stopped before this run finished';
    }
    if (run.status === 'cancelled') return 'Cancelled';
    return 'Failed' + (run.error ? ' — ' + run.error : '');
  }

  function statusColor(run) {
    var status = typeof run === 'string' ? run : run && run.status;
    var partial = typeof run === 'object' && run && run.papers_failed > 0;
    // A succeeded run that still lost papers is not a clean success. Painting it the
    // same green as a perfect run is how a partial failure went unnoticed.
    if (status === 'succeeded') return partial ? 'darkorange' : 'green';
    if (status === 'failed') return 'crimson';
    if (status === 'interrupted' || status === 'cancelled') return 'darkorange';
    return '';
  }

  /* ------------------------------------------------------------ attach/track */

  var activeStop = null;
  var activeRunId = null;
  var activeEls = null;        // remembered so a hidden->visible resume can rebuild
  var activeOnFinished = null;
  var activeOnRun = null;
  var lastAnnounced = '';

  /**
   * Write the one-sentence summary into the live region, but only when it changed.
   *
   * The stage list is deliberately not a live region: re-announcing eleven items
   * every second is unusable. A small, separate, initially-empty region carrying one
   * sentence is the documented pattern.
   * https://www.w3.org/WAI/WCAG22/Techniques/aria/ARIA22
   */
  function announce(el, sentence) {
    if (!el || sentence === lastAnnounced) return;
    lastAnnounced = sentence;
    el.textContent = sentence;
  }

  /**
   * Follow a run: render into `els`, remember it across reloads, clean up after.
   * els = { status, stages, progress, cancel } — every one optional.
   * onFinished(run) fires on a terminal run AND on a vanished one (run === null).
   * onRun(run) fires on every poll, so a page can sync UI (e.g. the pillar dropdown)
   * to the run it is actually displaying.
   */
  function attach(runId, els, onFinished, onRun) {
    detach();
    activeRunId = runId;
    activeEls = els;
    activeOnFinished = onFinished;
    activeOnRun = onRun;
    lastAnnounced = '';
    try { localStorage.setItem(STORAGE_KEY, runId); } catch (e) { /* private mode */ }
    // ?run= makes the in-flight run survive a reload and be shareable.
    try {
      var url = new URL(window.location.href);
      url.searchParams.set('run', runId);
      history.replaceState({}, '', url);
    } catch (e) { /* non-fatal */ }

    if (els.cancel) {
      els.cancel.hidden = false;
      els.cancel.disabled = false;
      els.cancel.dataset.runId = runId;
    }

    activeStop = pollRun(runId, {
      onUpdate: function (run) {
        announce(els.status, summarise(run));
        if (els.status) els.status.style.color = statusColor(run);
        renderProgress(els.progress, run);
        renderStages(els.stages, run);
        if (onRun) onRun(run);
      },
      onDone: function (run) {
        clearStored();
        hideCancel(els);
        if (onFinished) onFinished(run);
      },
      onGone: function () {
        clearStored();
        hideCancel(els);
        // Say so rather than blanking the panel: an empty panel is indistinguishable
        // from "nothing has happened yet".
        announce(els.status, 'This run no longer exists — it may have been cleared.');
        if (els.status) els.status.style.color = 'darkorange';
        if (els.progress) els.progress.hidden = true;
        if (els.stages) els.stages.replaceChildren();
        // MUST fire. onFinished is what re-enables the page's button; without it a
        // stale run id left the UI disabled on "Running…" with no way forward.
        if (onFinished) onFinished(null);
      },
      onError: function () {
        announce(els.status, 'Lost contact with the server — retrying. The run is '
                             + 'still going.');
        if (els.status) els.status.style.color = 'darkorange';
      },
    });
  }

  function hideCancel(els) {
    if (els && els.cancel) {
      els.cancel.hidden = true;
      els.cancel.disabled = true;
    }
  }

  function detach() {
    if (activeStop) activeStop();
    activeStop = null;
    activeRunId = null;
    activeEls = null;
    activeOnFinished = null;
    activeOnRun = null;
  }

  function clearStored() {
    activeStop = null;
    activeRunId = null;
    activeEls = null;
    activeOnFinished = null;
    activeOnRun = null;
    try { localStorage.removeItem(STORAGE_KEY); } catch (e) { /* ignore */ }
    try {
      var url = new URL(window.location.href);
      url.searchParams.delete('run');
      history.replaceState({}, '', url);
    } catch (e) { /* ignore */ }
  }

  /**
   * Ask the server to stop a run. Cooperative: it finishes the stage it is in.
   */
  async function cancel(runId) {
    var res = await fetch('/api/pipeline-runs/' + encodeURIComponent(runId) + '/cancel',
                          { method: 'POST' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    return res.json();
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
   * Reattach to a run already in flight. Returns the run id when one was found, or
   * null. Callers that need the run's pillar should use the `onRun` callback rather
   * than this return value — the id may come from the URL or storage, where the
   * pillar is not yet known.
   *
   * NOTE the server lookup is deliberately NOT filtered by pillar. Filtering by
   * whatever the pillar dropdown happens to show means a run started for another
   * pillar is invisible after a reload, which is exactly when you most want to see
   * it. For a single-user tool "the active run" is well defined, so callers get told
   * which pillar it belongs to (via onRun) and can sync their own UI.
   */
  async function reattach(els, onFinished, onRun) {
    var fromUrl = new URLSearchParams(window.location.search).get('run');
    var stored = null;
    try { stored = localStorage.getItem(STORAGE_KEY); } catch (e) { /* ignore */ }
    var runId = fromUrl || stored;

    if (runId) {
      attach(runId, els, onFinished, onRun);
      return runId;
    }

    var run = await findActive();
    if (run) {
      attach(run.id, els, onFinished, onRun);
      return run.id;
    }
    return null;
  }

  // Stop polling while hidden and resume on return, rather than fighting the
  // browser's background-tab throttling.
  //
  // Guarded because this file is also require()d by `node --test`, where there is no
  // document and a bare addEventListener would throw at import time.
  if (typeof document !== 'undefined') {
    document.addEventListener('visibilitychange', function () {
      if (document.visibilityState === 'hidden' && activeStop) {
        activeStop();
        activeStop = null;        // keep activeRunId/activeEls so we can resume
      } else if (
        document.visibilityState === 'visible' && activeRunId && !activeStop && activeEls
      ) {
        // Re-attaching fires an immediate fetch, so the panel catches up at once
        // rather than after one poll interval.
        attach(activeRunId, activeEls, activeOnFinished, activeOnRun);
      }
    });
    window.addEventListener('pagehide', function () { if (activeStop) activeStop(); });
  }

  global.RunProgress = {
    attach: attach,
    detach: detach,
    reattach: reattach,
    findActive: findActive,
    cancel: cancel,
    isTerminal: isTerminal,
    escapeHtml: escapeHtml,
    stageLabel: stageLabel,
    summarise: summarise,
    statusColor: statusColor,
    countCompleted: countCompleted,
    displayStatus: displayStatus,
    duration: duration,
    candidateCount: candidateCount,
  };

  // Also reachable from `node --test` (tests/js/). CommonJS on purpose: the browser
  // loads this file with a plain <script> tag and there is no build step, so it
  // cannot be an ES module. Only the DOM-free helpers are exported here.
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      isTerminal: isTerminal,
      stageLabel: stageLabel,
      summarise: summarise,
      statusColor: statusColor,
      countCompleted: countCompleted,
      displayStatus: displayStatus,
      duration: duration,
      candidateCount: candidateCount,
      safely: safely,
    };
  }
})(typeof window !== 'undefined' ? window : globalThis);
