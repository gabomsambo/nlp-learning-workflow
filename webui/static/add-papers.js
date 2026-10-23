/*
 * Add papers: Find (discovery), Add by URL / PDF (shared upload panel), and
 * Auto-run (the daily pipeline). One in-flow run card for every kind.
 *
 * The selectedPapers object posted to /select is the paper's permanent record.
 * Shorten a title or abstract in renderCandidates, never in that object.
 */
(function (global) {
  'use strict';

  var OWNED = {
    discover: 'find',
    process_selected: 'find',
    upload: 'upload',
    run_daily: 'auto',
  };

  var RUN_TITLE = {
    discover: 'Discovery run',
    process_selected: 'Processing run',
    upload: 'Upload',
    run_daily: 'Pipeline run',
  };

  function paperPageHref(id) {
    if (typeof id !== 'string' || !id) return null;
    return '/papers/' + encodeURIComponent(id);
  }

  /**
   * True only when `ids` is a known set and contains the paper.
   * null means the library list could not be read: unknown is not "absent",
   * so the caller must not paint an "in library" chip.
   */
  function inLibrary(paperId, ids) {
    if (!ids || typeof paperId !== 'string' || !paperId) return false;
    return ids.has(paperId);
  }

  function modeFromSearch(search, fallback) {
    var params = new URLSearchParams(String(search || '').replace(/^\?/, ''));
    var mode = params.get('mode');
    if (mode === 'find' || mode === 'upload' || mode === 'auto') return mode;
    if (fallback === 'upload' || fallback === 'auto' || fallback === 'find') return fallback;
    return 'find';
  }

  function safePdfHref(url) {
    if (typeof url !== 'string' || !url.trim()) return null;
    var scheme = /^([a-z][a-z0-9+.-]*):/i.exec(url.trim());
    if (!scheme) return null;
    var s = scheme[1].toLowerCase();
    if (s === 'http' || s === 'https' || s === 'file') return url.trim();
    return null;
  }

  function authorsText(authors) {
    if (!Array.isArray(authors) || !authors.length) return '';
    return authors.filter(function (name) {
      return name != null && String(name).trim();
    }).join(', ');
  }

  function init(opts) {
    var root = document.getElementById('add-papers');
    var initial = (opts && opts.initialMode) || (root && root.dataset.mode) || 'find';

    var candidates = [];
    var attemptedIds = new Set();
    var totalProcessed = 0;
    var totalFailed = 0;
    var pendingSelection = null;
    var libraryIds = null;

    var pillarSelect = document.getElementById('pillar-select');
    var discoverForm = document.getElementById('discover-form');
    var topicsInput = document.getElementById('topics-input');
    var statusEl = document.getElementById('status');
    var resultsArea = document.getElementById('results-area');
    var resultsCount = document.getElementById('results-count');
    var candidatesBody = document.getElementById('candidates-body');
    var selectAllCheckbox = document.getElementById('select-all');
    var processBtn = document.getElementById('process-btn');
    var selectedCount = document.getElementById('selected-count');
    var sessionSummary = document.getElementById('session-summary');
    var totalProcessedEl = document.getElementById('total-processed');
    var totalFailedEl = document.getElementById('total-failed');
    var runPanel = document.getElementById('run-panel');
    var runTitleEl = document.getElementById('run-title');
    var runStatusEl = document.getElementById('run-status');
    var runProgressEl = document.getElementById('run-progress');
    var runBarEl = document.getElementById('run-bar');
    var cancelBtn = document.getElementById('cancel-btn');
    var discoverBtn = document.getElementById('discover-btn');
    var queriesPanel = document.getElementById('queries-panel');
    var queriesText = document.getElementById('queries-text');
    var runForm = document.getElementById('run-form');
    var runBtn = document.getElementById('run-btn');
    var papersInput = document.getElementById('papers-input');

    var progressEls = {
      status: runStatusEl,
      stages: runProgressEl,
      progress: runBarEl,
      cancel: cancelBtn,
    };

    function paintLine(el, tone) {
      if (!el) return;
      el.classList.remove('is-ok', 'is-warn', 'is-danger');
      if (tone) el.classList.add(tone);
    }

    function setMode(mode, writeUrl) {
      if (mode !== 'find' && mode !== 'upload' && mode !== 'auto') mode = 'find';
      document.querySelectorAll('[data-add-mode]').forEach(function (el) {
        el.hidden = el.getAttribute('data-add-mode') !== mode;
      });
      var radio = document.querySelector('input[name="add-mode"][value="' + mode + '"]');
      if (radio) radio.checked = true;
      if (root) root.dataset.mode = mode;
      if (!writeUrl) return;
      try {
        var url = new URL(global.location.href);
        url.searchParams.set('mode', mode);
        history.replaceState({}, '', url);
      } catch (e) { /* non-fatal */ }
    }

    function rememberPillar() {
      if (!pillarSelect || !pillarSelect.value) return;
      try {
        var url = new URL(global.location.href);
        url.searchParams.set('pillar', pillarSelect.value);
        history.replaceState({}, '', url);
      } catch (e) { /* non-fatal */ }
    }

    function setDiscovering(active) {
      if (!discoverBtn) return;
      discoverBtn.disabled = active;
      discoverBtn.textContent = active ? 'Discovering…' : 'Discover papers';
    }

    function setPipelineBusy(busy) {
      if (!runBtn) return;
      runBtn.disabled = busy;
      runBtn.textContent = busy ? 'Running…' : 'Run pipeline';
    }

    function showRun(kind) {
      if (runPanel) runPanel.hidden = false;
      if (runTitleEl) runTitleEl.textContent = RUN_TITLE[kind] || 'Run';
    }

    function renderQueries(run) {
      var stage = (run.stages || []).find(function (s) { return s.name === 'discover_queries'; });
      if (!queriesPanel || !queriesText) return;
      if (!stage || !stage.detail) {
        queriesPanel.hidden = true;
        return;
      }
      queriesText.textContent = stage.detail;
      RunProgress.applyStatusClass(queriesText, stage.status === 'failed' ? 'run-warn' : '');
      queriesPanel.hidden = false;
    }

    function metaSpan(text) {
      var span = document.createElement('span');
      span.textContent = text;
      return span;
    }

    function renderCandidates() {
      var available = candidates.filter(function (c) {
        return c && c.paper && !attemptedIds.has(c.paper.id);
      });
      if (resultsCount) resultsCount.textContent = '(' + available.length + ' available)';
      if (!candidatesBody) return;

      if (available.length === 0) {
        var empty = document.createElement('p');
        empty.className = 'empty-message';
        empty.textContent = 'No more candidates available';
        candidatesBody.replaceChildren(empty);
        if (processBtn) processBtn.disabled = true;
        return;
      }

      var frag = document.createDocumentFragment();
      available.forEach(function (candidate) {
        var paper = candidate.paper || {};
        var row = document.createElement('article');
        row.className = 'candidate';

        var checkWrap = document.createElement('label');
        checkWrap.className = 'candidate-check';
        var box = document.createElement('input');
        box.type = 'checkbox';
        box.className = 'paper-checkbox';
        box.dataset.id = paper.id;
        var titleText = String(paper.title || '');
        box.setAttribute('aria-label', 'Select ' + (titleText || paper.id || 'paper'));
        checkWrap.appendChild(box);

        var year = document.createElement('p');
        year.className = 'candidate-year';
        year.textContent = paper.year ? String(paper.year) : '—';

        var main = document.createElement('div');
        main.className = 'candidate-main';
        var title = document.createElement('h3');
        title.className = 'candidate-title';
        title.textContent = titleText || 'Untitled paper';

        var authors = document.createElement('p');
        authors.className = 'candidate-authors';
        authors.textContent = authorsText(paper.authors);

        var meta = document.createElement('p');
        meta.className = 'candidate-meta';
        if (candidate.source) meta.appendChild(metaSpan(String(candidate.source)));
        if (candidate.citation_count != null) {
          meta.appendChild(metaSpan(String(candidate.citation_count) + ' citations'));
        }
        var score = Number(candidate.relevance_score);
        if (!Number.isNaN(score)) meta.appendChild(metaSpan('Score ' + score.toFixed(2)));

        var href = safePdfHref(paper.url_pdf);
        if (href) {
          var link = document.createElement('a');
          link.href = href;
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
          link.textContent = 'PDF';
          meta.appendChild(link);
        } else {
          meta.appendChild(metaSpan('No PDF'));
        }

        if (inLibrary(paper.id, libraryIds)) {
          var pageHref = paperPageHref(paper.id);
          if (pageHref) {
            var flag = document.createElement('a');
            flag.className = 'chip chip-ok';
            flag.href = pageHref;
            flag.textContent = 'In library';
            meta.appendChild(flag);
          }
        }

        main.append(title, authors, meta);

        if (paper.abstract) {
          var details = document.createElement('details');
          details.className = 'candidate-abstract';
          var summary = document.createElement('summary');
          summary.textContent = 'Abstract';
          var body = document.createElement('p');
          body.textContent = String(paper.abstract);
          details.append(summary, body);
          main.appendChild(details);
        }

        row.append(checkWrap, year, main);
        frag.appendChild(row);
      });
      candidatesBody.replaceChildren(frag);
      updateSelectedCount();
    }

    function updateSelectedCount() {
      var checked = document.querySelectorAll('.paper-checkbox:checked').length;
      if (selectedCount) selectedCount.textContent = checked + ' selected';
      if (processBtn) processBtn.disabled = checked === 0;
    }

    function onDiscoveryFinished(run) {
      setDiscovering(false);
      var result = run.result || {};
      candidates = Array.isArray(result.candidates) ? result.candidates : [];
      attemptedIds = new Set();

      if (!candidates.length) {
        // Deliberately not "no papers found": the panel above says whether the
        // sources answered or fell over.
        statusEl.textContent = run.status === 'succeeded'
          ? 'No candidates matched. Try different topic hints.'
          : 'Discovery did not finish — see the run panel above.';
        paintLine(statusEl, run.status === 'succeeded' ? '' : 'is-danger');
        resultsArea.hidden = true;
        return;
      }

      var sources = (result.sources_used || []).join(', ');
      statusEl.textContent = 'Found ' + candidates.length + ' candidates'
        + (sources ? ' from: ' + sources : '');
      paintLine(statusEl, 'is-ok');
      renderCandidates();
      resultsArea.hidden = false;
    }

    function onRunUpdate(run) {
      if (!run) return;
      if (!OWNED[run.kind]) {
        // Not this page's run. Stop painting it, but leave the stored id alone
        // so the page that owns it can still reattach.
        if (runStatusEl) runStatusEl.textContent = '';
        if (runPanel) runPanel.hidden = true;
        RunProgress.detach();
        return;
      }
      if (run.pillar_id && pillarSelect && pillarSelect.value !== run.pillar_id) {
        var previous = pillarSelect.value;
        pillarSelect.value = run.pillar_id;
        if (pillarSelect.value !== previous) pillarSelect.dispatchEvent(new Event('change'));
      }
      setMode(OWNED[run.kind], true);
      showRun(run.kind);

      var live = !RunProgress.isTerminal(run.status);
      if (run.kind === 'discover') {
        renderQueries(run);
        if (live) setDiscovering(true);
      } else if (queriesPanel) {
        queriesPanel.hidden = true;
      }

      if (run.kind === 'process_selected' && live) {
        if (discoverBtn) discoverBtn.disabled = true;
        if (processBtn) {
          processBtn.disabled = true;
          processBtn.textContent = 'Processing…';
        }
      }
      if (run.kind === 'run_daily' && live) setPipelineBusy(true);
      if (run.kind === 'upload' && global.PillarUpload && global.PillarUpload.onRun) {
        global.PillarUpload.onRun(run);
      }
    }

    function onRunFinished(run) {
      if (processBtn) {
        processBtn.disabled = false;
        processBtn.textContent = 'Process selected';
      }
      setDiscovering(false);
      setPipelineBusy(false);
      updateSelectedCount();
      if (global.PillarUpload && global.PillarUpload.onFinished) {
        global.PillarUpload.onFinished(run && run.kind === 'upload' ? run : null);
      }
      if (!run) {
        pendingSelection = null;
        return;
      }
      if (run.kind === 'discover') {
        onDiscoveryFinished(run);
        return;
      }
      if (run.kind === 'run_daily') return;
      if (run.kind !== 'process_selected') return;

      var attempted = pendingSelection ? pendingSelection.count : run.papers_processed;
      pendingSelection = null;
      totalProcessed += run.papers_processed || 0;
      totalFailed += run.papers_failed || Math.max(0, attempted - (run.papers_processed || 0));
      if (sessionSummary) sessionSummary.hidden = false;
      if (totalProcessedEl) totalProcessedEl.textContent = String(totalProcessed);
      if (totalFailedEl) totalFailedEl.textContent = String(totalFailed);
    }

    function attachOrConflict(res, data, pillar) {
      if (res.status === 409) {
        runStatusEl.textContent = data.detail || data.error || 'A run is already in progress for this pillar.';
        RunProgress.applyStatusClass(runStatusEl, 'run-warn');
        setDiscovering(false);
        setPipelineBusy(false);
        if (processBtn) {
          processBtn.disabled = false;
          processBtn.textContent = 'Process selected';
        }
        return RunProgress.findActive(pillar).then(function (active) {
          return active || RunProgress.findActive();
        }).then(function (active) {
          if (active) RunProgress.attach(active.id, progressEls, onRunFinished, onRunUpdate);
        });
      }
      if (!res.ok || !data.run_id) {
        throw new Error(data.detail || data.error || ('HTTP ' + res.status));
      }
      RunProgress.attach(data.run_id, progressEls, onRunFinished, onRunUpdate);
      return Promise.resolve();
    }

    discoverForm.addEventListener('submit', function (e) {
      e.preventDefault();
      var pillar = pillarSelect.value;
      var topics = topicsInput.value.split(',').map(function (t) { return t.trim(); }).filter(Boolean);

      statusEl.textContent = '';
      paintLine(statusEl, '');
      resultsArea.hidden = true;
      candidates = [];
      attemptedIds = new Set();
      totalProcessed = 0;
      totalFailed = 0;
      pendingSelection = null;

      setDiscovering(true);
      showRun('discover');
      runStatusEl.textContent = 'Starting…';
      RunProgress.applyStatusClass(runStatusEl, '');
      queriesPanel.hidden = true;
      runProgressEl.replaceChildren();

      fetch('/api/pillars/' + encodeURIComponent(pillar) + '/discover', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ priority_topics: topics, limit: 10 }),
      }).then(function (res) {
        return res.json().then(function (data) { return { res: res, data: data }; });
      }).then(function (out) {
        return attachOrConflict(out.res, out.data, pillar);
      }).catch(function (err) {
        statusEl.textContent = 'Error: ' + err.message;
        paintLine(statusEl, 'is-danger');
        runStatusEl.textContent = '';
        setDiscovering(false);
      });
    });

    processBtn.addEventListener('click', function () {
      var pillar = pillarSelect.value;
      var selectedIds = Array.from(document.querySelectorAll('.paper-checkbox:checked'))
        .map(function (cb) { return cb.dataset.id; });
      if (!selectedIds.length) return;

      // These objects are the paper's permanent record, not a view of it: /select
      // persists exactly what is sent here. Send the candidate's metadata through
      // whole. Anything this list shortens for display is shortened in
      // renderCandidates, never on the way back to the server.
      const selectedPapers = candidates
        .filter(c => selectedIds.includes(c.paper.id))
        .map(c => ({
          id: c.paper.id,
          title: c.paper.title,
          authors: c.paper.authors || [],
          venue: c.paper.venue,
          year: c.paper.year,
          url_pdf: c.paper.url_pdf,
          abstract: c.paper.abstract,
          citation_count: c.paper.citation_count
        }));

      processBtn.disabled = true;
      processBtn.textContent = 'Processing…';
      showRun('process_selected');
      runStatusEl.textContent = 'Starting…';
      RunProgress.applyStatusClass(runStatusEl, '');
      queriesPanel.hidden = true;

      fetch('/api/pillars/' + encodeURIComponent(pillar) + '/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ papers: selectedPapers }),
      }).then(function (res) {
        return res.json().then(function (data) { return { res: res, data: data }; });
      }).then(function (out) {
        if (out.res.status !== 409 && out.res.ok && out.data.run_id) {
          selectedIds.forEach(function (id) { attemptedIds.add(id); });
          pendingSelection = { count: selectedIds.length };
          renderCandidates();
        }
        return attachOrConflict(out.res, out.data, pillar);
      }).catch(function (err) {
        runStatusEl.textContent = 'Error: ' + err.message;
        RunProgress.applyStatusClass(runStatusEl, 'run-danger');
        processBtn.disabled = false;
        processBtn.textContent = 'Process selected';
        renderCandidates();
      });

      selectAllCheckbox.checked = false;
    });

    selectAllCheckbox.addEventListener('change', function () {
      document.querySelectorAll('.paper-checkbox').forEach(function (cb) {
        cb.checked = selectAllCheckbox.checked;
      });
      updateSelectedCount();
    });
    candidatesBody.addEventListener('change', updateSelectedCount);

    cancelBtn.addEventListener('click', function () {
      var runId = cancelBtn.dataset.runId;
      if (!runId) return;
      cancelBtn.disabled = true;
      cancelBtn.textContent = 'Cancelling…';
      RunProgress.cancel(runId).then(function (result) {
        runStatusEl.textContent = result.message;
        RunProgress.applyStatusClass(runStatusEl, 'run-warn');
      }).catch(function (err) {
        runStatusEl.textContent = 'Could not cancel: ' + err.message;
        RunProgress.applyStatusClass(runStatusEl, 'run-danger');
        cancelBtn.disabled = false;
      }).then(function () {
        cancelBtn.textContent = 'Cancel run';
      });
    });

    runForm.addEventListener('submit', function (e) {
      e.preventDefault();
      var pillar = pillarSelect.value;
      var papers = papersInput.value;
      setPipelineBusy(true);
      showRun('run_daily');
      runStatusEl.textContent = 'Starting…';
      RunProgress.applyStatusClass(runStatusEl, '');
      queriesPanel.hidden = true;
      runProgressEl.replaceChildren();

      var url = '/pipeline/run?pillar=' + encodeURIComponent(pillar)
        + '&papers=' + encodeURIComponent(papers);
      fetch(url, { method: 'POST' }).then(function (res) {
        return res.json().then(function (data) { return { res: res, data: data }; });
      }).then(function (out) {
        return attachOrConflict(out.res, out.data, pillar);
      }).catch(function (err) {
        setPipelineBusy(false);
        runStatusEl.textContent = 'Error: ' + err.message;
        RunProgress.applyStatusClass(runStatusEl, 'run-danger');
      });
    });

    document.querySelectorAll('input[name="add-mode"]').forEach(function (input) {
      input.addEventListener('change', function () {
        if (input.checked) setMode(input.value, true);
      });
    });
    if (pillarSelect) pillarSelect.addEventListener('change', rememberPillar);

    setMode(modeFromSearch(global.location.search, initial), false);

    fetch('/api/papers/?limit=500').then(function (res) {
      if (!res.ok) return null;
      return res.json();
    }).then(function (data) {
      if (!data || !Array.isArray(data.papers)) return;
      libraryIds = new Set(data.papers.map(function (paper) { return paper.id; }));
      if (candidates.length) renderCandidates();
    }).catch(function () { /* unknown library: leave the chip off */ });

    RunProgress.reattach(progressEls, onRunFinished, onRunUpdate).then(function (found) {
      if (found && runPanel) runPanel.hidden = false;
    });
  }

  var api = {
    init: init,
    paperPageHref: paperPageHref,
    inLibrary: inLibrary,
    modeFromSearch: modeFromSearch,
    safePdfHref: safePdfHref,
  };
  global.AddPapers = api;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      paperPageHref: paperPageHref,
      inLibrary: inLibrary,
      modeFromSearch: modeFromSearch,
      safePdfHref: safePdfHref,
    };
  }
})(typeof window !== 'undefined' ? window : globalThis);
