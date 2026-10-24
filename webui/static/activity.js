/*
 * Activity page: run history and the status board.
 *
 * This file must not start the shared run poller. That poller rewrites the
 * run query and the stored run id; a history page that did that would steal
 * whatever run another page was following. Stages are fetched once, when a
 * row is opened.
 *
 * Nodes are createElement + textContent. Nothing here concatenates markup.
 */
(function (global) {
  'use strict';

  var LIST_URL = '/api/pipeline-runs?limit=100';
  var RUN_URL = '/api/pipeline-runs/';
  var REFRESH_MS = 30000;
  var PROBE_PILLAR = 'status-probe';

  var KIND_LABEL = {
    run_daily: 'Daily run',
    discover: 'Discovery',
    process_selected: 'Selected papers',
    upload: 'Upload',
    podcast_script: 'Script',
    podcast_audio: 'Audio',
    podcast_factcheck: 'Fact check',
  };

  var STATUS_LABEL = {
    succeeded: 'Succeeded',
    failed: 'Failed',
    running: 'Running',
    pending: 'Waiting',
    cancelled: 'Cancelled',
    interrupted: 'Interrupted',
  };

  function kindLabel(kind) {
    return KIND_LABEL[kind] || 'Run';
  }

  function statusLabel(status) {
    return STATUS_LABEL[status] || (status ? String(status) : 'Unknown');
  }

  function isAdded(run) {
    if (!run) return false;
    if (run.result_added === true || run.result_added === 'true') return true;
    return !!(run.result && run.result.added);
  }

  function titleOf(run) {
    if (!run) return '';
    if (typeof run.result_title === 'string' && run.result_title.trim()) return run.result_title.trim();
    if (run.result && typeof run.result.title === 'string') return run.result.title.trim();
    return '';
  }

  function paperIdOf(run) {
    if (!run) return '';
    if (typeof run.result_paper_id === 'string' && run.result_paper_id) return run.result_paper_id;
    if (run.result && typeof run.result.paper_id === 'string') return run.result.paper_id;
    return '';
  }

  function paperHref(id) {
    if (typeof id !== 'string' || !id) return null;
    return '/papers/' + encodeURIComponent(id);
  }

  function outcomePhrase(run) {
    var kind = kindLabel(run && run.kind);
    var title = titleOf(run);
    var status = run && run.status;
    var quoted = title ? '“' + title + '”' : '';
    if (status === 'pending' || status === 'running') return kind + ' is still running';
    if (status === 'cancelled') return kind + ' was cancelled';
    if (status === 'interrupted') return kind + ' stopped when the server restarted';
    if (run && run.kind === 'upload' && isAdded(run) && status === 'failed') {
      return quoted
        ? 'Added ' + quoted + ', but processing did not finish'
        : 'Added the paper, but processing did not finish';
    }
    if (run && run.kind === 'upload' && status === 'succeeded') {
      return quoted ? 'Added ' + quoted : 'Upload finished';
    }
    if (status === 'failed') return kind + ' did not finish';
    if (status === 'succeeded' && run.papers_failed) {
      var done = run.papers_processed || 0;
      return 'Processed ' + done + ', ' + run.papers_failed + ' failed';
    }
    if (status === 'succeeded' && run.papers_processed) {
      var n = run.papers_processed;
      return 'Processed ' + n + ' paper' + (n === 1 ? '' : 's');
    }
    if (status === 'succeeded' && run.kind === 'run_daily') return 'Finished with nothing new to process';
    if (status === 'succeeded' && quoted) return kind + ' finished — ' + quoted;
    if (status === 'succeeded') return kind + ' finished';
    return kind + ' updated';
  }

  function toneOf(run) {
    var status = run && run.status;
    if (status === 'failed' && isAdded(run)) return 'warn';
    if (status === 'failed') return 'danger';
    if (status === 'cancelled' || status === 'interrupted') return 'warn';
    if (status === 'succeeded' && run.papers_failed) return 'warn';
    if (status === 'succeeded') return 'ok';
    if (status === 'running' || status === 'pending') return 'accent';
    return 'neutral';
  }

  function formatDuration(start, end) {
    if (!start || !end) return '';
    var ms = new Date(end).getTime() - new Date(start).getTime();
    if (!isFinite(ms) || ms < 0) return '';
    var seconds = Math.round(ms / 1000);
    if (seconds < 60) return seconds + 's';
    var minutes = Math.floor(seconds / 60);
    seconds = seconds % 60;
    if (minutes < 60) return seconds ? minutes + 'm ' + seconds + 's' : minutes + 'm';
    var hours = Math.floor(minutes / 60);
    minutes = minutes % 60;
    return minutes ? hours + 'h ' + minutes + 'm' : hours + 'h';
  }

  function runDuration(run, now) {
    if (!run) return '';
    var start = run.started_at || run.created_at;
    var end = run.finished_at;
    if (!end && (run.status === 'running' || run.status === 'pending')) {
      end = (now || new Date()).toISOString();
    }
    return formatDuration(start, end);
  }

  function humaniseTime(iso, now) {
    var then = new Date(iso);
    var clock = now || new Date();
    if (!iso || isNaN(then.getTime())) return '';
    var seconds = Math.round((clock.getTime() - then.getTime()) / 1000);
    var future = seconds < 0;
    var abs = Math.abs(seconds);
    if (abs < 45) return 'just now';
    if (abs < 90) return future ? 'in a minute' : 'a minute ago';
    var minutes = Math.round(abs / 60);
    if (minutes < 60) {
      return future ? 'in ' + minutes + ' minutes' : minutes + ' minutes ago';
    }
    var hours = Math.round(abs / 3600);
    if (hours < 24) {
      var hourWord = hours === 1 ? 'hour' : 'hours';
      return future ? 'in ' + hours + ' ' + hourWord : hours + ' ' + hourWord + ' ago';
    }
    var days = Math.round(abs / 86400);
    if (days < 7) {
      var dayWord = days === 1 ? 'day' : 'days';
      return future ? 'in ' + days + ' ' + dayWord : days + ' ' + dayWord + ' ago';
    }
    try {
      return then.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    } catch (e) {
      return then.toDateString();
    }
  }

  function matchesFilters(run, kind, pillar) {
    if (!run || run.pillar_id === PROBE_PILLAR) return false;
    if (kind && kind !== 'all' && run.kind !== kind) return false;
    if (pillar && run.pillar_id !== pillar) return false;
    return true;
  }

  function readPillars() {
    var node = document.getElementById('activity-pillars');
    if (!node) return [];
    try {
      var parsed = JSON.parse(node.textContent || '[]');
      return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
      return [];
    }
  }

  function pillarName(id, pillars) {
    for (var i = 0; i < pillars.length; i++) {
      if (pillars[i] && pillars[i].id === id) return pillars[i].name || id;
    }
    return id || '';
  }

  function chip(label, variant) {
    var el = document.createElement('span');
    el.className = 'chip' + (variant && variant !== 'neutral' ? ' chip-' + variant : '');
    el.textContent = label;
    return el;
  }

  function renderStages(container, run) {
    var list = document.createElement('ul');
    (run.stages || []).forEach(function (stage) {
      var status = global.RunProgress && global.RunProgress.displayStatus
        ? global.RunProgress.displayStatus(stage, run)
        : (stage.status || 'pending');
      var li = document.createElement('li');
      li.className = 'run-stage run-stage-' + status;
      var marker = document.createElement('span');
      marker.className = 'run-stage-icon';
      marker.setAttribute('aria-hidden', 'true');
      marker.textContent = status === 'failed' ? '●' : '○';
      li.appendChild(marker);
      var body = document.createElement('span');
      body.className = 'run-stage-body';
      var name = document.createElement('span');
      name.className = 'run-stage-name';
      name.textContent = global.RunProgress && global.RunProgress.stageLabel
        ? global.RunProgress.stageLabel(stage.name)
        : (stage.name || 'Step');
      body.appendChild(name);
      var state = document.createElement('span');
      state.className = 'run-stage-detail';
      state.textContent = status;
      body.appendChild(state);
      if (stage.detail) {
        var detail = document.createElement('span');
        detail.className = 'run-stage-detail';
        detail.textContent = stage.detail;
        body.appendChild(detail);
      }
      li.appendChild(body);
      var dur = global.RunProgress && global.RunProgress.duration
        ? global.RunProgress.duration(stage)
        : '';
      if (dur) {
        var time = document.createElement('span');
        time.className = 'run-stage-duration';
        time.textContent = dur;
        li.appendChild(time);
      }
      list.appendChild(li);
    });
    container.replaceChildren(list);
  }

  function detailFacts(run) {
    var result = run && run.result;
    if (!result || typeof result !== 'object') return [];
    var facts = [];
    if (Array.isArray(result.candidates)) {
      facts.push(result.candidates.length + ' candidate papers');
    }
    if (result.word_count != null) facts.push(result.word_count + ' words');
    if (result.saved === false) facts.push('Script was not saved');
    return facts;
  }

  function renderRun(run, pillars, now) {
    var article = document.createElement('article');
    article.className = 'run-entry';

    var top = document.createElement('div');
    top.className = 'run-entry-top';

    var main = document.createElement('div');
    main.className = 'run-entry-main';

    var chips = document.createElement('div');
    chips.className = 'chip-row';
    chips.appendChild(chip(kindLabel(run.kind), 'neutral'));
    chips.appendChild(chip(statusLabel(run.status), toneOf(run)));
    main.appendChild(chips);

    var summary = document.createElement('p');
    summary.className = 'run-summary';
    summary.textContent = outcomePhrase(run);
    main.appendChild(summary);

    var metaBits = [];
    var pillar = pillarName(run.pillar_id, pillars);
    if (pillar) metaBits.push(pillar);
    var when = humaniseTime(run.created_at, now);
    if (when) metaBits.push(when);
    var dur = runDuration(run, now);
    if (dur) metaBits.push(dur);
    var meta = document.createElement('p');
    meta.className = 'run-meta';
    meta.textContent = metaBits.join(' · ');
    main.appendChild(meta);

    var paper = paperHref(paperIdOf(run));
    if (paper) {
      var link = document.createElement('a');
      link.className = 'run-paper';
      link.href = paper;
      link.textContent = 'Open paper';
      main.appendChild(link);
    }
    top.appendChild(main);
    article.appendChild(top);

    if (run.error) {
      var details = document.createElement('details');
      details.className = 'run-error';
      var summaryEl = document.createElement('summary');
      summaryEl.textContent = 'Technical detail';
      var pre = document.createElement('pre');
      pre.textContent = run.error;
      details.appendChild(summaryEl);
      details.appendChild(pre);
      article.appendChild(details);
    }

    var steps = document.createElement('div');
    steps.className = 'run-stage-list';
    steps.hidden = true;
    var toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'btn btn-ghost btn-sm';
    toggle.textContent = 'Show steps';
    toggle.setAttribute('aria-expanded', 'false');
    toggle.addEventListener('click', function () {
      var open = toggle.getAttribute('aria-expanded') === 'true';
      if (open) {
        steps.hidden = true;
        steps.classList.remove('is-open');
        toggle.textContent = 'Show steps';
        toggle.setAttribute('aria-expanded', 'false');
        return;
      }
      toggle.disabled = true;
      toggle.textContent = 'Loading steps…';
      fetch(RUN_URL + encodeURIComponent(run.id), { headers: { Accept: 'application/json' } })
        .then(function (res) {
          if (!res.ok) throw new Error('HTTP ' + res.status);
          return res.json();
        })
        .then(function (full) {
          renderStages(steps, full);
          var facts = detailFacts(full);
          if (facts.length) {
            var note = document.createElement('p');
            note.className = 'run-meta';
            note.textContent = facts.join(' · ');
            steps.insertBefore(note, steps.firstChild);
          }
          steps.hidden = false;
          steps.classList.add('is-open');
          toggle.textContent = 'Hide steps';
          toggle.setAttribute('aria-expanded', 'true');
        })
        .catch(function (err) {
          steps.replaceChildren();
          var note = document.createElement('p');
          note.className = 'field-error';
          note.textContent = 'Could not load the steps for this run. ' + (err && err.message ? err.message : '');
          steps.appendChild(note);
          steps.hidden = false;
          steps.classList.add('is-open');
          toggle.textContent = 'Hide steps';
          toggle.setAttribute('aria-expanded', 'true');
        })
        .finally(function () {
          toggle.disabled = false;
        });
    });
    article.appendChild(toggle);
    article.appendChild(steps);
    return article;
  }

  function renderList(runs, kind, pillar, pillars, now) {
    var host = document.getElementById('activity-list');
    if (!host) return;
    var matched = (runs || []).filter(function (run) {
      return matchesFilters(run, kind, pillar);
    });
    if (!matched.length) {
      var empty = document.createElement('p');
      empty.className = 'activity-meta';
      empty.textContent = runs && runs.length
        ? 'No runs match these filters.'
        : 'No runs recorded yet.';
      host.replaceChildren(empty);
      return;
    }
    var frag = document.createDocumentFragment();
    matched.forEach(function (run) {
      frag.appendChild(renderRun(run, pillars, now));
    });
    host.replaceChildren(frag);
  }

  function renderStatus(payload) {
    var board = document.getElementById('status-board');
    if (!board) return;
    var checks = payload && Array.isArray(payload.checks) ? payload.checks : [];
    if (!checks.length) {
      var empty = document.createElement('p');
      empty.className = 'activity-meta';
      empty.textContent = 'Status could not be loaded.';
      board.replaceChildren(empty);
      return;
    }
    var frag = document.createDocumentFragment();
    checks.forEach(function (check) {
      var row = document.createElement('div');
      row.className = 'status-row';
      var text = document.createElement('div');
      var name = document.createElement('p');
      name.className = 'status-name';
      name.textContent = check.label || check.id || 'Check';
      var summary = document.createElement('p');
      summary.className = 'status-summary';
      summary.textContent = check.summary || '';
      text.appendChild(name);
      text.appendChild(summary);
      if (check.fix) {
        var fix = document.createElement('p');
        fix.className = 'status-fix';
        fix.textContent = check.fix;
        text.appendChild(fix);
      }
      var mark = chip(check.state === 'ok' ? 'OK' : 'DOWN', check.state === 'ok' ? 'ok' : 'danger');
      row.appendChild(text);
      row.appendChild(mark);
      frag.appendChild(row);
    });
    board.replaceChildren(frag);
  }

  function currentFilters() {
    var pressed = document.querySelector('#kind-filters [aria-pressed="true"]');
    var select = document.getElementById('activity-pillar');
    return {
      kind: pressed ? pressed.getAttribute('data-kind') || 'all' : 'all',
      pillar: select ? select.value : '',
    };
  }

  function writeFilters(kind, pillar) {
    try {
      var params = new URLSearchParams(global.location.search);
      if (kind && kind !== 'all') params.set('kind', kind);
      else params.delete('kind');
      if (pillar) params.set('pillar', pillar);
      else params.delete('pillar');
      var query = params.toString();
      var next = global.location.pathname + (query ? '?' + query : '') + global.location.hash;
      global.history.replaceState(null, '', next);
    } catch (e) { /* the list still filters in memory */ }
  }

  function applyQuery() {
    var params;
    try { params = new URLSearchParams(global.location.search); }
    catch (e) { return; }
    var kind = params.get('kind') || 'all';
    var pillar = params.get('pillar') || '';
    document.querySelectorAll('#kind-filters [data-kind]').forEach(function (button) {
      button.setAttribute('aria-pressed', button.getAttribute('data-kind') === kind ? 'true' : 'false');
    });
    var select = document.getElementById('activity-pillar');
    if (select && pillar) select.value = pillar;
  }

  var runs = [];
  var pillars = [];
  var refreshTimer = null;

  function listIsOpen() {
    return !!document.querySelector('#activity-list details[open], #activity-list .is-open');
  }

  function scheduleRefresh() {
    if (refreshTimer) clearTimeout(refreshTimer);
    var live = runs.some(function (run) {
      return run && (run.status === 'running' || run.status === 'pending');
    });
    if (!live) return;
    refreshTimer = setTimeout(function () {
      refreshTimer = null;
      if (listIsOpen()) {
        scheduleRefresh();
        return;
      }
      loadRuns();
    }, REFRESH_MS);
  }

  function showError(message) {
    var node = document.getElementById('activity-error');
    if (!node) return;
    if (!message) {
      node.hidden = true;
      node.textContent = '';
      return;
    }
    node.hidden = false;
    node.textContent = message;
  }

  function paint() {
    var filters = currentFilters();
    renderList(runs, filters.kind, filters.pillar, pillars, new Date());
  }

  function loadRuns() {
    return fetch(LIST_URL, { headers: { Accept: 'application/json' } })
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(function (body) {
        showError('');
        runs = Array.isArray(body) ? body : [];
        paint();
        scheduleRefresh();
      })
      .catch(function (err) {
        var host = document.getElementById('activity-list');
        if (host) host.replaceChildren();
        showError('Could not load run history: ' + (err && err.message ? err.message : 'unknown error'));
      });
  }

  function loadStatus(force) {
    var loader = global.NlpStatus && global.NlpStatus.load;
    if (!loader) {
      renderStatus(null);
      return Promise.resolve();
    }
    return loader({ force: !!force }).then(renderStatus).catch(function () {
      renderStatus(null);
    });
  }

  function boot() {
    pillars = readPillars();
    applyQuery();
    document.querySelectorAll('#kind-filters [data-kind]').forEach(function (button) {
      button.addEventListener('click', function () {
        document.querySelectorAll('#kind-filters [data-kind]').forEach(function (other) {
          other.setAttribute('aria-pressed', other === button ? 'true' : 'false');
        });
        var filters = currentFilters();
        writeFilters(filters.kind, filters.pillar);
        paint();
      });
    });
    var select = document.getElementById('activity-pillar');
    if (select) {
      select.addEventListener('change', function () {
        var filters = currentFilters();
        writeFilters(filters.kind, filters.pillar);
        paint();
      });
    }
    var reload = document.getElementById('activity-reload');
    if (reload) reload.addEventListener('click', function () { loadRuns(); });
    var refresh = document.getElementById('status-refresh');
    if (refresh) refresh.addEventListener('click', function () { loadStatus(true); });
    loadRuns();
    loadStatus(false);
  }

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
    else boot();
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      KIND_LABEL: KIND_LABEL,
      kindLabel: kindLabel,
      statusLabel: statusLabel,
      outcomePhrase: outcomePhrase,
      toneOf: toneOf,
      formatDuration: formatDuration,
      runDuration: runDuration,
      humaniseTime: humaniseTime,
      paperHref: paperHref,
      matchesFilters: matchesFilters,
      titleOf: titleOf,
    };
  }
})(typeof window !== 'undefined' ? window : globalThis);
