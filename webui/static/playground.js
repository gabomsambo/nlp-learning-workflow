/*
 * The Playground tab on the paper page (docs/PLAYGROUND.md).
 *
 * Read-only in this version: a notebook's study steps, each code cell as a
 * card with its one-line explanation, tap-a-line notes, saved output and
 * "try this" diffs. Nothing runs here.
 *
 * Everything is built with createElement + textContent + setAttribute.
 * Notebook HTML outputs arrive already sanitized into a tag tree by the
 * server (nlp_pillars/playground/notebook.py); buildTree() checks the tag
 * allow-list again and sets no attribute but numeric colspan/rowspan, so a
 * hostile notebook cannot reach innerHTML, an event handler or a URL.
 */
(function (global) {
  'use strict';

  var HIDE_KEY = 'nlp:playgroundHideExplanations';
  var ALLOWED_TAGS = [
    'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td', 'caption',
    'b', 'strong', 'i', 'em', 'u', 'code', 'pre', 'br', 'hr', 'p', 'span',
    'div', 'sub', 'sup', 'ul', 'ol', 'li', 'small',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
  ];
  var VOID = ['br', 'hr'];
  var BASE64 = /^[A-Za-z0-9+/]+={0,2}$/;
  var MAX_DIFF_CELLS = 250000;

  /* ------------------------------------------------------------ pure helpers */

  /** Build a sanitized node tree into DOM nodes. Unknown tags keep their text. */
  function buildTree(nodes, doc) {
    var frag = doc.createDocumentFragment();
    (nodes || []).forEach(function (node) { appendNode(frag, node, doc); });
    return frag;
  }

  function appendNode(parent, node, doc) {
    if (typeof node === 'string') {
      parent.appendChild(doc.createTextNode(node));
      return;
    }
    if (!node || typeof node !== 'object') return;
    var tag = String(node.t || '').toLowerCase();
    var children = Array.isArray(node.c) ? node.c : [];
    if (ALLOWED_TAGS.indexOf(tag) === -1) {
      children.forEach(function (child) { appendNode(parent, child, doc); });
      return;
    }
    var el = doc.createElement(tag);
    if ((tag === 'td' || tag === 'th') && node.a) {
      ['colspan', 'rowspan'].forEach(function (name) {
        var value = node.a[name];
        if (typeof value === 'string' && /^[1-9][0-9]{0,3}$/.test(value)) el.setAttribute(name, value);
      });
    }
    if (VOID.indexOf(tag) === -1) {
      children.forEach(function (child) { appendNode(el, child, doc); });
    }
    parent.appendChild(el);
  }

  /** Line diff (LCS). Returns [{op: 'same'|'del'|'add', text}] in order. */
  function lineDiff(before, after) {
    var a = String(before || '').split('\n');
    var b = String(after || '').split('\n');
    var n = a.length;
    var m = b.length;
    if (n * m > MAX_DIFF_CELLS) {
      return a.map(function (t) { return { op: 'del', text: t }; })
        .concat(b.map(function (t) { return { op: 'add', text: t }; }));
    }
    var table = [];
    for (var i = 0; i <= n; i++) table.push(new Array(m + 1).fill(0));
    for (i = n - 1; i >= 0; i--) {
      for (var j = m - 1; j >= 0; j--) {
        table[i][j] = a[i] === b[j] ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1]);
      }
    }
    var out = [];
    i = 0;
    j = 0;
    while (i < n && j < m) {
      if (a[i] === b[j]) { out.push({ op: 'same', text: a[i] }); i++; j++; }
      else if (table[i + 1][j] >= table[i][j + 1]) { out.push({ op: 'del', text: a[i] }); i++; }
      else { out.push({ op: 'add', text: b[j] }); j++; }
    }
    while (i < n) out.push({ op: 'del', text: a[i++] });
    while (j < m) out.push({ op: 'add', text: b[j++] });
    return out;
  }

  /**
   * Which cells each step shows, in notebook order.
   *
   * A step covers every cell after the previous step's last code cell up to
   * its own last code cell, so the markdown that introduces a step travels
   * with it. Cells after the last step (or every cell, when there are no
   * steps) form a final group with step null: shown, but not explained.
   */
  function groupCells(cells, steps) {
    var groups = [];
    var start = 0;
    (steps || []).forEach(function (step) {
      var indices = (step.cells || []).map(function (c) { return c.cell_index; });
      if (!indices.length) return;
      var last = Math.max.apply(null, indices);
      var members = cells.filter(function (c) { return c.index >= start && c.index <= last; });
      groups.push({ step: step, cells: members });
      start = last + 1;
    });
    var rest = cells.filter(function (c) { return c.index >= start; });
    if (rest.length) groups.push({ step: null, cells: rest });
    return groups;
  }

  function explanationsByCell(steps) {
    var map = {};
    (steps || []).forEach(function (step) {
      (step.cells || []).forEach(function (c) { map[c.cell_index] = c; });
    });
    return map;
  }

  function datasetSummary(datasets) {
    var list = datasets || [];
    if (!list.length) return 'No dataset URLs in the code.';
    var ok = list.filter(function (d) { return !d.error; }).length;
    var bytes = list.reduce(function (sum, d) { return sum + (d.bytes || 0); }, 0);
    var size = bytes >= 1e9 ? (bytes / 1e9).toFixed(1) + ' GB' : (bytes / 1e6).toFixed(1) + ' MB';
    if (ok === list.length) return ok + ' dataset' + (ok === 1 ? '' : 's') + ' saved for offline runs (' + size + ').';
    return ok + ' of ' + list.length + ' datasets saved (' + size + '); ' + (list.length - ok) + ' could not be downloaded.';
  }

  function statusText(pg, run) {
    if (run && (run.status === 'pending' || run.status === 'running')) return 'Writing the study guide…';
    if (!pg) return '';
    if (pg.status === 'ready') return pg.steps.length + ' study steps' + (pg.model ? ' · written by ' + pg.model : '');
    if (pg.status === 'partial') return 'Study guide covers only part of the notebook.';
    if (pg.status === 'failed') return 'The study guide could not be written.';
    return 'Notebook imported. No study guide yet.';
  }

  function explainLabel(pg) {
    if (!pg || pg.status === 'imported') return 'Generate study guide';
    if (pg.status === 'ready') return 'Regenerate study guide';
    return 'Retry study guide';
  }

  /**
   * Notebook text, split for display: a short cell is shown whole; a long one
   * shows its first line (a heading, usually) and folds the rest away so the
   * code cards stay in view.
   */
  function markdownParts(source) {
    var text = String(source || '').trim();
    if (!text) return null;
    var lines = text.split('\n');
    if (text.length <= 280 && lines.length <= 5) return { head: text, rest: '' };
    var head = lines[0].replace(/^#+\s*/, '').trim() || 'Notebook text';
    return { head: head, rest: lines.slice(1).join('\n').trim() };
  }

  function safePng(data) {
    return typeof data === 'string' && data.length > 8 && BASE64.test(data) ? data : null;
  }

  /* --------------------------------------------------------------- rendering */

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function renderOutput(out) {
    var box;
    if (out.kind === 'stream' || out.kind === 'text' || out.kind === 'error') {
      box = el('pre', 'pg-out' + (out.kind === 'error' || out.name === 'stderr' ? ' is-error' : ''), out.text || '');
    } else if (out.kind === 'html') {
      box = el('div', 'pg-out pg-out-html');
      box.appendChild(buildTree(out.tree, document));
    } else if (out.kind === 'png' && safePng(out.data)) {
      box = el('div', 'pg-out pg-out-img');
      var img = document.createElement('img');
      img.setAttribute('alt', out.alt || 'Saved figure');
      img.setAttribute('loading', 'lazy');
      img.setAttribute('src', 'data:image/png;base64,' + out.data);
      box.appendChild(img);
    } else {
      return el('p', 'pg-out-note', 'An output of type ' + ((out.mime || []).join(', ') || 'unknown') + ' is not shown.');
    }
    if (out.truncated) {
      var wrap = el('div');
      wrap.appendChild(box);
      wrap.appendChild(el('p', 'pg-out-note', 'Output cut for display. The stored notebook has all of it.'));
      return wrap;
    }
    return box;
  }

  function renderCode(source, notes) {
    var byLine = {};
    (notes || []).forEach(function (n) { byLine[n.line] = n.note; });
    var block = el('div', 'pg-code');
    String(source || '').split('\n').forEach(function (text, i) {
      var number = i + 1;
      var note = byLine[number];
      var row = el(note ? 'button' : 'div', 'pg-line' + (note ? ' has-note' : ''));
      if (note) {
        row.setAttribute('type', 'button');
        row.setAttribute('aria-expanded', 'false');
        row.setAttribute('title', 'Show the note for line ' + number);
      }
      row.appendChild(el('span', 'pg-ln', String(number)));
      row.appendChild(el('code', 'pg-src', text || ' '));
      block.appendChild(row);
      if (note) {
        var noteEl = el('p', 'pg-line-note', note);
        noteEl.hidden = true;
        row.addEventListener('click', function () {
          noteEl.hidden = !noteEl.hidden;
          row.setAttribute('aria-expanded', noteEl.hidden ? 'false' : 'true');
        });
        block.appendChild(noteEl);
      }
    });
    return block;
  }

  function renderTry(source, t) {
    var box = el('details', 'pg-try');
    box.appendChild(el('summary', null, 'Try: ' + t.label));
    box.appendChild(el('p', 'pg-expect', t.expect));
    var diff = el('div', 'pg-diff');
    diff.setAttribute('aria-label', 'Original and changed code, line by line');
    lineDiff(source, t.code).forEach(function (row) {
      var line = el('div', 'pg-diff-line is-' + row.op);
      line.appendChild(el('span', 'pg-diff-mark', row.op === 'add' ? '+' : row.op === 'del' ? '−' : ' '));
      line.appendChild(el('code', null, row.text || ' '));
      if (row.op !== 'same') line.setAttribute('aria-label', (row.op === 'add' ? 'added: ' : 'removed: ') + row.text);
      diff.appendChild(line);
    });
    box.appendChild(diff);
    return box;
  }

  function renderCell(cell, guide, state) {
    if (cell.type !== 'code') {
      var parts = markdownParts(cell.source);
      if (!parts) return null;
      if (!parts.rest) return el('div', 'pg-md', parts.head);
      var fold = el('details', 'pg-md');
      fold.appendChild(el('summary', null, parts.head));
      fold.appendChild(el('div', 'pg-md-body', parts.rest));
      return fold;
    }
    var card = el('article', 'card pg-card');
    card.id = 'pg-cell-' + cell.index;
    var head = el('header', 'pg-card-head');
    var count = cell.execution_count != null ? 'In [' + cell.execution_count + ']' : 'Cell ' + cell.index;
    head.appendChild(el('span', 'pg-count', count));
    card.appendChild(head);

    if (guide) {
      var explain = el('p', 'pg-explain', guide.explanation);
      explain.setAttribute('data-pg-explain', '');
      var reveal = el('button', 'btn btn-ghost btn-sm pg-reveal', 'Show explanation');
      reveal.setAttribute('type', 'button');
      reveal.addEventListener('click', function () {
        explain.hidden = false;
        reveal.hidden = true;
      });
      explain.hidden = state.hide;
      reveal.hidden = !state.hide;
      head.appendChild(reveal);
      card.appendChild(explain);
    } else if (state.hasGuide) {
      card.appendChild(el('p', 'pg-out-note', 'Not explained yet.'));
    }

    card.appendChild(renderCode(cell.source, guide && guide.line_notes));
    if (guide && guide.line_notes && guide.line_notes.length) {
      card.appendChild(el('p', 'pg-hint', 'Tap a highlighted line for its note.'));
    }

    var outputs = cell.outputs || [];
    if (outputs.length) {
      var outBox = el('div', 'pg-outputs');
      outBox.appendChild(el('p', 'pg-label', 'Saved output'));
      outputs.forEach(function (o) { outBox.appendChild(renderOutput(o)); });
      card.appendChild(outBox);
    }

    if (guide) {
      (guide.tries || []).forEach(function (t) { card.appendChild(renderTry(cell.source, t)); });
      if (guide.quiz) {
        var quiz = el('p', 'pg-quiz');
        quiz.appendChild(el('strong', null, 'Can you answer: '));
        quiz.appendChild(document.createTextNode(guide.quiz));
        card.appendChild(quiz);
      }
    }
    return card;
  }

  function renderDatasets(datasets) {
    var box = el('details', 'pg-datasets');
    box.appendChild(el('summary', null, datasetSummary(datasets)));
    var list = el('ul');
    (datasets || []).forEach(function (d) {
      var item = el('li', d.error ? 'is-error' : '');
      item.appendChild(el('code', null, d.url));
      item.appendChild(document.createTextNode(d.error ? ' — ' + d.error : ' — ' + (d.bytes || 0).toLocaleString() + ' bytes'));
      list.appendChild(item);
    });
    if ((datasets || []).length) box.appendChild(list);
    return box;
  }

  function render(root, data, state) {
    var pg = data.playground;
    root.replaceChildren();
    if (!pg) {
      root.appendChild(el('p', 'paper-muted',
        'No notebook yet. Import the paper’s .ipynb below and each code cell becomes a study card.'));
      return;
    }
    var head = el('div', 'pg-head');
    head.appendChild(el('h3', 'pg-title', pg.title || 'Notebook'));
    if (pg.notebook_name) head.appendChild(el('p', 'paper-muted pg-file', pg.notebook_name));
    head.appendChild(renderDatasets(pg.datasets));
    root.appendChild(head);

    if (pg.error && (pg.status === 'partial' || pg.status === 'failed' || pg.status === 'ready')) {
      var callout = el('div', 'pg-error');
      callout.setAttribute('role', 'note');
      callout.appendChild(el('strong', null, pg.status === 'partial' ? 'Some cells were not explained. ' : 'The last attempt failed. '));
      callout.appendChild(document.createTextNode(pg.error));
      root.appendChild(callout);
    }

    var guide = explanationsByCell(pg.steps);
    state.hasGuide = (pg.steps || []).length > 0;
    var groups = groupCells(pg.cells || [], pg.steps || []);
    if ((pg.steps || []).length) {
      var nav = el('ol', 'pg-steps-nav');
      nav.setAttribute('aria-label', 'Study steps');
      pg.steps.forEach(function (step, i) {
        var li = el('li');
        var a = el('a', null, step.title);
        a.setAttribute('href', '#pg-step-' + (i + 1));
        a.addEventListener('click', function (ev) {
          ev.preventDefault();
          var target = document.getElementById('pg-step-' + (i + 1));
          if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
        li.appendChild(a);
        nav.appendChild(li);
      });
      root.appendChild(nav);
    }

    var n = 0;
    groups.forEach(function (group) {
      var section = el('section', 'pg-step');
      if (group.step) {
        n += 1;
        section.id = 'pg-step-' + n;
        section.appendChild(el('p', 'pg-step-kicker', 'Step ' + n));
        section.appendChild(el('h3', 'pg-step-title', group.step.title));
        section.appendChild(el('p', 'pg-goal', group.step.goal));
      } else if ((pg.steps || []).length) {
        section.appendChild(el('h3', 'pg-step-title', 'Rest of the notebook'));
        section.appendChild(el('p', 'pg-goal', 'These cells are not in the study guide.'));
      }
      group.cells.forEach(function (cell) {
        var node = renderCell(cell, guide[cell.index], state);
        if (node) section.appendChild(node);
      });
      root.appendChild(section);
    });
  }

  /* -------------------------------------------------------------------- boot */

  function boot() {
    if (typeof document === 'undefined') return;
    var panel = document.getElementById('playground');
    if (!panel) return;
    var paperId = panel.getAttribute('data-paper-id') || '';
    var root = document.getElementById('playground-root');
    var status = document.getElementById('playground-status');
    var explainBtn = document.getElementById('playground-explain-btn');
    var hideBtn = document.getElementById('playground-hide-explanations');
    var form = document.getElementById('playground-import');
    var fileInput = document.getElementById('playground-file');
    var importBtn = document.getElementById('playground-import-btn');
    var runCard = document.getElementById('playground-run-card');
    var runEls = {
      status: document.getElementById('playground-run-status'),
      stages: document.getElementById('playground-run-stages'),
      progress: document.getElementById('playground-run-progress'),
      cancel: document.getElementById('playground-run-cancel'),
    };
    var state = { hide: false, data: null };
    try { state.hide = global.localStorage.getItem(HIDE_KEY) === 'true'; } catch (e) { /* default */ }

    function say(text, tone) {
      status.textContent = text || '';
      status.className = 'pg-status' + (tone ? ' is-' + tone : '');
    }

    function paintHide() {
      hideBtn.textContent = state.hide ? 'Show explanations' : 'Hide explanations';
      hideBtn.setAttribute('aria-pressed', state.hide ? 'true' : 'false');
    }

    function paint() {
      var data = state.data || { playground: null };
      var pg = data.playground;
      render(root, data, state);
      var running = data.latest_run && (data.latest_run.status === 'pending' || data.latest_run.status === 'running');
      explainBtn.hidden = !pg;
      explainBtn.disabled = !!running;
      explainBtn.textContent = explainLabel(pg);
      hideBtn.hidden = !(pg && (pg.steps || []).length);
      paintHide();
      importBtn.textContent = pg ? 'Replace notebook' : 'Import notebook';
      if (!status.textContent || status.dataset.auto === '1') {
        say(statusText(pg, data.latest_run));
        status.dataset.auto = '1';
      }
    }

    function load() {
      return fetch('/api/playgrounds?paper_id=' + encodeURIComponent(paperId)).then(function (res) {
        return res.json().catch(function () { return {}; }).then(function (body) {
          if (!res.ok) throw new Error((body && body.detail) || 'Could not load the playground (HTTP ' + res.status + ').');
          return body;
        });
      }).then(function (body) {
        state.data = body;
        paint();
        var run = body.latest_run;
        if (run && (run.status === 'pending' || run.status === 'running')) follow(run.id);
      }).catch(function (err) {
        root.replaceChildren(el('p', 'pg-error', err.message));
        explainBtn.hidden = true;
        say('');
      });
    }

    function follow(runId) {
      if (!global.RunProgress) return;
      runCard.hidden = false;
      explainBtn.disabled = true;
      global.RunProgress.attach(runId, runEls, function () {
        explainBtn.disabled = false;
        status.dataset.auto = '1';
        load();
      }, function (run) {
        if (run && run.kind && run.kind !== 'playground_explain') global.RunProgress.detach();
      });
    }

    hideBtn.addEventListener('click', function () {
      state.hide = !state.hide;
      try { global.localStorage.setItem(HIDE_KEY, state.hide ? 'true' : 'false'); } catch (e) { /* this view only */ }
      root.querySelectorAll('[data-pg-explain]').forEach(function (node) { node.hidden = state.hide; });
      root.querySelectorAll('.pg-reveal').forEach(function (node) { node.hidden = !state.hide; });
      paintHide();
    });

    explainBtn.addEventListener('click', function () {
      var pg = state.data && state.data.playground;
      if (!pg || explainBtn.disabled) return;
      explainBtn.disabled = true;
      status.dataset.auto = '0';
      say('Starting…');
      fetch('/api/playgrounds/' + encodeURIComponent(pg.id) + '/explain', { method: 'POST' })
        .then(function (res) {
          return res.json().catch(function () { return {}; }).then(function (body) {
            if ((res.status === 202 || res.status === 409) && body.run_id) {
              say(res.status === 409 ? 'Already writing — following that run.' : 'Writing the study guide…');
              follow(body.run_id);
              return;
            }
            throw new Error((body && (body.detail || body.error)) || 'HTTP ' + res.status);
          });
        })
        .catch(function (err) {
          say(err.message, 'error');
          explainBtn.disabled = false;
        });
    });

    form.addEventListener('submit', function (ev) {
      ev.preventDefault();
      var file = fileInput.files && fileInput.files[0];
      if (!file) { say('Choose a .ipynb file first.', 'error'); return; }
      var body = new FormData();
      body.append('file', file);
      importBtn.disabled = true;
      status.dataset.auto = '0';
      say('Importing and downloading the notebook’s datasets… a large dataset can take a minute.');
      fetch('/api/playgrounds/import?paper_id=' + encodeURIComponent(paperId), { method: 'POST', body: body })
        .then(function (res) {
          return res.json().catch(function () { return {}; }).then(function (data) {
            if (!res.ok) throw new Error((data && data.detail) || 'HTTP ' + res.status);
            return data;
          });
        })
        .then(function (data) {
          say(data.message || 'Notebook imported.', 'ok');
          form.reset();
          return load();
        })
        .catch(function (err) { say(err.message, 'error'); })
        .then(function () { importBtn.disabled = false; });
    });

    if (runEls.cancel) {
      runEls.cancel.addEventListener('click', function () {
        var id = runEls.cancel.dataset.runId;
        if (id && global.RunProgress) global.RunProgress.cancel(id).catch(function () { /* the poll shows the truth */ });
      });
    }

    // The notebook can be a few hundred KB, so it is fetched when the tab is
    // first opened, not on every paper page view.
    var loaded = false;
    function maybeLoad() {
      if (loaded || global.location.hash !== '#playground') return;
      loaded = true;
      load();
    }
    global.addEventListener('hashchange', maybeLoad);
    maybeLoad();
  }

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
    else boot();
  }

  var api = {
    buildTree: buildTree,
    lineDiff: lineDiff,
    groupCells: groupCells,
    explanationsByCell: explanationsByCell,
    datasetSummary: datasetSummary,
    statusText: statusText,
    explainLabel: explainLabel,
    safePng: safePng,
    markdownParts: markdownParts,
    HIDE_KEY: HIDE_KEY,
  };
  global.Playground = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
