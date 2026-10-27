/*
 * The Playground tab on the paper page (docs/PLAYGROUND.md).
 *
 * A notebook's study steps, each code cell as a card with its one-line
 * explanation, tap-a-line notes and saved output. Cells run in the
 * playground runner (one session per open playground, playground-session.js):
 * Run, Run up to here, edit with Reset to original, and "try this" tweaks
 * shown as a Before / After. The pure logic is in playground-core.js.
 *
 * Live outputs arrive from the execute proxy in the same view shape as saved
 * ones (notebook.py::runner_output_view), so one renderer serves both.
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

  var Core = global.PlaygroundCore ||
    (typeof require === 'function' ? require('./playground-core.js') : {});
  var SessionApi = global.PlaygroundSession ||
    (typeof require === 'function' ? require('./playground-session.js') : {});
  var lineDiff = Core.lineDiff;

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

  /**
   * The cell a link asked for (`?cell=12#playground`, from the Listen page's
   * code card), or null.
   */
  function requestedCell(search) {
    var raw = new URLSearchParams(search || '').get('cell');
    if (raw == null || !/^[0-9]{1,5}$/.test(raw)) return null;
    return Number(raw);
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

  function renderOutput(out, opts) {
    var box;
    if (out.kind === 'stream' || out.kind === 'text' || out.kind === 'error') {
      box = el('pre', 'pg-out' + (out.kind === 'error' || out.name === 'stderr' ? ' is-error' : ''), out.text || '');
    } else if (out.kind === 'html') {
      box = el('div', 'pg-out pg-out-html');
      box.appendChild(buildTree(out.tree, document));
    } else if (out.kind === 'png' && safePng(out.data)) {
      box = el('div', 'pg-out pg-out-img');
      var img = document.createElement('img');
      img.setAttribute('alt', out.alt || 'Figure');
      img.setAttribute('loading', 'lazy');
      img.setAttribute('src', 'data:image/png;base64,' + out.data);
      box.appendChild(img);
    } else {
      return el('p', 'pg-out-note', 'An output of type ' + ((out.mime || []).join(', ') || 'unknown') + ' is not shown.');
    }
    if (out.truncated) {
      var wrap = el('div');
      wrap.appendChild(box);
      wrap.appendChild(el('p', 'pg-out-note', opts && opts.live
        ? 'Output cut at the runner’s size limit.'
        : 'Output cut for display. The stored notebook has all of it.'));
      return wrap;
    }
    return box;
  }

  function renderOutputs(box, outputs, opts) {
    box.replaceChildren();
    if (!outputs || !outputs.length) {
      box.appendChild(el('p', 'pg-out-note', 'No output.'));
      return;
    }
    outputs.forEach(function (o) { box.appendChild(renderOutput(o, opts)); });
  }

  /** One output with its changed lines or table cells marked (Before / After). */
  function renderMarked(out, lineFlags, tableFlags) {
    if (lineFlags && (out.kind === 'stream' || out.kind === 'text' || out.kind === 'error')) {
      var pre = el('pre', 'pg-out' + (out.kind === 'error' || out.name === 'stderr' ? ' is-error' : ''));
      String(out.text || '').split('\n').forEach(function (line, i, all) {
        var span = el('span', 'pg-hl-line' + (lineFlags[i] ? ' is-changed' : ''), line + (i < all.length - 1 ? '\n' : ''));
        pre.appendChild(span);
      });
      return pre;
    }
    var node = renderOutput(out, { live: true });
    if (tableFlags && node.querySelectorAll) {
      var cells = node.querySelectorAll('td, th');
      tableFlags.forEach(function (changed, i) {
        if (changed && cells[i]) cells[i].classList.add('is-changed');
      });
    }
    return node;
  }

  function renderCompare(before, after, labels) {
    var grid = el('div', 'pg-ba');
    var colB = el('div', 'pg-ba-col');
    var colA = el('div', 'pg-ba-col');
    colB.appendChild(el('p', 'pg-label', 'Before · ' + labels.before));
    colA.appendChild(el('p', 'pg-label', 'After · ' + labels.after));
    var pairs = Core.compareOutputs(before, after);
    var marked = false;
    pairs.forEach(function (p) {
      if (p.text) marked = marked || p.text.after.some(Boolean) || p.text.before.some(Boolean);
      if (p.table) marked = marked || p.table.some(Boolean);
      if (p.before) colB.appendChild(renderMarked(p.before, p.text && p.text.before, p.table));
      if (p.after) colA.appendChild(renderMarked(p.after, p.text && p.text.after, p.table));
    });
    if (!(before || []).length) colB.appendChild(el('p', 'pg-out-note', 'No output.'));
    if (!(after || []).length) colA.appendChild(el('p', 'pg-out-note', 'No output.'));
    grid.appendChild(colB);
    grid.appendChild(colA);
    var wrap = el('div', 'pg-compare');
    var note = Core.sameOutputs(before, after)
      ? 'The output did not change.'
      : marked ? 'Changed lines and table cells are highlighted.' : '';
    if (note) wrap.appendChild(el('p', 'pg-hint', note));
    wrap.appendChild(grid);
    return wrap;
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

  function renderDiff(rows, label) {
    var diff = el('div', 'pg-diff');
    diff.setAttribute('aria-label', label || 'Original and changed code, line by line');
    rows.forEach(function (row) {
      if (row.op === 'gap') {
        diff.appendChild(el('div', 'pg-diff-line is-gap', '⋯ ' + row.count + ' unchanged line' + (row.count === 1 ? '' : 's')));
        return;
      }
      var line = el('div', 'pg-diff-line is-' + row.op);
      line.appendChild(el('span', 'pg-diff-mark', row.op === 'add' ? '+' : row.op === 'del' ? '−' : ' '));
      line.appendChild(el('code', null, row.text || ' '));
      if (row.op !== 'same') line.setAttribute('aria-label', (row.op === 'add' ? 'added: ' : 'removed: ') + row.text);
      diff.appendChild(line);
    });
    return diff;
  }

  function button(text, className) {
    var b = el('button', 'btn btn-sm ' + (className || 'btn-secondary'), text);
    b.setAttribute('type', 'button');
    return b;
  }

  /**
   * The editor: a textarea over a line-number gutter. No library. Tab
   * indents (Shift+Tab outdents); Escape then Tab leaves, so the keyboard is
   * never trapped. Enter keeps the indent. Shift+Enter is the card's.
   */
  function buildEditor(value, onInput) {
    var wrap = el('div', 'pg-editor');
    var gutter = el('pre', 'pg-gutter');
    gutter.setAttribute('aria-hidden', 'true');
    var ta = el('textarea', 'pg-ta');
    ta.value = value;
    ['spellcheck', 'autocapitalize', 'autocomplete', 'autocorrect'].forEach(function (name) {
      ta.setAttribute(name, name === 'spellcheck' ? 'false' : 'off');
    });
    ta.setAttribute('wrap', 'off');
    ta.setAttribute('aria-label', 'Code for this cell (editable)');
    var escaped = false;
    function sync() {
      gutter.textContent = Core.lineNumbers(ta.value);
      var lines = ta.value.split('\n').length;
      ta.rows = Math.max(3, Math.min(lines + 1, 30));
      gutter.scrollTop = ta.scrollTop;
    }
    function changed() { sync(); onInput(ta.value); }
    ta.addEventListener('input', changed);
    ta.addEventListener('scroll', function () { gutter.scrollTop = ta.scrollTop; });
    ta.addEventListener('focus', function () { escaped = false; });
    ta.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape') { escaped = true; return; }
      if (ev.key === 'Tab' && !ev.ctrlKey && !ev.metaKey && !ev.altKey && !escaped) {
        ev.preventDefault();
        var r = Core.indentLines(ta.value, ta.selectionStart, ta.selectionEnd, ev.shiftKey);
        ta.value = r.value;
        ta.setSelectionRange(r.start, r.end);
        changed();
        return;
      }
      escaped = false;
      if (ev.key === 'Enter' && !ev.shiftKey && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
        ev.preventDefault();
        ta.setRangeText(Core.newlineIndent(ta.value, ta.selectionStart), ta.selectionStart, ta.selectionEnd, 'end');
        changed();
      }
    });
    wrap.appendChild(gutter);
    wrap.appendChild(ta);
    sync();
    return { wrap: wrap, ta: ta };
  }

  /* ----------------------------------------------------------- a code cell */

  /**
   * One runnable cell. Holds the cell's live state (edited code, latest live
   * output) and builds its controls. `live` is the tab's shared context:
   * {session, pg, cells, codeIndices}.
   */
  function CellCtl(cell, guide, live) {
    this.cell = cell;
    this.index = cell.index;
    this.guide = guide || null;
    this.live = live;
    this.original = cell.source || '';
    this.edited = null;
    this.liveOutputs = null;
    this.liveCode = null;
    this.timer = null;
    this.tries = [];
  }

  CellCtl.prototype.code = function () {
    return this.edited != null ? this.edited : this.original;
  };

  /** What "Before" is for a try: the latest live output, else the saved one. */
  CellCtl.prototype.beforeState = function () {
    if (this.liveOutputs) return { outputs: this.liveOutputs, code: this.liveCode, label: 'your last run' };
    return { outputs: this.cell.outputs || [], code: this.original, label: 'saved output' };
  };

  CellCtl.prototype.build = function (card) {
    var self = this;
    this.card = card;
    card.setAttribute('data-cell', String(this.index));
    card.setAttribute('tabindex', '-1');

    this.toolbar = el('div', 'pg-toolbar');
    this.runBtn = button('Run', 'btn-primary');
    this.runBtn.setAttribute('title', 'Run this cell (Shift+Enter)');
    this.upBtn = button('Run up to here');
    this.upBtn.setAttribute('title', 'Run every earlier cell this session has not run yet, then this one');
    this.editBtn = button('Edit', 'btn-ghost');
    this.resetBtn = button('Reset to original', 'btn-ghost');
    this.resetBtn.hidden = true;
    this.runBtn.addEventListener('click', function () { self.run(); });
    this.upBtn.addEventListener('click', function () { self.runUpTo(); });
    this.editBtn.addEventListener('click', function () { self.startEdit(true); });
    this.resetBtn.addEventListener('click', function () { self.reset(); });
    [this.runBtn, this.upBtn, this.editBtn, this.resetBtn].forEach(function (b) { self.toolbar.appendChild(b); });
    if (this.live.codeIndices[0] === this.index) this.upBtn.hidden = true;

    this.status = el('div', 'pg-run-status');
    this.statusText = el('span', 'pg-run-text');
    this.statusText.setAttribute('role', 'status');
    this.statusText.setAttribute('aria-live', 'polite');
    this.interruptBtn = button('Interrupt', 'btn-ghost');
    this.interruptBtn.hidden = true;
    this.interruptBtn.addEventListener('click', function () { self.live.session.interrupt(); });
    this.longerBtn = button('Run again with a longer limit', 'btn-secondary');
    this.longerBtn.hidden = true;
    this.status.appendChild(this.statusText);
    this.status.appendChild(this.interruptBtn);
    this.status.appendChild(this.longerBtn);
    this.status.hidden = true;

    this.editInfo = el('details', 'pg-edit-diff');
    this.editInfo.hidden = true;
    this.editSummary = el('summary');
    this.editInfo.appendChild(this.editSummary);
  };

  CellCtl.prototype.say = function (text, tone) {
    this.status.hidden = !text;
    this.statusText.textContent = text || '';
    this.status.className = 'pg-run-status' + (tone ? ' is-' + tone : '');
  };

  CellCtl.prototype.startTimer = function (label, target) {
    var started = Date.now();
    var self = this;
    var write = target || function (t) { self.say(t, 'busy'); };
    this.stopTimer();
    write(label + ' ' + Core.formatElapsed(0));
    this.timer = setInterval(function () { write(label + ' ' + Core.formatElapsed(Date.now() - started)); }, 200);
  };

  CellCtl.prototype.stopTimer = function () {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  };

  CellCtl.prototype.setBusy = function (busy) {
    this.runBtn.disabled = busy;
    this.upBtn.disabled = busy;
    this.tries.forEach(function (t) { t.runBtn.disabled = busy; });
  };

  /**
   * Run `code` in the session with a running timer and an Interrupt button:
   * the cell's own, or a try's (`ui` = {say, interrupt}).
   * Resolves {result, generation} | {error} | {cancelled}.
   */
  CellCtl.prototype.execute = function (code, timeoutS, ui) {
    var self = this;
    var session = this.live.session;
    var say = (ui && ui.say) || function (t, tone) { self.say(t, tone); };
    var stopBtn = (ui && ui.interrupt) || this.interruptBtn;
    say('Waiting for the session…', 'busy');
    this.setBusy(true);
    stopBtn.hidden = false;
    this.longerBtn.hidden = true;
    return session.execute(code, timeoutS, function () {
      self.startTimer('Running…', function (t) { say(t, 'busy'); });
    }).then(function (res) {
      self.stopTimer();
      self.setBusy(false);
      stopBtn.hidden = true;
      return res;
    });
  };

  CellCtl.prototype.showLive = function (outputs, result, code) {
    this.liveOutputs = outputs || [];
    this.liveCode = code;
    this.outLabel.textContent = 'Live output';
    renderOutputs(this.liveBox, this.liveOutputs, { live: true });
    this.liveBox.hidden = false;
    if ((this.cell.outputs || []).length) {
      this.savedHead.hidden = false;
      this.savedToggle.hidden = false;
      this.savedBox.hidden = true;
      this.savedToggle.textContent = 'Compare with saved output';
      this.savedToggle.setAttribute('aria-pressed', 'false');
    } else {
      this.savedBox.hidden = true;
    }
    if (result && result.execution_count != null) this.countEl.textContent = 'In [' + result.execution_count + ']';
  };

  /** Run this cell. Resolves true when it ran without error. */
  CellCtl.prototype.run = function (timeoutS) {
    var self = this;
    var session = this.live.session;
    var limit = timeoutS || session.timeoutS();
    var code = this.code();
    return this.execute(code, timeoutS).then(function (res) {
      if (res.cancelled) { self.say('Not run: stopped before its turn.', 'warn'); return false; }
      if (res.error) { self.say(res.error, 'error'); return false; }
      var r = res.result;
      self.showLive(r.outputs, r, code);
      var summary = Core.runSummary(r, limit);
      self.say(summary.text, summary.tone);
      if (r.status === 'timeout') self.offerLonger(limit);
      if (r.status === 'ok') session.markRan(self.index, res.generation);
      self.live.paintRan();
      return r.status === 'ok';
    });
  };

  CellCtl.prototype.offerLonger = function (limit) {
    var self = this;
    var next = Core.longerTimeout(limit, this.live.session.maxTimeoutS());
    if (!next) {
      this.say(this.statusText.textContent + ' That is the runner’s maximum.', 'warn');
      return;
    }
    this.longerBtn.textContent = 'Run again with a ' + Core.seconds(next) + ' limit';
    this.longerBtn.hidden = false;
    this.longerBtn.onclick = function () { self.run(next); };
  };

  /** Run every earlier cell this session has not run, then this one. */
  CellCtl.prototype.runUpTo = function () {
    var self = this;
    var live = this.live;
    var plan = Core.planRunUpTo(live.codeIndices, this.index, live.session.ran);
    var earlier = plan.slice(0, -1);
    var i = 0;
    function next() {
      if (i >= earlier.length) return self.run();
      var ctl = live.cells[earlier[i]];
      i += 1;
      self.say('Running earlier cells: ' + i + ' of ' + earlier.length + '…', 'busy');
      return ctl.run().then(function (ok) {
        if (ok) return next();
        self.say('Stopped: the cell ' + (ctl.countEl.textContent || '') + ' above did not finish cleanly. Fix it, or run from there.', 'error');
        if (ctl.card.scrollIntoView) ctl.card.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return false;
      });
    }
    return next();
  };

  CellCtl.prototype.startEdit = function (focus) {
    var self = this;
    if (!this.editor) {
      this.editor = buildEditor(this.code(), function (value) {
        self.edited = value;
        self.paintEdit();
      });
      this.editorHint = el('p', 'pg-hint', 'Tab indents · Esc then Tab leaves the editor · Shift+Enter runs.');
      this.codeView.after(this.editor.wrap);
      this.editor.wrap.after(this.editorHint);
    }
    if (this.edited == null) this.edited = this.original;
    this.codeView.hidden = true;
    if (this.noteHint) this.noteHint.hidden = true;
    this.editor.wrap.hidden = false;
    this.editorHint.hidden = false;
    this.editBtn.hidden = true;
    this.resetBtn.hidden = false;
    this.paintEdit();
    if (focus) this.editor.ta.focus();
  };

  CellCtl.prototype.paintEdit = function () {
    var dirty = this.edited != null && this.edited !== this.original;
    this.editInfo.hidden = !dirty;
    this.card.classList.toggle('is-edited', dirty);
    if (!dirty) return;
    this.editSummary.textContent = Core.editSummary(this.original, this.edited);
    var old = this.editInfo.querySelector('.pg-diff');
    var rows = Core.compactDiff(lineDiff(this.original, this.edited), 1);
    var diff = renderDiff(rows, 'Your edit against the original code');
    if (old) old.replaceWith(diff); else this.editInfo.appendChild(diff);
  };

  CellCtl.prototype.reset = function () {
    this.edited = null;
    if (this.editor) {
      this.editor.wrap.remove();
      this.editorHint.remove();
      this.editor = null;
    }
    this.codeView.hidden = false;
    if (this.noteHint) this.noteHint.hidden = false;
    this.editBtn.hidden = false;
    this.resetBtn.hidden = true;
    this.paintEdit();
    this.editBtn.focus();
  };

  CellCtl.prototype.paintRan = function () {
    var ran = this.live.session.hasRan(this.index);
    this.ranEl.hidden = !ran;
    var generation = this.live.session.generation;
    this.tries.forEach(function (t) {
      if (t.notice && t.generation !== generation && !t.notice.hidden) {
        t.notice.hidden = true;
        t.undoneNote.hidden = false;
      }
    });
  };

  /* ------------------------------------------------------------ "try this" */

  function renderTry(source, t, ctl) {
    var box = el('details', 'pg-try');
    box.appendChild(el('summary', null, 'Try: ' + t.label));
    box.appendChild(renderDiff(lineDiff(source, t.code)));
    box.appendChild(el('p', 'pg-expect', t.expect));
    if (!ctl) return box;

    var entry = { runBtn: button('Run this try', 'btn-primary'), box: box, generation: null };
    var bar = el('div', 'pg-toolbar');
    var status = el('span', 'pg-run-text');
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');
    var stop = button('Interrupt', 'btn-ghost');
    stop.hidden = true;
    stop.addEventListener('click', function () { ctl.live.session.interrupt(); });
    bar.appendChild(entry.runBtn);
    bar.appendChild(status);
    bar.appendChild(stop);
    box.appendChild(bar);
    var result = el('div', 'pg-try-result');
    result.hidden = true;
    box.appendChild(result);
    ctl.tries.push(entry);

    function say(text, tone) {
      status.textContent = text || '';
      status.className = 'pg-run-text' + (tone ? ' is-' + tone : '');
    }

    entry.run = function () {
      var before = ctl.beforeState();
      var limit = ctl.live.session.timeoutS();
      return ctl.execute(t.code, null, { say: say, interrupt: stop }).then(function (res) {
        if (res.cancelled) { say('Not run: stopped before its turn.', 'warn'); return; }
        if (res.error) { say(res.error, 'error'); return; }
        var r = res.result;
        var summary = Core.runSummary(r, limit);
        say(summary.text, summary.tone);
        entry.generation = ctl.live.session.generation;
        showResult(before, r);
      });
    };
    entry.runBtn.addEventListener('click', entry.run);

    function showResult(before, r) {
      result.replaceChildren();
      result.appendChild(renderCompare(before.outputs, r.outputs || [], { before: before.label, after: 'this try' }));

      var explainRow = el('div', 'pg-toolbar');
      var explainBtn = button('Explain the difference', 'btn-ghost');
      var answer = el('p', 'pg-diff-answer');
      answer.setAttribute('aria-live', 'polite');
      answer.hidden = true;
      explainRow.appendChild(explainBtn);
      result.appendChild(explainRow);
      result.appendChild(answer);
      explainBtn.addEventListener('click', function () {
        explainBtn.disabled = true;
        answer.hidden = false;
        answer.className = 'pg-diff-answer is-busy';
        answer.textContent = 'Asking Claude…';
        explainDifference(ctl, t, before, r.outputs || []).then(function (body) {
          answer.className = 'pg-diff-answer';
          answer.replaceChildren(el('span', null, body.explanation));
          var meta = (body.model || '') + (body.cost_usd != null ? ' · $' + Number(body.cost_usd).toFixed(4) : '') + ' · not saved';
          answer.appendChild(el('span', 'pg-diff-meta', meta));
        }).catch(function (err) {
          answer.className = 'pg-diff-answer is-error';
          answer.textContent = err.message;
        }).then(function () { explainBtn.disabled = false; });
      });

      entry.notice = el('div', 'pg-try-notice');
      entry.notice.setAttribute('role', 'note');
      entry.notice.appendChild(el('p', null,
        'This try changed the session’s variables. Later cells now see the tweaked state until you undo it.'));
      var undo = button('Undo this try');
      undo.setAttribute('title', 'Restart the session and rerun every cell up to this one');
      undo.addEventListener('click', function () {
        undo.disabled = true;
        ctl.live.session.restart().then(function (ok) {
          if (!ok) { undo.disabled = false; return; }
          return ctl.runUpTo();
        }).then(function () { undo.disabled = false; });
      });
      entry.notice.appendChild(undo);
      entry.undoneNote = el('p', 'pg-hint', 'The session has restarted since this try, so it no longer affects later cells.');
      entry.undoneNote.hidden = true;
      result.appendChild(entry.notice);
      result.appendChild(entry.undoneNote);
      result.hidden = false;
    }
    return box;
  }

  function explainDifference(ctl, t, before, afterOutputs) {
    var pg = ctl.live.pg;
    return fetch('/api/playgrounds/' + encodeURIComponent(pg.id) + '/explain-difference', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        before_code: before.code || ctl.original,
        after_code: t.code,
        before_output: Core.outputsText(before.outputs),
        after_output: Core.outputsText(afterOutputs),
        explanation: (ctl.guide && ctl.guide.explanation) || '',
        expect: t.expect || '',
      }),
    }).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (body) {
        if (!res.ok) throw new Error((body && body.detail) || 'Could not explain the difference (HTTP ' + res.status + ').');
        return body;
      });
    });
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
    var runnable = state.live && String(cell.source || '').trim();
    var ctl = runnable ? new CellCtl(cell, guide, state.live) : null;
    if (ctl) {
      state.live.cells[cell.index] = ctl;
      ctl.build(card);
    }
    var head = el('header', 'pg-card-head');
    var count = cell.execution_count != null ? 'In [' + cell.execution_count + ']' : 'Cell ' + cell.index;
    var countWrap = el('span', 'pg-count-wrap');
    var countEl = el('span', 'pg-count', count);
    countWrap.appendChild(countEl);
    if (ctl) {
      ctl.countEl = countEl;
      ctl.ranEl = el('span', 'chip chip-ok pg-ran', 'Ran');
      ctl.ranEl.hidden = true;
      ctl.ranEl.setAttribute('title', 'Ran in this session');
      countWrap.appendChild(ctl.ranEl);
    }
    head.appendChild(countWrap);
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

    var codeView = renderCode(cell.source, guide && guide.line_notes);
    card.appendChild(codeView);
    var noteHint = null;
    if (guide && guide.line_notes && guide.line_notes.length) {
      noteHint = el('p', 'pg-hint', 'Tap a highlighted line for its note.');
      card.appendChild(noteHint);
    }
    if (ctl) {
      ctl.codeView = codeView;
      ctl.noteHint = noteHint;
      card.appendChild(ctl.editInfo);
      card.appendChild(ctl.toolbar);
      card.appendChild(ctl.status);
    }

    var outputs = cell.outputs || [];
    if (outputs.length || ctl) {
      var outBox = el('div', 'pg-outputs');
      var label = el('p', 'pg-label', outputs.length ? 'Saved output' : 'No saved output');
      outBox.appendChild(label);
      var savedBox = el('div', 'pg-saved');
      outputs.forEach(function (o) { savedBox.appendChild(renderOutput(o)); });
      if (ctl) {
        ctl.outLabel = label;
        ctl.liveBox = el('div', 'pg-live');
        ctl.liveBox.hidden = true;
        ctl.savedBox = savedBox;
        ctl.savedToggle = button('Compare with saved output', 'btn-ghost');
        ctl.savedToggle.hidden = true;
        ctl.savedToggle.setAttribute('aria-pressed', 'false');
        ctl.savedToggle.addEventListener('click', function () {
          var show = savedBox.hidden;
          savedBox.hidden = !show;
          ctl.savedToggle.setAttribute('aria-pressed', show ? 'true' : 'false');
          ctl.savedToggle.textContent = show ? 'Hide saved output' : 'Compare with saved output';
        });
        // Shown only once a live output sits above it.
        var savedHead = el('p', 'pg-label pg-saved-label', 'Saved output (from the notebook)');
        savedBox.prepend(savedHead);
        ctl.savedHead = savedHead;
        savedHead.hidden = true;
        outBox.appendChild(ctl.liveBox);
        outBox.appendChild(ctl.savedToggle);
      }
      outBox.appendChild(savedBox);
      card.appendChild(outBox);
    }

    if (guide) {
      (guide.tries || []).forEach(function (t) { card.appendChild(renderTry(cell.source, t, ctl)); });
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

  /* ------------------------------------------------------------------ setup */

  /** The setup code that runs: what was saved, else the suggestion. */
  function setupCode(pg) {
    if (!pg) return '';
    return pg.setup_code != null ? String(pg.setup_code) : String(pg.setup_suggested || '');
  }

  /** The Setup card's summary line. */
  function setupSummary(code) {
    var n = String(code || '').split('\n').filter(function (l) { return l.trim(); }).length;
    return n ? 'Setup · ' + n + ' line' + (n === 1 ? '' : 's') : 'Setup · nothing to run';
  }

  /** What the Setup card says about the last run of the setup code. */
  function setupRunText(setup) {
    if (!setup) return '';
    if (setup.status === 'error') return Core.setupFailure(setup);
    if (setup.status === 'ok') return 'Ran when this session started.';
    return '';
  }

  /**
   * "Setup": a small, collapsed card with the code the runner runs before any
   * cell, on every start and Restart. Editable; Save stores it, Reset puts
   * the suggestion back in the editor.
   */
  function renderSetup(pg, state) {
    var box = el('details', 'pg-setup');
    box.id = 'pg-setup';
    var current = setupCode(pg);
    var summary = el('summary', null, setupSummary(current));
    box.appendChild(summary);
    box.appendChild(el('p', 'pg-hint', 'Imports this notebook uses but never wrote down. ' +
      'They run before any cell when the session starts, and again after Restart.'));
    var draft = current;
    var editor = buildEditor(current, function (value) { draft = value; paintNote(); });
    editor.ta.setAttribute('aria-label', 'Setup code (editable)');
    box.appendChild(editor.wrap);
    var actions = el('div', 'pg-setup-actions');
    var save = button('Save', 'btn-primary');
    var reset = button('Reset to suggested');
    actions.appendChild(save);
    actions.appendChild(reset);
    box.appendChild(actions);
    var note = el('p', 'pg-hint pg-setup-note');
    note.setAttribute('role', 'status');
    box.appendChild(note);
    var said = '';

    function paintNote() {
      var parts = [];
      if (said) parts.push(said);
      else if (draft !== current) parts.push('Not saved.');
      if (pg.setup_migration) parts.push('Saving needs ' + pg.setup_migration + ' applied; the suggestion still runs.');
      var live = state.live;
      if (!said && draft === current && live && live.session.id) {
        var ran = setupRunText(live.session.setup);
        if (ran) parts.push(ran);
      }
      note.textContent = parts.join(' ');
      note.className = 'pg-hint pg-setup-note' + (live && live.session.setup && live.session.setup.status === 'error' ? ' is-error' : '');
      save.disabled = draft === current && pg.setup_code != null;
    }

    save.addEventListener('click', function () {
      save.disabled = true;
      var value = draft;
      fetch('/api/playgrounds/' + encodeURIComponent(pg.id) + '/setup', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ setup_code: value }),
      }).then(function (res) {
        return res.json().catch(function () { return {}; }).then(function (body) {
          if (!res.ok) throw new Error((body && body.detail) || 'Could not save (HTTP ' + res.status + ').');
          pg.setup_code = value;
          current = value;
          summary.textContent = setupSummary(value);
          if (state.live) state.live.session.setSetupCode(value);
          said = state.live && state.live.session.id ? 'Saved. Restart to run it now.' : 'Saved.';
        });
      }).catch(function (err) {
        said = err.message;
      }).then(paintNote);
    });

    reset.addEventListener('click', function () {
      editor.ta.value = String(pg.setup_suggested || '');
      editor.ta.dispatchEvent(new Event('input'));
      said = '';
      paintNote();
    });

    if (state.live) state.live.paintSetup = function () { said = ''; paintNote(); };
    paintNote();
    return box;
  }

  function render(root, data, state) {
    var pg = data.playground;
    root.replaceChildren();
    if (state.live) {
      state.live.pg = pg;
      state.live.cells = {};
      state.live.codeIndices = (pg && pg.cells || []).filter(function (c) {
        return c.type === 'code' && String(c.source || '').trim();
      }).map(function (c) { return c.index; });
    }
    if (!pg) {
      root.appendChild(el('p', 'paper-muted',
        'No notebook yet. Import the paper’s .ipynb below and each code cell becomes a study card.'));
      return;
    }
    var head = el('div', 'pg-head');
    head.appendChild(el('h3', 'pg-title', pg.title || 'Notebook'));
    if (pg.notebook_name) head.appendChild(el('p', 'paper-muted pg-file', pg.notebook_name));
    head.appendChild(renderDatasets(pg.datasets));
    head.appendChild(renderSetup(pg, state));
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
    if (state.live) state.live.paintRan();
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
    var sessionEls = {
      bar: document.getElementById('playground-session'),
      chip: document.getElementById('playground-session-chip'),
      hint: document.getElementById('playground-session-hint'),
      restart: document.getElementById('playground-session-restart'),
      stop: document.getElementById('playground-session-stop'),
    };
    var state = { hide: false, data: null, live: null };
    try { state.hide = global.localStorage.getItem(HIDE_KEY) === 'true'; } catch (e) { /* default */ }

    function paintSession() {
      var live = state.live;
      if (!sessionEls.bar) return;
      sessionEls.bar.hidden = !live;
      if (!live) return;
      var session = live.session;
      var chip = Core.sessionChip(session.phase, session.message);
      sessionEls.chip.textContent = chip.label;
      sessionEls.chip.className = 'chip chip-' + chip.tone;
      sessionEls.hint.textContent = chip.hint;
      sessionEls.restart.textContent = session.id ? 'Restart' : 'Start';
      sessionEls.restart.disabled = session.phase === 'starting';
      sessionEls.stop.disabled = !session.id;
    }

    /** One runner session per open playground; a different playground gets a new one. */
    function ensureLive(pg) {
      if (!pg || !SessionApi.Session) {
        if (state.live) state.live.session.leave();
        state.live = null;
        paintSession();
        return;
      }
      if (state.live && state.live.session.playgroundId === pg.id) return;
      if (state.live) state.live.session.leave();
      var live = { pg: pg, cells: {}, codeIndices: [] };
      live.paintRan = function () {
        Object.keys(live.cells).forEach(function (k) { live.cells[k].paintRan(); });
      };
      live.paintSetup = function () {};
      live.session = new SessionApi.Session({
        playgroundId: pg.id,
        setupCode: setupCode(pg),
        onChange: function () { paintSession(); live.paintRan(); live.paintSetup(); },
      });
      state.live = live;
      paintSession();
      // Start the kernel as the tab opens, so the first Run is quick. A
      // runner that is not up says so on the chip, with how to start it.
      live.session.health().then(function (limits) {
        if (limits && !live.session.id && state.live === live) live.session.start();
      });
    }

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
        ensureLive(body.playground);
        if (state.live) state.live.session.setSetupCode(setupCode(body.playground));
        paint();
        showRequestedCell();
        var run = body.latest_run;
        if (run && (run.status === 'pending' || run.status === 'running')) follow(run.id);
      }).catch(function (err) {
        root.replaceChildren(el('p', 'pg-error', err.message));
        explainBtn.hidden = true;
        say('');
      });
    }

    function showRequestedCell() {
      var wanted = requestedCell(global.location.search);
      if (wanted == null) return;
      var card = document.getElementById('pg-cell-' + wanted);
      if (!card) return;
      card.classList.add('is-target');
      card.scrollIntoView({ block: 'start' });
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
          // New code and possibly new cached datasets: the kernel reads the
          // dataset manifest when it starts, so give it a fresh one.
          if (state.live && state.live.session.id) state.live.session.restart();
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

    if (sessionEls.restart) {
      sessionEls.restart.addEventListener('click', function () {
        if (state.live) state.live.session.restart();
      });
      sessionEls.stop.addEventListener('click', function () {
        if (state.live) state.live.session.stop();
      });
    }

    // Shift+Enter runs the focused cell (or the try it is in), as in Jupyter.
    root.addEventListener('keydown', function (ev) {
      if (!Core.isRunShortcut(ev) || !state.live || !ev.target.closest) return;
      var card = ev.target.closest('.pg-card[data-cell]');
      var ctl = card && state.live.cells[card.getAttribute('data-cell')];
      if (!ctl) return;
      ev.preventDefault();
      var tryBox = ev.target.closest('.pg-try');
      var entry = tryBox && ctl.tries.filter(function (t) { return t.box === tryBox; })[0];
      if (entry) { if (!entry.runBtn.disabled) entry.run(); return; }
      if (!ctl.runBtn.disabled) ctl.run();
    });

    // Leaving the page closes the kernel; the runner would otherwise hold
    // it (and a session slot) until it idles out.
    global.addEventListener('pagehide', function () {
      if (state.live) state.live.session.leave();
    });
    global.addEventListener('pageshow', function (ev) {
      if (ev.persisted && state.live) {
        state.live.session.markGone('Stopped when you left the page. Run a cell to start again.');
      }
    });

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
    requestedCell: requestedCell,
    setupCode: setupCode,
    setupSummary: setupSummary,
    setupRunText: setupRunText,
    // The Listen page's follow-along panel renders its card with these, so a
    // cell looks the same (and is sanitized the same way) in both places.
    renderCode: renderCode,
    renderOutput: renderOutput,
    HIDE_KEY: HIDE_KEY,
  };
  global.Playground = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
