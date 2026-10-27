/*
 * Playground: the pure part of the live tab (docs/PLAYGROUND.md).
 *
 * No DOM, no network, no timers, so node --test covers all of it
 * (tests/js/playground-core.test.js). playground.js renders with these;
 * playground-session.js talks to the runner.
 */
(function (global) {
  'use strict';

  var MAX_DIFF_CELLS = 250000;
  var INDENT = '    ';

  /* -------------------------------------------------------------- code diffs */

  /** Line diff (LCS). Returns [{op: 'same'|'del'|'add', text}] in order. */
  function lineDiff(before, after) {
    var a = String(before == null ? '' : before).split('\n');
    var b = String(after == null ? '' : after).split('\n');
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
   * Only the changed lines and `context` unchanged lines around each change.
   * Skipped runs become one {op: 'gap', count} row.
   */
  function compactDiff(rows, context) {
    context = context == null ? 1 : context;
    var keep = rows.map(function () { return false; });
    rows.forEach(function (row, i) {
      if (row.op === 'same') return;
      for (var k = Math.max(0, i - context); k <= Math.min(rows.length - 1, i + context); k++) keep[k] = true;
    });
    var out = [];
    var skipped = 0;
    rows.forEach(function (row, i) {
      if (keep[i]) {
        if (skipped) out.push({ op: 'gap', count: skipped });
        skipped = 0;
        out.push(row);
      } else {
        skipped += 1;
      }
    });
    if (skipped && out.length) out.push({ op: 'gap', count: skipped });
    return out;
  }

  function editSummary(original, edited) {
    var added = 0;
    var removed = 0;
    lineDiff(original, edited).forEach(function (row) {
      if (row.op === 'add') added += 1;
      else if (row.op === 'del') removed += 1;
    });
    if (!added && !removed) return 'No changes from the original.';
    var parts = [];
    if (added) parts.push(added + ' line' + (added === 1 ? '' : 's') + ' added');
    if (removed) parts.push(removed + ' removed');
    return 'Your edit: ' + parts.join(', ') + '.';
  }

  /* ------------------------------------------------------------- running */

  /**
   * The cells "Run up to here" executes, in order: every earlier code cell
   * this session has not run yet, then the target itself (always).
   * `codeIndices` are the runnable cells in notebook order; `ran` answers
   * has(index).
   */
  function planRunUpTo(codeIndices, target, ran) {
    var plan = [];
    (codeIndices || []).forEach(function (index) {
      if (index < target && !(ran && ran.has(index))) plan.push(index);
    });
    plan.push(target);
    return plan;
  }

  function formatElapsed(ms) {
    if (!isFinite(ms) || ms < 0) ms = 0;
    var s = ms / 1000;
    if (s < 10) return s.toFixed(1) + ' s';
    if (s < 60) return Math.floor(s) + ' s';
    var m = Math.floor(s / 60);
    var rest = Math.floor(s % 60);
    return m + ':' + (rest < 10 ? '0' : '') + rest;
  }

  /** The next timeout to offer after a timeout: double, capped. null at the cap. */
  function longerTimeout(current, max) {
    if (!isFinite(current) || current <= 0 || !isFinite(max)) return null;
    if (current >= max) return null;
    return Math.min(max, Math.max(current * 2, current + 30));
  }

  function seconds(n) {
    n = Math.round(n);
    if (n < 120 || n % 60) return n + ' s';
    return n / 60 + ' min';
  }

  /**
   * One line under a cell after it ran: {tone: ok|error|warn, text}.
   * `timeoutS` is the limit the run had.
   */
  function runSummary(result, timeoutS) {
    var r = result || {};
    var took = formatElapsed(r.execution_ms || 0);
    var count = r.execution_count != null ? 'In [' + r.execution_count + '] · ' : '';
    var lines = [];
    var tone = 'ok';
    if (r.kernel_restarted) {
      tone = 'error';
      lines.push('The kernel restarted, so every variable is gone. Use “Run up to here” to rebuild them.');
    } else if (r.status === 'timeout') {
      tone = 'warn';
      lines.push('Stopped at the ' + seconds(timeoutS || 0) + ' limit. Variables from earlier cells are kept.');
    } else if (r.interrupted) {
      tone = 'warn';
      lines.push('Interrupted after ' + took + '.');
    } else if (r.status === 'error') {
      tone = 'error';
      lines.push(count + 'Raised an error after ' + took + '.');
    } else {
      lines.push(count + 'Ran in ' + took + '.');
    }
    if (r.truncated) lines.push('Output cut at the runner’s size limit.');
    if (r.dropped_outputs) {
      lines.push(r.dropped_outputs + ' later output' + (r.dropped_outputs === 1 ? ' was' : 's were') + ' dropped.');
    }
    return { tone: tone, text: lines.join(' ') };
  }

  /** What a failed proxy call means to the person. */
  function errorMessage(status, detail) {
    var text = typeof detail === 'string' ? detail : (detail && detail.message) || '';
    if (status === 0) return 'Could not reach the app. Check the connection and try again.';
    if (status === 404) return 'The session had ended (it closes after 30 idle minutes), so its variables are gone. Run again to start a new one.';
    if (status === 409) return 'The session is still running other code. Wait for it, or press Interrupt.';
    if (status === 413) return 'This code is too large for the runner.';
    if (status === 429) return text || 'The runner already has the most sessions it allows. Close the Playground in another tab and try again.';
    if (status === 504) return 'The runner did not answer in time. ' + (text || '');
    return text || 'The runner answered HTTP ' + status + '.';
  }

  /**
   * The session chip: {label, tone, hint}. phase is idle, starting, ready,
   * busy, stopped, unavailable or error.
   */
  function sessionChip(phase, message) {
    switch (phase) {
      case 'starting': return { label: 'Starting…', tone: 'accent', hint: 'Starting a Python kernel. This takes a few seconds.' };
      case 'ready': return { label: 'Ready', tone: 'ok', hint: message || '' };
      case 'busy': return { label: 'Busy', tone: 'warn', hint: message || 'Running code.' };
      case 'stopped': return { label: 'Stopped', tone: 'neutral', hint: message || 'Run a cell to start a new session.' };
      case 'unavailable': return { label: 'Runner not running', tone: 'danger', hint: message || 'Start it with ./workstation up.' };
      case 'error': return { label: 'Not started', tone: 'danger', hint: message || '' };
      default: return { label: 'Not started', tone: 'neutral', hint: message || 'Run a cell to start a session.' };
    }
  }

  /**
   * Remove the indent every non-blank line shares. Older Jupyter stripped a
   * cell's leading indent before running it, so saved notebooks can carry
   * cells like "  x = 1\n  print(x)" that ran there and are an
   * IndentationError for the runner's IPython. Comment and blank lines do not
   * count towards the shared indent; a line without it is left as it is. A
   * cell with no common indent is returned unchanged.
   */
  function dedentCell(code) {
    var text = String(code == null ? '' : code);
    var lines = text.split('\n');
    var common = null;
    lines.forEach(function (line) {
      // Blank lines and comments do not count: Python ignores their indent.
      if (!line.trim() || /^[ \t]*#/.test(line)) return;
      var indent = /^[ \t]*/.exec(line)[0];
      if (common === null) { common = indent; return; }
      var k = 0;
      while (k < common.length && k < indent.length && common[k] === indent[k]) k++;
      common = common.slice(0, k);
    });
    if (!common) return text;
    return lines.map(function (line) {
      return line.slice(0, common.length) === common ? line.slice(common.length) : line.replace(/^[ \t]+$/, '');
    }).join('\n');
  }

  function isRunShortcut(ev) {
    return !!ev && ev.key === 'Enter' && !!ev.shiftKey && !ev.ctrlKey && !ev.metaKey && !ev.altKey;
  }

  /* ------------------------------------------------------- comparing outputs */

  var TEXT_KINDS = ['stream', 'text', 'error'];

  function isTextKind(out) { return !!out && TEXT_KINDS.indexOf(out.kind) !== -1; }

  /** Sanitized tag tree (notebook.py::sanitize_html) as plain text, rows on lines. */
  function treeText(nodes) {
    var parts = [];
    function walk(node) {
      if (typeof node === 'string') { parts.push(node.split(/\s+/).join(' ')); return; }
      if (!node || typeof node !== 'object') return;
      (node.c || []).forEach(walk);
      var t = node.t;
      if (t === 'td' || t === 'th') parts.push('\t');
      else if (['tr', 'p', 'div', 'br', 'li', 'caption', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'].indexOf(t) !== -1) parts.push('\n');
    }
    (nodes || []).forEach(walk);
    return parts.join('').split('\n').map(function (line) {
      return line.split('\t').map(function (s) { return s.trim(); }).join('\t').trim();
    }).filter(Boolean).join('\n');
  }

  function outputText(out) {
    if (!out) return '';
    if (isTextKind(out)) return out.text || '';
    if (out.kind === 'html') return out.plain || treeText(out.tree);
    if (out.kind === 'png') return '[figure: ' + (out.alt || 'image') + ']';
    return '[output of type ' + ((out.mime || []).join(', ') || 'unknown') + ']';
  }

  /** All outputs as one plain text, for the "Explain the difference" request. */
  function outputsText(outputs, maxChars) {
    var text = (outputs || []).map(outputText).filter(Boolean).join('\n');
    var cap = maxChars || 20000;
    return text.length > cap ? text.slice(0, cap) + '\n[cut]' : text;
  }

  /**
   * Changed lines between two texts, for highlighting: {before: [bool],
   * after: [bool]}, one flag per line of each.
   */
  function textChanges(before, after) {
    var b = [];
    var a = [];
    lineDiff(before, after).forEach(function (row) {
      if (row.op === 'same') { b.push(false); a.push(false); }
      else if (row.op === 'del') b.push(true);
      else a.push(true);
    });
    return { before: b, after: a };
  }

  /** Every td/th of a tag tree, in document order, grouped by row: [[text]]. */
  function tableCells(nodes) {
    var rows = [];
    var current = null;
    function cellText(node) {
      if (typeof node === 'string') return node;
      return (node && node.c || []).map(cellText).join('');
    }
    function walk(node) {
      if (!node || typeof node !== 'object') return;
      if (node.t === 'tr') {
        current = [];
        rows.push(current);
        (node.c || []).forEach(walk);
        current = null;
        return;
      }
      if (node.t === 'td' || node.t === 'th') {
        if (!current) { current = []; rows.push(current); }
        current.push(cellText(node).split(/\s+/).join(' ').trim());
        return;
      }
      (node.c || []).forEach(walk);
    }
    (nodes || []).forEach(walk);
    return rows;
  }

  /**
   * Which table cells differ between two outputs rendered from the same
   * shape: a flat list of flags in td/th document order, or null when the
   * shapes differ (then nothing is highlighted).
   */
  function tableChanges(treeA, treeB) {
    var a = tableCells(treeA);
    var b = tableCells(treeB);
    if (!a.length || a.length !== b.length) return null;
    var flags = [];
    for (var r = 0; r < a.length; r++) {
      if (a[r].length !== b[r].length) return null;
      for (var c = 0; c < a[r].length; c++) flags.push(a[r][c] !== b[r][c]);
    }
    return flags;
  }

  /**
   * Pair up two output lists for a Before / After. Outputs are matched by
   * position; a pair gets highlights only when both sides are text, or both
   * are tables of the same shape.
   *   [{before, after, text: {before, after}|null, table: [bool]|null}]
   */
  function compareOutputs(before, after) {
    var b = before || [];
    var a = after || [];
    var pairs = [];
    for (var i = 0; i < Math.max(a.length, b.length); i++) {
      var pair = { before: b[i] || null, after: a[i] || null, text: null, table: null };
      if (isTextKind(pair.before) && isTextKind(pair.after)) {
        pair.text = textChanges(pair.before.text || '', pair.after.text || '');
      } else if (pair.before && pair.after && pair.before.kind === 'html' && pair.after.kind === 'html') {
        pair.table = tableChanges(pair.before.tree, pair.after.tree);
      }
      pairs.push(pair);
    }
    return pairs;
  }

  function sameOutputs(before, after) {
    return outputsText(before, 1e9) === outputsText(after, 1e9) &&
      (before || []).length === (after || []).length;
  }

  /* ------------------------------------------------------------- the editor */

  function lineStart(value, pos) {
    return value.lastIndexOf('\n', pos - 1) + 1;
  }

  /**
   * Tab / Shift+Tab over a selection. A caret with no selection and no
   * outdent inserts four spaces. Returns {value, start, end}.
   */
  function indentLines(value, start, end, outdent) {
    value = String(value || '');
    if (start === end && !outdent) {
      return { value: value.slice(0, start) + INDENT + value.slice(end), start: start + 4, end: start + 4 };
    }
    var from = lineStart(value, start);
    // A selection ending at the start of a line does not include that line.
    var to = end > start && value[end - 1] === '\n' ? end - 1 : end;
    var block = value.slice(from, to);
    var lines = block.split('\n');
    var firstShift = 0;
    var total = 0;
    var changed = lines.map(function (line, i) {
      var shift;
      if (outdent) {
        var m = /^( {1,4}|\t)/.exec(line);
        shift = m ? -m[0].length : 0;
        line = m ? line.slice(m[0].length) : line;
      } else {
        shift = INDENT.length;
        line = INDENT + line;
      }
      if (i === 0) firstShift = shift;
      total += shift;
      return line;
    });
    var next = value.slice(0, from) + changed.join('\n') + value.slice(to);
    var newStart = Math.max(from, start + firstShift);
    var newEnd = start === end ? newStart : end + total;
    return { value: next, start: newStart, end: Math.max(newStart, newEnd) };
  }

  /** What Enter inserts at `pos`: a newline, the line's indent, and one more after a colon. */
  function newlineIndent(value, pos) {
    value = String(value || '');
    var from = lineStart(value, pos);
    var line = value.slice(from, pos);
    var indent = /^[ \t]*/.exec(line)[0];
    if (/:\s*(#.*)?$/.test(line)) indent += INDENT;
    return '\n' + indent;
  }

  function lineNumbers(value) {
    var count = String(value == null ? '' : value).split('\n').length;
    var out = [];
    for (var i = 1; i <= count; i++) out.push(String(i));
    return out.join('\n');
  }

  var api = {
    lineDiff: lineDiff,
    compactDiff: compactDiff,
    editSummary: editSummary,
    planRunUpTo: planRunUpTo,
    formatElapsed: formatElapsed,
    longerTimeout: longerTimeout,
    seconds: seconds,
    runSummary: runSummary,
    errorMessage: errorMessage,
    sessionChip: sessionChip,
    isRunShortcut: isRunShortcut,
    dedentCell: dedentCell,
    treeText: treeText,
    outputText: outputText,
    outputsText: outputsText,
    textChanges: textChanges,
    tableCells: tableCells,
    tableChanges: tableChanges,
    compareOutputs: compareOutputs,
    sameOutputs: sameOutputs,
    indentLines: indentLines,
    newlineIndent: newlineIndent,
    lineNumbers: lineNumbers,
  };
  global.PlaygroundCore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
