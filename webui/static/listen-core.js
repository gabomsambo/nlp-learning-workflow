/*
 * Listen: the pure part shared by the desktop Listen page and the cloud reader.
 *
 * Moved out of listen.js unchanged, apart from groupTokens(), which lifts the
 * "which words share a fact-check mark" decision out of the read-along painter
 * so both pages make it the same way. No DOM, no network, no timers: node --test
 * imports this file, and cloud/reader copies it next to readalong.js and
 * factcheck.js (cloud/reader/scripts/build-assets.sh).
 */
(function (global) {
  'use strict';

  function formatClock(seconds) {
    if (!isFinite(seconds) || seconds < 0) return '0:00';
    seconds = Math.floor(seconds);
    var h = Math.floor(seconds / 3600);
    var m = Math.floor((seconds % 3600) / 60);
    var s = seconds % 60;
    function pad(n) { return (n < 10 ? '0' : '') + n; }
    if (h) return h + ':' + pad(m) + ':' + pad(s);
    return m + ':' + pad(s);
  }

  function minuteSecondPhrase(seconds) {
    seconds = Math.floor(Math.max(0, isFinite(seconds) ? seconds : 0));
    var h = Math.floor(seconds / 3600);
    var m = Math.floor((seconds % 3600) / 60);
    var s = seconds % 60;
    if (h) return h + ' hours ' + m + ' minutes ' + s;
    return m + ' minutes ' + s;
  }

  function formatValueText(current, duration) {
    if (!isFinite(duration) || duration <= 0) return minuteSecondPhrase(current);
    return minuteSecondPhrase(current) + ' of ' + minuteSecondPhrase(duration);
  }

  /*
   * The lines the fact-check engine numbered.
   *
   * The engine numbers claims by the index of `script.splitlines()`, and the
   * TTS assembler walks the same list, so this has to agree with Python
   * exactly — a line off by one paints every highlight on the wrong words.
   * Python's splitlines() breaks on more than \n and \r (vertical tab, form
   * feed, the file/group/record/unit separators, NEL, and the two Unicode
   * line separators), and it leaves no trailing empty line for a trailing
   * newline. An empty script has no lines at all.
   */
  function scriptLines(text) {
    var source = text == null ? '' : String(text);
    if (!source) return [];
    var lines = source.split(/\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]/);
    if (lines.length > 1 && lines[lines.length - 1] === '') lines.pop();
    return lines;
  }

  /*
   * Where a claim's highlight sits in its line.
   *
   * The engine stores character offsets into the line as Python counts them —
   * code points. JavaScript counts UTF-16 code units. The two agree until a
   * character outside the Basic Multilingual Plane turns up, and a script with
   * one emoji in it would then paint every highlight after it one character to
   * the left. When the stored offsets do not produce the claim's own span
   * text, look the text up in the line instead.
   *
   * Returns null when the span cannot be placed at all: a claim whose text is
   * not in the line is not highlighted rather than highlighted in the wrong
   * place.
   */
  function resolveSpan(line, claim) {
    line = String(line == null ? '' : line);
    claim = claim || {};
    var start = Number(claim.span_start);
    var end = Number(claim.span_end);
    if (!Number.isInteger(start) || !Number.isInteger(end)) return null;
    if (start < 0 || end <= start || end > line.length) return null;
    var text = claim.span_text == null ? '' : String(claim.span_text);
    if (text && line.slice(start, end) !== text) {
      var at = line.indexOf(text);
      if (at === -1) return null;
      if (line.indexOf(text, at + 1) === -1) {
        start = at;
        end = at + text.length;
      }
    }
    return { start: start, end: end };
  }

  function visibleClaimSpans(line, claims, showSupported) {
    line = String(line == null ? '' : line);
    var result = [];
    (claims || []).forEach(function (claim, index) {
      if (!claim || (claim.verdict === 'supported' && !showSupported)) return;
      var span = resolveSpan(line, claim);
      if (!span) return;
      result.push({ claim: claim, index: index, start: span.start, end: span.end });
    });
    return result;
  }

  /*
   * Split one script line on every claim boundary. Overlapping claims therefore
   * share the middle segment instead of one mark swallowing the other. Each
   * returned segment retains every covering claim for the evidence popover.
   */
  function claimSegments(line, claims, showSupported) {
    line = line == null ? '' : String(line);
    var spans = visibleClaimSpans(line, claims, showSupported);
    if (!spans.length) return [{ start: 0, end: line.length, text: line, claims: [] }];
    var boundaries = [0, line.length];
    spans.forEach(function (span) {
      boundaries.push(span.start, span.end);
    });
    boundaries = boundaries.filter(function (value, index, all) {
      return all.indexOf(value) === index;
    }).sort(function (a, b) { return a - b; });
    var out = [];
    for (var i = 0; i < boundaries.length - 1; i++) {
      var start = boundaries[i];
      var end = boundaries[i + 1];
      if (end <= start) continue;
      var covering = spans.filter(function (span) {
        return span.start <= start && span.end >= end;
      }).map(function (span) { return span.claim; });
      out.push({ start: start, end: end, text: line.slice(start, end), claims: covering });
    }
    return out;
  }

  /*
   * Claims bucketed by the script line they were found on, ignoring any whose
   * line_index is not a line of this script.
   */
  function claimsByLine(claims, lineCount) {
    var byLine = {};
    (claims || []).forEach(function (claim) {
      var line = Number(claim && claim.line_index);
      if (!Number.isInteger(line) || line < 0 || line >= lineCount) return;
      if (!byLine[line]) byLine[line] = [];
      byLine[line].push(claim);
    });
    return byLine;
  }

  /*
   * The words of one line (readalong.js tokenize()) arranged for painting.
   *
   * A word belongs to the claim segment its first character falls in. Words
   * that share a flagged segment form one group, so a claim that wraps over
   * several words is one mark with one popover; every other word stands alone
   * with `segment: null`. `index` is the word's position in the line, which is
   * what the timeline's per-word times and a tap-to-seek are keyed by.
   *
   *   [{ segment: null|segment, words: [{ token, index }, ...] }, ...]
   */
  function groupTokens(tokens, segments) {
    var groups = [];
    var current = null;
    (tokens || []).forEach(function (token, index) {
      var segment = null;
      for (var i = 0; i < (segments || []).length; i++) {
        if (token.start >= segments[i].start && token.start < segments[i].end) {
          segment = segments[i];
          break;
        }
      }
      var flagged = segment && segment.claims.length ? segment : null;
      var entry = { token: token, index: index };
      if (flagged && current && current.segment === flagged) {
        current.words.push(entry);
        return;
      }
      current = { segment: flagged, words: [entry] };
      groups.push(current);
      if (!flagged) current = null;
    });
    return groups;
  }

  var api = {
    formatClock: formatClock,
    minuteSecondPhrase: minuteSecondPhrase,
    formatValueText: formatValueText,
    scriptLines: scriptLines,
    resolveSpan: resolveSpan,
    visibleClaimSpans: visibleClaimSpans,
    claimSegments: claimSegments,
    claimsByLine: claimsByLine,
    groupTokens: groupTokens,
  };

  global.ListenCore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
