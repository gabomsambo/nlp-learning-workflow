/*
 * Read-along: the pure part.
 *
 * Everything here turns an episode's stored timeline (audio_metadata.timeline,
 * written by nlp_pillars/tts/alignment.py) and a playback time into "which
 * line, which word", and a click on a line or word back into a time to seek
 * to. No DOM, no audio element, no timers: listen.js owns those and this file
 * is what node --test covers.
 *
 * The words are the script's own. The timeline stores, per spoken line,
 * [start_ms, end_ms, heard] for each whitespace-separated word of what the
 * narrator says, so tokenize() splits a line the way Python's
 * spoken_line_text() + str.split() did. A line whose word count disagrees with
 * the stored one falls back to line-level timing instead of putting the
 * highlight on the wrong word.
 */
(function (global) {
  'use strict';

  var HOST_PREFIX = /^\[HOST\]:\s*/i;
  var STANDALONE_CUE = /^\[(?:PAUSE|MUSIC|SFX|TRANSITION)\]\s*$/i;

  // A word stays lit this long past its end before the highlight lets go, so a
  // pause in the speech reads as a pause and not as a stuck highlight.
  var WORD_HOLD_MS = 1500;
  // Landing a hair before a word's start means its first syllable is heard.
  var SEEK_LEAD_MS = 150;

  /*
   * What the narrator says for one raw script line, and where each word sits.
   * Returns null for a blank line or a standalone cue. Offsets are UTF-16
   * positions in the raw line, which is what the fact-check spans are resolved
   * against in listen.js.
   */
  function tokenize(rawLine) {
    var line = rawLine == null ? '' : String(rawLine);
    var lead = line.length - line.replace(/^\s+/, '').length;
    var body = line.trim();
    if (!body || STANDALONE_CUE.test(body)) return null;
    var host = HOST_PREFIX.exec(body);
    var skip = lead + (host ? host[0].length : 0);
    var spoken = line.slice(skip);
    var tokens = [];
    var re = /\S+/g;
    var match;
    while ((match = re.exec(spoken))) {
      tokens.push({ text: match[0], start: skip + match.index, end: skip + match.index + match[0].length });
    }
    return tokens.length ? tokens : null;
  }

  /*
   * 'current'  the timeline describes this script's text
   * 'stale'    the script was edited after the audio was timed
   * 'missing'  no usable timeline (never timed, or an unknown format)
   */
  function timelineState(timeline, scriptSha) {
    if (!timeline || timeline.version !== 1 || !Array.isArray(timeline.lines) || !timeline.lines.length) {
      return 'missing';
    }
    if (scriptSha && timeline.script_sha256 && timeline.script_sha256 !== scriptSha) return 'stale';
    return 'current';
  }

  /*
   * The timeline as arrays the playback loop can binary-search.
   *
   * `lines` is ordered by start time and by line number. `words` is present on
   * a line only when the timeline carries per-word times for it AND the count
   * agrees with how this page splits the line. Returns null when the timeline
   * cannot be used at all.
   */
  function buildModel(timeline, scriptLines, scriptSha) {
    var state = timelineState(timeline, scriptSha);
    if (state !== 'current') return null;
    var lines = [];
    var byLine = {};
    var disagreements = 0;
    timeline.lines.forEach(function (entry) {
      var lineNo = Number(entry && entry.line);
      var start = Number(entry && entry.start_ms);
      var end = Number(entry && entry.end_ms);
      if (!Number.isInteger(lineNo) || !isFinite(start) || !isFinite(end)) return;
      if (lineNo < 0 || lineNo >= scriptLines.length) return;
      var tokens = tokenize(scriptLines[lineNo]);
      if (!tokens) return;
      var words = null;
      if (Array.isArray(entry.words)) {
        if (entry.words.length === tokens.length) {
          words = entry.words.map(function (w) {
            return { start: Number(w[0]), end: Number(w[1]), heard: w[2] !== 0 };
          });
        } else {
          disagreements += 1;
        }
      }
      var item = { line: lineNo, start: start, end: Math.max(end, start), tokens: tokens, words: words };
      byLine[lineNo] = lines.length;
      lines.push(item);
    });
    lines.sort(function (a, b) { return a.start - b.start; });
    lines.forEach(function (item, index) { byLine[item.line] = index; });
    if (!lines.length) return null;
    return {
      lines: lines,
      byLine: byLine,
      wordLevel: lines.some(function (item) { return !!item.words; }),
      disagreements: disagreements,
      durationMs: Number(timeline.duration_ms) || lines[lines.length - 1].end,
    };
  }

  /* Index into model.lines of the last line that has started by `ms`; -1 before the first. */
  function lineIndexAt(model, ms) {
    if (!model || !model.lines.length) return -1;
    var lo = 0;
    var hi = model.lines.length - 1;
    var found = -1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (model.lines[mid].start <= ms) {
        found = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return found;
  }

  /*
   * Index of the word being spoken within a line, or -1: the line has no word
   * times, nothing has started, or the last word ended long enough ago that the
   * speaker has clearly paused.
   */
  function wordIndexAt(line, ms) {
    if (!line || !line.words || !line.words.length) return -1;
    var words = line.words;
    var lo = 0;
    var hi = words.length - 1;
    var found = -1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (words[mid].start <= ms) {
        found = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    if (found === -1) return -1;
    if (ms > words[found].end + WORD_HOLD_MS && found === words.length - 1) return -1;
    return found;
  }

  /* Where playback is, as { line: index into model.lines or -1, word: index or -1 }. */
  function positionAt(model, ms) {
    var lineIdx = lineIndexAt(model, ms);
    if (lineIdx === -1) return { line: -1, word: -1 };
    return { line: lineIdx, word: wordIndexAt(model.lines[lineIdx], ms) };
  }

  /* Seconds to seek to for a script line number, or null if that line has no timing. */
  function seekSecondsForLine(model, lineNo) {
    if (!model) return null;
    var index = model.byLine[lineNo];
    if (index == null) return null;
    return Math.max(0, model.lines[index].start - SEEK_LEAD_MS) / 1000;
  }

  /* Seconds to seek to for one word of a line; falls back to the line's start. */
  function seekSecondsForWord(model, lineNo, wordIndex) {
    if (!model) return null;
    var index = model.byLine[lineNo];
    if (index == null) return null;
    var line = model.lines[index];
    var word = line.words && line.words[wordIndex];
    var start = word ? word.start : line.start;
    return Math.max(0, start - SEEK_LEAD_MS) / 1000;
  }

  /*
   * Where to scroll a reading surface so the current line sits at `anchor`
   * (0 = top edge, 1 = bottom edge) of its visible height. Clamped to the
   * scrollable range. Pure arithmetic so the target is the same on every
   * engine and can be tested.
   */
  function scrollTopFor(lineTop, lineHeight, viewHeight, contentHeight, anchor) {
    var wanted = lineTop - viewHeight * anchor + lineHeight / 2;
    var max = Math.max(0, contentHeight - viewHeight);
    return Math.min(Math.max(wanted, 0), max);
  }

  /* "3:07", "1:02:09": the timestamp beside a line. */
  function clockLabel(ms) {
    var total = Math.max(0, Math.floor((Number(ms) || 0) / 1000));
    var h = Math.floor(total / 3600);
    var m = Math.floor((total % 3600) / 60);
    var s = total % 60;
    var ss = (s < 10 ? '0' : '') + s;
    if (h) return h + ':' + (m < 10 ? '0' : '') + m + ':' + ss;
    return m + ':' + ss;
  }

  var api = {
    WORD_HOLD_MS: WORD_HOLD_MS,
    SEEK_LEAD_MS: SEEK_LEAD_MS,
    tokenize: tokenize,
    timelineState: timelineState,
    buildModel: buildModel,
    lineIndexAt: lineIndexAt,
    wordIndexAt: wordIndexAt,
    positionAt: positionAt,
    seekSecondsForLine: seekSecondsForLine,
    seekSecondsForWord: seekSecondsForWord,
    scrollTopFor: scrollTopFor,
    clockLabel: clockLabel,
  };

  global.ReadAlong = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
