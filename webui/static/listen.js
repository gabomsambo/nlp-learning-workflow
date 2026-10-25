/*
 * Listen: episode library and studio.
 *
 * The library is newest-first and grouped by pillar. Play hands the page
 * <audio> to NlpPlayer so the shell mini player keeps the same position.
 * Script text is built with text nodes. Nothing here calls generate or
 * synthesis except the two explicit buttons.
 *
 * Nodes are createElement + textContent. Nothing concatenates markup.
 */
(function (global) {
  'use strict';

  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  var CUE_RE = /\[HOST\]:|\[(?:MUSIC|SFX|PAUSE|TRANSITION)\]/g;

  function episodeInitial(title) {
    var text = String(title || '');
    for (var i = 0; i < text.length; i++) {
      if (/[0-9A-Za-z]/.test(text.charAt(i))) return text.charAt(i).toUpperCase();
    }
    return 'E';
  }

  function paperPageHref(id) {
    if (typeof id !== 'string' || !id) return null;
    return '/papers/' + encodeURIComponent(id);
  }

  function audioHref(fileName) {
    if (typeof fileName !== 'string' || !fileName) return null;
    if (fileName.indexOf('/') !== -1 || fileName.indexOf('\\') !== -1 || fileName.indexOf('..') !== -1) {
      return null;
    }
    for (var i = 0; i < fileName.length; i++) {
      if (fileName.charCodeAt(i) < 32) return null;
    }
    return '/api/podcast/audio/' + encodeURIComponent(fileName);
  }

  function formatDuration(seconds) {
    if (seconds == null || typeof seconds === 'boolean') return '';
    var n = Number(seconds);
    if (!isFinite(n)) return '';
    var total = Math.round(n);
    if (total <= 0) return '';
    var minutes = Math.floor(total / 60);
    var secs = total % 60;
    var hours = Math.floor(minutes / 60);
    minutes = minutes % 60;
    function pad(v) { return (v < 10 ? '0' : '') + v; }
    if (hours) return hours + ':' + pad(minutes) + ':' + pad(secs);
    return minutes + ':' + pad(secs);
  }

  function formatCreated(value) {
    if (!value) return '';
    var match = String(value).match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (!match) return '';
    var month = Number(match[2]);
    if (month < 1 || month > 12) return '';
    return Number(match[3]) + ' ' + MONTHS[month - 1] + ' ' + match[1];
  }

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

  function scriptSegments(text) {
    var src = String(text || '');
    var out = [];
    var re = new RegExp(CUE_RE.source, 'g');
    var last = 0;
    var match;
    while ((match = re.exec(src))) {
      if (match.index > last) out.push({ kind: 'text', text: src.slice(last, match.index) });
      var token = match[0];
      out.push({ kind: token.indexOf('[HOST]') === 0 ? 'host' : 'cue', text: token });
      last = match.index + token.length;
    }
    if (last < src.length || !out.length) out.push({ kind: 'text', text: src.slice(last) });
    return out;
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

  var Factcheck = global.Factcheck ||
    (typeof require === 'function' ? require('./factcheck.js') : {});
  var claimTone = Factcheck.claimTone;
  var claimBucket = Factcheck.claimBucket;
  var claimPresentation = Factcheck.claimPresentation;
  var strongestClaim = Factcheck.strongestClaim;
  var issueClaims = Factcheck.issueClaims;
  var factcheckSummary = Factcheck.factcheckSummary;
  var isFactcheckStale = Factcheck.isFactcheckStale;

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

  function requestedLine(value, lineCount) {
    if (value == null || !/^\d+$/.test(String(value))) return null;
    var line = Number(value);
    return line >= 0 && line < lineCount ? line : null;
  }

  function groupEpisodes(rows) {
    var ordered = (rows || []).slice().sort(function (a, b) {
      var as = String((a && a.created_sort) || '');
      var bs = String((b && b.created_sort) || '');
      if (as === bs) return 0;
      return as < bs ? 1 : -1;
    });
    var groups = [];
    var index = {};
    ordered.forEach(function (row) {
      if (!row) return;
      var key = row.pillar_id || '';
      if (!index[key]) {
        index[key] = {
          pillar_id: key,
          pillar_name: row.pillar_name || 'No pillar',
          episodes: [],
        };
        groups.push(index[key]);
      }
      index[key].episodes.push(row);
    });
    return groups;
  }

  function viewMode(search) {
    var params = typeof search === 'string' || search == null
      ? new URLSearchParams(search || '')
      : search;
    if (params.get('studio') === '1' || params.get('paper') || params.get('run')) return 'studio';
    return 'episodes';
  }

  function rowFromParts(summary, detail, catalog) {
    summary = summary || {};
    var script = (detail && detail.script) || {};
    var audio = script.audio_metadata || {};
    var paperId = summary.paper_id || script.paper_id || '';
    var pillarId = summary.pillar_id || script.pillar_id || '';
    var papers = (catalog && catalog.papers) || {};
    var pillars = (catalog && catalog.pillars) || {};
    var paper = papers[paperId] || {};
    var title = summary.title || script.title || 'Untitled episode';
    var id = summary.id || '';
    if (!id) return null;
    return {
      id: id,
      title: title,
      initial: episodeInitial(title),
      paper_id: paperId,
      paper_title: paper.title || paperId || 'Unknown paper',
      paper_href: paperPageHref(paperId),
      pillar_id: pillarId,
      pillar_name: pillars[pillarId] || pillarId || 'No pillar',
      word_count: summary.word_count || script.word_count || 0,
      created_label: formatCreated(summary.created_at || script.created_at),
      created_sort: summary.created_at || script.created_at || '',
      duration_label: formatDuration(audio.duration_seconds),
      audio_href: audioHref(audio.file_name),
      file_name: typeof audio.file_name === 'string' ? audio.file_name : '',
      script_download_href: '/api/podcast/' + encodeURIComponent(id) + '/download',
      studio_href: '/podcast/?studio=1&script=' + encodeURIComponent(id),
    };
  }

  function boot() {
    var catalogNode = document.getElementById('listen-catalog');
    var catalog = { papers: {}, pillars: {} };
    if (catalogNode) {
      try { catalog = JSON.parse(catalogNode.textContent || '') || catalog; } catch (e) { /* keep empty */ }
    }

    var generateForm = document.getElementById('generate-form');
    var pillarSelect = document.getElementById('pillar-select');
    var paperSelect = document.getElementById('paper-select');
    var paperField = document.getElementById('paper-field');
    var paperError = document.getElementById('paper-error');
    var generateBtn = document.getElementById('generate-btn');
    var statusEl = document.getElementById('status');
    var progressBar = document.getElementById('progress-bar');
    var scriptStages = document.getElementById('script-stages');
    var cancelScriptBtn = document.getElementById('cancel-script-btn');
    var scriptArea = document.getElementById('script-area');
    var scriptTitle = document.getElementById('script-title');
    var wordCount = document.getElementById('word-count');
    var keyPoints = document.getElementById('key-points');
    var keyPointsList = document.getElementById('key-points-list');
    var scriptContent = document.getElementById('script-content');
    var factcheckPanel = document.getElementById('factcheck-panel');
    var factcheckStatus = document.getElementById('factcheck-status');
    var factcheckSummaryEl = document.getElementById('factcheck-summary');
    var factcheckSummaryText = document.getElementById('factcheck-summary-text');
    var factcheckMeta = document.getElementById('factcheck-meta');
    var factcheckAlert = document.getElementById('factcheck-alert');
    var factcheckBtn = document.getElementById('factcheck-btn');
    var supportedControl = document.getElementById('factcheck-supported-control');
    var showSupported = document.getElementById('factcheck-show-supported');
    var factcheckIssues = document.getElementById('factcheck-issues');
    var factcheckIssuesList = document.getElementById('factcheck-issues-list');
    var factcheckAudioWarning = document.getElementById('factcheck-audio-warning');
    var downloadBtn = document.getElementById('download-btn');
    var warningsEl = document.getElementById('script-warnings');
    var settingsEl = document.getElementById('script-settings');
    var settingsTextEl = document.getElementById('script-settings-text');
    var ttsStatusMessage = document.getElementById('tts-status-message');
    var ttsStatusChip = document.getElementById('tts-status-chip');
    var ttsStartCommand = document.getElementById('tts-start-command');
    var ttsStatusPanel = document.getElementById('tts-status-panel');
    var audioSection = document.getElementById('audio-section');
    var voiceSelect = document.getElementById('voice-select');
    var voiceReason = document.getElementById('voice-reason');
    var previewVoiceBtn = document.getElementById('preview-voice-btn');
    var generateAudioBtn = document.getElementById('generate-audio-btn');
    var cancelAudioBtn = document.getElementById('cancel-audio-btn');
    var voicePreview = document.getElementById('voice-preview');
    var audioRunStatus = document.getElementById('audio-run-status');
    var audioProgressBar = document.getElementById('audio-progress-bar');
    var audioStages = document.getElementById('audio-stages');
    var episodePlayer = document.getElementById('episode-player');
    var audioFileHint = document.getElementById('audio-file-hint');
    var audioFilePath = document.getElementById('audio-file-path');
    var playerCover = document.getElementById('player-cover');
    var playerInitial = document.getElementById('player-initial');
    var playerMeta = document.getElementById('player-meta');
    var playerToggle = document.getElementById('player-toggle');
    var playerBack = document.getElementById('player-back');
    var playerForward = document.getElementById('player-forward');
    var playerRange = document.getElementById('player-range');
    var playerTime = document.getElementById('player-time');
    var playerRate = document.getElementById('player-rate');
    var playerNoAudio = document.getElementById('player-no-audio');
    var groupsEl = document.getElementById('episode-groups');
    var libraryPillar = document.getElementById('library-pillar');
    var libraryEmpty = document.getElementById('listen-empty');
    var libraryError = document.getElementById('listen-scripts-error');
    var filterEmpty = document.getElementById('listen-filter-empty');

    var currentScriptId = null;
    var loadedBodyId = null;
    var currentAudioHref = '';
    var unsavedScript = null;
    var ttsReady = false;
    var voicesCatalog = [];
    var audioRunId = null;
    var scriptRunId = null;
    var scrubbing = false;
    var currentScriptText = '';
    var currentFactcheck = null;
    var currentScriptHash = '';
    var factcheckRunId = null;
    var factcheckPollTimer = null;
    var factcheckPollController = null;
    var factcheckLoadToken = 0;
    var factcheckButtonAction = 'start';
    var factcheckState = 'none';
    var factcheckHighCount = 0;
    var activeFactcheckPopover = null;
    var factcheckPopoverSeq = 0;
    var currentTargetLine = null;
    var requestedLineFollowed = false;
    var customValue = generateForm ? (generateForm.getAttribute('data-custom-value') || '') : '';

    var scriptEls = {
      status: statusEl,
      stages: scriptStages,
      progress: progressBar,
      cancel: cancelScriptBtn,
    };
    var audioEls = {
      status: audioRunStatus,
      stages: audioStages,
      progress: audioProgressBar,
      cancel: cancelAudioBtn,
    };

    function pathOf(src) {
      if (!src) return '';
      try {
        var url = new URL(src, global.location.href);
        return url.pathname + url.search;
      } catch (e) {
        return src.charAt(0) === '/' ? src : '';
      }
    }

    function assignSource(href) {
      if (!episodePlayer || !href) return;
      if (pathOf(episodePlayer.currentSrc || episodePlayer.src) === href) return;
      var wasPlaying = !episodePlayer.paused && !!(episodePlayer.currentSrc || episodePlayer.src);
      episodePlayer.preload = 'metadata';
      episodePlayer.src = href;
      if (wasPlaying) {
        var pending = episodePlayer.play();
        if (pending && typeof pending.catch === 'function') pending.catch(function () {});
      }
    }

    function setCover(pillarId, initial) {
      if (!playerCover) return;
      var safe = /^[a-z0-9-]+$/.test(pillarId || '') ? pillarId : '';
      playerCover.className = 'player-cover' + (safe ? ' pillar-' + safe : '');
      if (playerInitial) playerInitial.textContent = initial || episodeInitial('');
    }

    function fillMeta(meta) {
      if (!playerMeta) return;
      playerMeta.replaceChildren();
      if (meta.paper_href) {
        var link = document.createElement('a');
        link.setAttribute('href', meta.paper_href);
        link.textContent = meta.paper_title || meta.paper_id || 'Paper';
        playerMeta.appendChild(link);
      } else {
        playerMeta.appendChild(document.createTextNode(meta.paper_title || 'Unknown paper'));
      }
      var bits = [];
      if (meta.pillar_name) bits.push(meta.pillar_name);
      if (meta.duration_label) bits.push(meta.duration_label);
      if (bits.length) playerMeta.appendChild(document.createTextNode(' · ' + bits.join(' · ')));
    }

    function setDownloadHref(scriptId) {
      if (!downloadBtn) return;
      if (scriptId) downloadBtn.setAttribute('href', '/api/podcast/' + encodeURIComponent(scriptId) + '/download');
      else downloadBtn.setAttribute('href', '#');
    }

    function showSettings(options) {
      if (!settingsEl || !settingsTextEl) return;
      var choices = (options && options.choices) || {};
      var parts = [];
      Object.keys(choices).forEach(function (key) {
        var choice = choices[key];
        if (choice && choice.label) parts.push(choice.label);
      });
      settingsTextEl.textContent = parts.join(' · ');
      settingsEl.hidden = parts.length === 0;
    }

    function showWarnings(warnings, opts) {
      if (!warningsEl) return;
      warningsEl.replaceChildren();
      if (!warnings || !warnings.length) {
        warningsEl.hidden = true;
        return;
      }
      warningsEl.classList.toggle('callout-error', !!(opts && opts.unsaved));
      warningsEl.classList.toggle('callout-warn', !(opts && opts.unsaved));
      var heading = document.createElement('strong');
      heading.textContent = (opts && opts.unsaved) ? 'Generated, but NOT saved' : 'About this script';
      warningsEl.appendChild(heading);
      var ul = document.createElement('ul');
      warnings.forEach(function (w) {
        var li = document.createElement('li');
        li.textContent = w;
        ul.appendChild(li);
      });
      warningsEl.appendChild(ul);
      warningsEl.hidden = false;
    }

    function appendCueText(parent, text) {
      scriptSegments(text).forEach(function (seg) {
        if (seg.kind === 'text') {
          parent.appendChild(document.createTextNode(seg.text));
          return;
        }
        var el = document.createElement(seg.kind === 'host' ? 'strong' : 'em');
        el.className = seg.kind === 'host' ? 'cue-host' : 'cue-mark';
        el.textContent = seg.text;
        parent.appendChild(el);
      });
    }

    function closeFactcheckPopover(returnFocus) {
      if (!activeFactcheckPopover) return;
      var trigger = activeFactcheckPopover.trigger;
      trigger.setAttribute('aria-expanded', 'false');
      trigger.removeAttribute('aria-controls');
      activeFactcheckPopover.node.remove();
      activeFactcheckPopover = null;
      if (returnFocus) trigger.focus();
    }

    function appendClaimExplanation(parent, claim) {
      var presentation = claimPresentation(claim);
      var heading = document.createElement('strong');
      heading.className = 'factcheck-popover-title is-' + presentation.tone;
      heading.textContent = presentation.icon + ' ' + presentation.label;
      parent.appendChild(heading);

      if (claim.claim) {
        var stated = document.createElement('p');
        stated.className = 'factcheck-claim';
        stated.textContent = '“' + claim.claim + '”';
        parent.appendChild(stated);
      }
      if (claim.rationale) {
        var rationale = document.createElement('p');
        rationale.textContent = claim.rationale;
        parent.appendChild(rationale);
      }
      var evidence = Array.isArray(claim.evidence) ? claim.evidence : [];
      if (evidence.length) {
        evidence.forEach(function (item) {
          if (!item || !item.quote) return;
          var quote = document.createElement('blockquote');
          quote.textContent = '“' + item.quote + '”';
          if (item.chunk_index != null) {
            var cite = document.createElement('small');
            cite.textContent = 'Paper text · chunk ' + item.chunk_index;
            quote.appendChild(cite);
          }
          parent.appendChild(quote);
        });
      } else if (claim.verdict !== 'not_checkable') {
        var absent = document.createElement('p');
        absent.className = 'factcheck-evidence-missing';
        absent.textContent = 'No evidence quote was located in the stored paper text.';
        parent.appendChild(absent);
      }
      var trail = Array.isArray(claim.judge_trail) ? claim.judge_trail : [];
      if (trail.length) {
        var models = document.createElement('small');
        models.className = 'factcheck-judges';
        models.textContent = 'Checked by ' + trail.map(function (item) {
          return (item && item.model) || '';
        }).filter(Boolean).join(' → ');
        parent.appendChild(models);
      }
    }

    function openFactcheckPopover(trigger, claims, lineRow) {
      // A second click on the same mark closes it. A popover that only opens
      // is a trap for the mouse.
      if (activeFactcheckPopover && activeFactcheckPopover.trigger === trigger) {
        closeFactcheckPopover(false);
        return;
      }
      closeFactcheckPopover(false);
      var popover = document.createElement('div');
      var popoverId = 'factcheck-popover-' + (++factcheckPopoverSeq);
      popover.id = popoverId;
      popover.className = 'factcheck-popover';
      popover.setAttribute('role', 'dialog');
      popover.setAttribute('aria-label', 'Claim evidence');
      var close = document.createElement('button');
      close.type = 'button';
      close.className = 'factcheck-popover-close';
      close.setAttribute('aria-label', 'Close evidence');
      close.textContent = '×';
      close.addEventListener('click', function () { closeFactcheckPopover(true); });
      popover.appendChild(close);
      claims.forEach(function (claim, index) {
        if (index) popover.appendChild(document.createElement('hr'));
        appendClaimExplanation(popover, claim);
      });
      trigger.setAttribute('aria-expanded', 'true');
      trigger.setAttribute('aria-controls', popoverId);
      lineRow.appendChild(popover);
      activeFactcheckPopover = { node: popover, trigger: trigger };
    }

    /*
     * Put one line in view.
     *
     * The script scrolls inside its own box, and the page scrolls around it.
     * `scrollIntoView` on the row sorts out both at once and fights itself
     * when they disagree, so the two are moved separately: the box is set to
     * the line, and the page is set to the box.
     *
     * `instant` is for the deep link. A smooth scroll is a long animation, and
     * the fact check arriving mid-animation would restart the page and leave
     * the line half-way down the box.
     */
    function scrollScriptToLine(line, instant) {
      var target = document.getElementById('script-line-' + line);
      if (!target || !scriptContent) return;
      var boxRect = scriptContent.getBoundingClientRect();
      var targetRect = target.getBoundingClientRect();
      var offset = targetRect.top - boxRect.top + scriptContent.scrollTop;
      var top = Math.max(0, offset - (scriptContent.clientHeight - targetRect.height) / 2);
      if (instant || !scriptContent.scrollTo) {
        scriptContent.scrollTop = top;
      } else {
        scriptContent.scrollTo({ top: top, behavior: 'smooth' });
      }
      var pageTop = window.pageYOffset + boxRect.top -
        Math.max(0, (window.innerHeight - boxRect.height) / 2);
      if (window.scrollTo) {
        window.scrollTo({ top: pageTop, behavior: instant ? 'auto' : 'smooth' });
      }
    }

    function jumpToScriptLine(line, updateUrl) {
      var target = document.getElementById('script-line-' + line);
      if (!target) return;
      document.querySelectorAll('.script-line.is-target').forEach(function (row) {
        row.classList.remove('is-target');
      });
      target.classList.add('is-target');
      currentTargetLine = line;
      scrollScriptToLine(line, false);
      if (updateUrl && currentScriptId) {
        try {
          var url = new URL(global.location.href);
          url.searchParams.set('script', currentScriptId);
          url.searchParams.set('line', String(line));
          global.history.replaceState({}, '', url);
        } catch (e) { /* a line jump still works without history */ }
      }
    }

    /*
     * A deep link names one line: /podcast/?script=<id>&line=<n>.
     *
     * The line is scrolled to once, when the script first paints. Every later
     * repaint — the fact check arriving, the supported toggle, a re-check —
     * keeps the line highlighted but does not yank the reader back to it.
     */
    function followRequestedLine() {
      if (requestedLineFollowed) {
        markTargetLine();
        return;
      }
      var lines = scriptLines(currentScriptText);
      var value = new URLSearchParams(global.location.search).get('line');
      var line = requestedLine(value, lines.length);
      if (line == null) return;
      requestedLineFollowed = true;
      currentTargetLine = line;
      global.requestAnimationFrame(function () { markTargetLine(true); });
    }

    function markTargetLine(scroll) {
      if (currentTargetLine == null) return;
      var target = document.getElementById('script-line-' + currentTargetLine);
      if (!target) return;
      document.querySelectorAll('.script-line.is-target').forEach(function (row) {
        row.classList.remove('is-target');
      });
      target.classList.add('is-target');
      if (scroll) {
        scrollScriptToLine(currentTargetLine, true);
        return;
      }
      // A repaint rebuilds the script, so the reader's place in it is kept by
      // restoring the scroll offset. If the linked line is still off-screen —
      // the rebuild put it there — it is scrolled back into view. If it is
      // already visible, the reader is left alone.
      if (!scriptContent) return;
      var line = target.getBoundingClientRect();
      var view = scriptContent.getBoundingClientRect();
      if (line.top >= view.top && line.bottom <= view.bottom) return;
      scrollScriptToLine(currentTargetLine, true);
    }

    function paintAddressableScript(text, claims) {
      if (!scriptContent) return;
      closeFactcheckPopover(false);
      // Rebuilding the script resets its scroll offset, which would throw the
      // reader back to the top every time the fact check arrives or the
      // supported toggle is flipped.
      var keepScroll = scriptContent.scrollTop;
      scriptContent.replaceChildren();
      var lines = scriptLines(text);
      var byLine = {};
      (claims || []).forEach(function (claim) {
        var line = Number(claim && claim.line_index);
        if (!Number.isInteger(line) || line < 0 || line >= lines.length) return;
        if (!byLine[line]) byLine[line] = [];
        byLine[line].push(claim);
      });
      lines.forEach(function (textLine, lineIndex) {
        var row = document.createElement('div');
        row.className = 'script-line';
        row.id = 'script-line-' + lineIndex;
        row.setAttribute('data-line', String(lineIndex));
        var number = document.createElement('span');
        number.className = 'script-line-number';
        number.textContent = 'L' + lineIndex;
        number.setAttribute('aria-hidden', 'true');
        row.appendChild(number);
        var body = document.createElement('span');
        body.className = 'script-line-text';
        claimSegments(textLine, byLine[lineIndex] || [], !!(showSupported && showSupported.checked)).forEach(function (segment) {
          if (!segment.claims.length) {
            appendCueText(body, segment.text || '\u00a0');
            return;
          }
          var primary = strongestClaim(segment.claims) || segment.claims[0];
          var presentation = claimPresentation(primary);
          var mark = document.createElement('button');
          mark.type = 'button';
          mark.className = 'factcheck-mark is-' + presentation.tone;
          mark.setAttribute('aria-expanded', 'false');
          mark.setAttribute('aria-label', presentation.label + ': ' + segment.text);
          appendCueText(mark, segment.text);
          mark.addEventListener('focus', function () {
            openFactcheckPopover(mark, segment.claims, row);
          });
          mark.addEventListener('click', function () {
            openFactcheckPopover(mark, segment.claims, row);
          });
          body.appendChild(mark);
        });
        row.appendChild(body);
        scriptContent.appendChild(row);
      });
      scriptContent.scrollTop = keepScroll;
      followRequestedLine();
    }

    function renderFactcheckIssues(claims) {
      if (!factcheckIssues || !factcheckIssuesList) return;
      factcheckIssuesList.replaceChildren();
      var issues = issueClaims(claims);
      if (!issues.length) {
        var empty = document.createElement('p');
        empty.className = 'field-help';
        empty.textContent = 'No high, medium, or low severity flags. Opinions and supported claims are counted separately above.';
        factcheckIssuesList.appendChild(empty);
      }
      issues.forEach(function (claim) {
        var presentation = claimPresentation(claim);
        var card = document.createElement('article');
        card.className = 'factcheck-issue is-' + presentation.tone;
        var title = document.createElement('strong');
        title.textContent = presentation.icon + ' ' + presentation.label;
        card.appendChild(title);
        var line = document.createElement('span');
        line.className = 'factcheck-issue-line';
        line.textContent = 'L' + claim.line_index;
        card.appendChild(line);
        var span = document.createElement('q');
        span.textContent = claim.span_text || claim.claim || '';
        card.appendChild(span);
        if (claim.rationale) {
          var reason = document.createElement('p');
          reason.textContent = claim.rationale;
          card.appendChild(reason);
        }
        var evidence = Array.isArray(claim.evidence) ? claim.evidence : [];
        if (evidence.length && evidence[0].quote) {
          var quote = document.createElement('blockquote');
          quote.textContent = 'Paper: “' + evidence[0].quote + '”';
          card.appendChild(quote);
        } else if (claim.verdict !== 'not_checkable') {
          var noQuote = document.createElement('p');
          noQuote.className = 'factcheck-evidence-missing';
          noQuote.textContent = 'Evidence quote not located.';
          card.appendChild(noQuote);
        }
        var jump = document.createElement('button');
        jump.type = 'button';
        jump.className = 'btn btn-ghost btn-sm factcheck-jump';
        jump.textContent = 'Jump to line';
        jump.addEventListener('click', function () {
          jumpToScriptLine(Number(claim.line_index), true);
        });
        card.appendChild(jump);
        factcheckIssuesList.appendChild(card);
      });
      factcheckIssues.hidden = false;
    }

    function clearFactcheckDetails() {
      currentFactcheck = null;
      if (factcheckSummaryEl) factcheckSummaryEl.hidden = true;
      if (factcheckAlert) factcheckAlert.hidden = true;
      if (supportedControl) supportedControl.hidden = true;
      if (showSupported) showSupported.checked = false;
      if (factcheckIssues) factcheckIssues.hidden = true;
      if (factcheckIssuesList) factcheckIssuesList.replaceChildren();
      factcheckState = 'loading';
      factcheckHighCount = 0;
      updateAudioWarning();
      paintAddressableScript(currentScriptText, []);
    }

    /*
     * What the recording button should say about the fact check.
     *
     * The audio warning never blocks: the research says a warning, not a
     * gate. But it has to name the state, because "no warning" would read as
     * "nothing is wrong" for a script that was never checked at all.
     */
    var AUDIO_NOTES = {
      none: 'This script has not been fact-checked, so no claim in it has been graded. Recording is still available.',
      loading: '',
      running: 'A fact check is running. Recording is still available.',
      failed: 'The fact check failed, so no claim in this script has been graded. Recording is still available.',
      stale: 'The script changed after its fact check. Re-check it before recording audio.',
      unavailable: 'Fact-check results are unavailable, so no claim in this script has been graded. Recording is still available.',
      done: '',
    };

    function updateAudioWarning() {
      if (!factcheckAudioWarning) return;
      var note = AUDIO_NOTES[factcheckState] || '';
      if (factcheckState === 'done' && factcheckHighCount > 0) {
        note = factcheckHighCount + ' high-severity claim' +
               (factcheckHighCount === 1 ? ' needs' : 's need') +
               ' review before recording. Audio generation is still available.';
      }
      factcheckAudioWarning.textContent = note;
      factcheckAudioWarning.hidden = !note;
    }

    function showFactcheckState(message, options) {
      options = options || {};
      factcheckButtonAction = options.action || 'start';
      factcheckState = options.state || 'none';
      if (options.highCount != null) factcheckHighCount = options.highCount;
      updateAudioWarning();
      if (factcheckPanel) factcheckPanel.hidden = false;
      if (factcheckStatus) {
        factcheckStatus.textContent = message;
        factcheckStatus.className = 'factcheck-status' + (options.tone ? ' is-' + options.tone : '');
      }
      if (factcheckBtn) {
        factcheckBtn.hidden = !!options.hideButton;
        factcheckBtn.disabled = !!options.disableButton;
        factcheckBtn.textContent = options.buttonLabel || 'Check facts';
      }
      if (options.alert && factcheckAlert) {
        factcheckAlert.textContent = options.alert;
        factcheckAlert.classList.toggle('callout-error', options.tone === 'error');
        factcheckAlert.classList.toggle('callout-warn', options.tone !== 'error');
        factcheckAlert.hidden = false;
      }
    }

    function formatFactcheckMeta(check) {
      var parts = [];
      var date = formatCreated(check && check.created_at);
      if (date) parts.push('Checked ' + date);
      var cost = Number(check && check.cost_usd);
      if (isFinite(cost) && cost > 0) parts.push('$' + cost.toFixed(2));
      return parts.join(' · ');
    }

    function renderFactcheck(check, latestRun, runHistoryError) {
      var claims = Array.isArray(check.claims) ? check.claims : [];
      currentFactcheck = check;
      factcheckButtonAction = 'start';
      if (factcheckPanel) factcheckPanel.hidden = false;
      if (factcheckStatus) {
        factcheckStatus.textContent = 'Completed against the stored paper text.';
        factcheckStatus.className = 'factcheck-status';
      }
      if (factcheckBtn) {
        factcheckBtn.hidden = false;
        factcheckBtn.disabled = false;
        factcheckBtn.textContent = 'Re-check';
      }
      if (factcheckSummaryText) factcheckSummaryText.textContent = factcheckSummary(claims);
      if (factcheckMeta) factcheckMeta.textContent = formatFactcheckMeta(check);
      if (factcheckSummaryEl) factcheckSummaryEl.hidden = false;
      if (supportedControl) supportedControl.hidden = false;
      paintAddressableScript(currentScriptText, claims);
      renderFactcheckIssues(claims);

      var high = claims.filter(function (claim) {
        return claim && claim.severity === 'high' && !claim.dismissed_at;
      }).length;
      factcheckState = 'done';
      factcheckHighCount = high;
      updateAudioWarning();

      var alertText = '';
      var alertTone = 'warn';
      if (latestRun && ['failed', 'cancelled', 'interrupted'].indexOf(latestRun.status) !== -1) {
        var checkTime = Date.parse(check.created_at || '') || 0;
        var runTime = Date.parse(latestRun.created_at || '') || 0;
        if (runTime >= checkTime) {
          alertTone = latestRun.status === 'failed' ? 'error' : 'warn';
          alertText = latestRun.status === 'failed'
            ? 'The latest re-check failed: ' + (latestRun.error || 'No reason was recorded.') +
              ' The results below are from the previous completed check.'
            : 'The latest re-check was ' + latestRun.status +
              '. The results below are from the previous completed check.';
        }
      }
      if (runHistoryError) {
        // The stored check is readable but the run history is not, so this may
        // not be the latest attempt. Say so instead of implying it is.
        alertText = alertText
          ? alertText + ' ' + runHistoryError
          : 'The fact-check run history could not be read, so this may not be the latest check.';
      }
      if (alertText && factcheckAlert) {
        factcheckAlert.textContent = alertText;
        factcheckAlert.classList.toggle('callout-error', alertTone === 'error');
        factcheckAlert.classList.toggle('callout-warn', alertTone !== 'error');
        factcheckAlert.hidden = false;
      }
    }

    async function readLatestFactcheckRun(scriptId) {
      var query = '?kind=podcast_factcheck&script_id=' + encodeURIComponent(scriptId) + '&limit=1';
      var res = await fetch('/api/pipeline-runs' + query);
      var data = await res.json().catch(function () { return null; });
      if (!res.ok) {
        return { ok: false, detail: (data && data.detail) || 'Could not read fact-check run status.' };
      }
      return { ok: true, run: Array.isArray(data) && data.length ? data[0] : null };
    }

    function stopFactcheckPoll() {
      if (factcheckPollTimer) global.clearTimeout(factcheckPollTimer);
      factcheckPollTimer = null;
      if (factcheckPollController) factcheckPollController.abort();
      factcheckPollController = null;
      factcheckRunId = null;
    }

    function watchFactcheckRun(runId, scriptId) {
      stopFactcheckPoll();
      factcheckRunId = runId;
      var delay = 1000;
      async function tick() {
        if (factcheckRunId !== runId || currentScriptId !== scriptId) return;
        factcheckPollController = new AbortController();
        try {
          var res = await fetch('/api/pipeline-runs/' + encodeURIComponent(runId), {
            signal: factcheckPollController.signal,
          });
          if (!res.ok) throw new Error('HTTP ' + res.status);
          var run = await res.json();
          delay = 1000;
          showFactcheckState(
            global.RunProgress ? global.RunProgress.summarise(run) : 'Fact check is running…',
            { disableButton: true, buttonLabel: 'Checking…', state: 'running' },
          );
          if (global.RunProgress && global.RunProgress.isTerminal(run.status)) {
            stopFactcheckPoll();
            loadFactcheck(scriptId, currentScriptText);
            return;
          }
          if (['succeeded', 'failed', 'cancelled', 'interrupted'].indexOf(run.status) !== -1) {
            stopFactcheckPoll();
            loadFactcheck(scriptId, currentScriptText);
            return;
          }
        } catch (err) {
          if (err && err.name === 'AbortError') return;
          delay = Math.min(delay * 2, 15000);
          showFactcheckState(
            'Lost contact with the fact-check run — retrying. This is not a completed check.',
            { tone: 'warn', disableButton: true, buttonLabel: 'Checking…', state: 'running' },
          );
        }
        factcheckPollTimer = global.setTimeout(tick, delay);
      }
      tick();
    }

    async function loadFactcheck(scriptId, scriptText) {
      var token = ++factcheckLoadToken;
      stopFactcheckPoll();
      currentScriptText = scriptText == null ? '' : String(scriptText);
      clearFactcheckDetails();
      showFactcheckState(
        'Loading fact check…',
        { disableButton: true, buttonLabel: 'Loading…', state: 'loading' },
      );
      var checkPromise = fetch('/api/podcast/' + encodeURIComponent(scriptId) + '/factcheck').then(async function (res) {
        return { status: res.status, ok: res.ok, data: await res.json().catch(function () { return {}; }) };
      }).catch(function (err) {
        return { status: 0, ok: false, data: { detail: err.message || 'Could not load fact check.' } };
      });
      var runPromise = readLatestFactcheckRun(scriptId).catch(function (err) {
        return { ok: false, detail: err.message || 'Could not read fact-check run status.' };
      });
      var results = await Promise.all([checkPromise, runPromise]);
      if (token !== factcheckLoadToken || currentScriptId !== scriptId) return;
      var checkResult = results[0];
      var runResult = results[1];
      var currentHash = currentScriptHash;
      var latestRun = runResult.run || null;

      if (!runResult.ok && checkResult.status === 404) {
        showFactcheckState(
          'Could not determine whether a fact check is missing, running, or failed.',
          {
            tone: 'error',
            buttonLabel: 'Reload status',
            action: 'reload',
            alert: runResult.detail,
            state: 'unavailable',
          },
        );
        return;
      }
      if (latestRun && ['pending', 'running'].indexOf(latestRun.status) !== -1) {
        showFactcheckState(
          global.RunProgress ? global.RunProgress.summarise(latestRun) : 'Fact check is running…',
          { disableButton: true, buttonLabel: 'Checking…', state: 'running' },
        );
        watchFactcheckRun(latestRun.id, scriptId);
        return;
      }
      if (checkResult.status === 503) {
        showFactcheckState(
          'Fact-check results are unavailable because the database migration is missing.',
          {
            tone: 'error',
            disableButton: true,
            buttonLabel: 'Unavailable',
            alert: checkResult.data.detail,
            state: 'unavailable',
          },
        );
        return;
      }
      if (!checkResult.ok && checkResult.status !== 404) {
        showFactcheckState(
          'The fact check could not be loaded. This is not the same as having no issues.',
          {
            tone: 'error',
            buttonLabel: 'Reload status',
            action: 'reload',
            alert: checkResult.data.detail || 'Request failed.',
            state: 'unavailable',
          },
        );
        return;
      }
      if (checkResult.status === 404) {
        if (latestRun && latestRun.status === 'failed') {
          showFactcheckState(
            'The fact check failed. The script was not changed.',
            {
              tone: 'error',
              buttonLabel: 'Try again',
              alert: latestRun.error || 'No failure reason was recorded.',
              state: 'failed',
            },
          );
          return;
        }
        if (latestRun && ['cancelled', 'interrupted'].indexOf(latestRun.status) !== -1) {
          showFactcheckState(
            'The fact check was ' + latestRun.status + '. No completed result is available.',
            { tone: 'warn', buttonLabel: 'Check facts', state: 'failed' },
          );
          return;
        }
        if (latestRun && latestRun.status === 'succeeded') {
          showFactcheckState(
            'The fact-check run finished, but its saved result could not be found.',
            { tone: 'error', buttonLabel: 'Re-check', state: 'unavailable' },
          );
          return;
        }
        showFactcheckState(
          'This script has not been fact-checked. No claims have been graded yet.',
          { buttonLabel: 'Check facts', state: 'none' },
        );
        return;
      }

      var check = checkResult.data || {};
      if (isFactcheckStale(check.script_sha256, currentHash)) {
        showFactcheckState(
          'This fact check is stale because the script changed after it was checked.',
          {
            tone: 'warn',
            buttonLabel: 'Re-check',
            alert: 'Old claim highlights are hidden because their line and character anchors may no longer match.',
            state: 'stale',
          },
        );
        return;
      }
      renderFactcheck(
        check,
        latestRun,
        runResult.ok ? '' : 'The fact-check run history could not be read.',
      );
    }

    async function startFactcheck() {
      if (!currentScriptId || factcheckRunId) return;
      clearFactcheckDetails();
      showFactcheckState(
        'Starting fact check…',
        { disableButton: true, buttonLabel: 'Starting…', state: 'running' },
      );
      try {
        var res = await fetch('/api/podcast/' + encodeURIComponent(currentScriptId) + '/factcheck', {
          method: 'POST',
        });
        var data = await res.json().catch(function () { return {}; });
        if (res.status !== 202 && res.status !== 409) {
          showFactcheckState(
            res.status === 503
              ? 'The fact check cannot start because a database migration is missing.'
              : 'The fact check could not be started.',
            {
              tone: 'error',
              buttonLabel: 'Try again',
              alert: data.detail || 'Request failed.',
              state: 'failed',
            },
          );
          return;
        }
        if (!data.run_id) throw new Error('The server did not return a fact-check run id.');
        showFactcheckState(
          'Fact check is starting…',
          { disableButton: true, buttonLabel: 'Checking…', state: 'running' },
        );
        watchFactcheckRun(data.run_id, currentScriptId);
      } catch (err) {
        showFactcheckState(
          'The fact check could not be started.',
          {
            tone: 'error',
            buttonLabel: 'Try again',
            alert: err.message || 'Request failed.',
            state: 'failed',
          },
        );
      }
    }

    function showStoredAudio(meta) {
      var href = meta && meta.file_name ? audioHref(meta.file_name) : null;
      currentAudioHref = href || '';
      if (!href) {
        if (playerNoAudio) playerNoAudio.hidden = false;
        if (playerToggle) playerToggle.disabled = true;
        if (audioFileHint) audioFileHint.hidden = true;
        if (episodePlayer) {
          episodePlayer.removeAttribute('src');
          try { episodePlayer.load(); } catch (e) { /* ignore */ }
        }
        renderPlayer();
        return;
      }
      if (playerNoAudio) playerNoAudio.hidden = true;
      assignSource(href);
      if (playerToggle) playerToggle.disabled = false;
      if (audioFilePath) audioFilePath.textContent = meta.file_name;
      if (audioFileHint) audioFileHint.hidden = false;
      renderPlayer();
    }

    function renderScript(script) {
      if (scriptTitle) scriptTitle.textContent = script.title || 'Episode';
      if (wordCount) wordCount.textContent = script.word_count == null ? '0' : String(script.word_count);
      if (keyPointsList) {
        keyPointsList.replaceChildren();
        (script.key_points || []).forEach(function (point) {
          var li = document.createElement('li');
          li.textContent = point;
          keyPointsList.appendChild(li);
        });
      }
      if (keyPoints) keyPoints.hidden = !(script.key_points && script.key_points.length);
      currentScriptText = script.script == null ? '' : String(script.script);
      // The hash of the text the fact-check engine hashed, so the view can say
      // whether a stored check still describes this script.
      currentScriptHash = script.script_sha256 ? String(script.script_sha256) : '';
      // A different script means a different line numbering: the deep-linked
      // line and the highlight both have to be resolved again.
      currentTargetLine = null;
      requestedLineFollowed = false;
      paintAddressableScript(currentScriptText, []);
      if (factcheckPanel) factcheckPanel.hidden = !currentScriptId;
      if (scriptArea) scriptArea.hidden = false;
      if (progressBar) progressBar.hidden = true;
      if (audioSection) audioSection.hidden = !currentScriptId;
    }

    function applyScriptView(script, scriptId) {
      var paperId = script.paper_id || '';
      var pillarId = script.pillar_id || '';
      var paper = (catalog.papers && catalog.papers[paperId]) || {};
      var audio = script.audio_metadata || {};
      setCover(pillarId, episodeInitial(script.title));
      fillMeta({
        paper_href: paperPageHref(paperId),
        paper_title: paper.title || paperId || 'Unknown paper',
        paper_id: paperId,
        pillar_name: (catalog.pillars && catalog.pillars[pillarId]) || pillarId || '',
        duration_label: formatDuration(audio.duration_seconds),
      });
      setDownloadHref(scriptId);
      renderScript(script);
      showStoredAudio(audio);
    }

    function renderPlayer() {
      if (!episodePlayer) return;
      var time = 0;
      var duration = 0;
      var paused = true;
      if (episodePlayer.readyState >= 1) {
        if (isFinite(episodePlayer.currentTime)) time = episodePlayer.currentTime;
        if (isFinite(episodePlayer.duration)) duration = episodePlayer.duration;
        paused = !!episodePlayer.paused;
      }
      if (playerRange && !scrubbing) {
        var max = duration > 0 ? duration : 0;
        playerRange.max = String(max);
        playerRange.value = String(Math.min(time, max || time));
        playerRange.disabled = !(duration > 0);
        playerRange.setAttribute('aria-valuemin', '0');
        playerRange.setAttribute('aria-valuemax', String(max));
        playerRange.setAttribute('aria-valuenow', String(Math.floor(time)));
        playerRange.setAttribute('aria-valuetext', formatValueText(time, duration));
      }
      if (playerTime) {
        playerTime.textContent = duration > 0
          ? formatClock(time) + ' / ' + formatClock(duration)
          : formatClock(time);
      }
      if (playerToggle) {
        var label = paused ? 'Play' : 'Pause';
        playerToggle.textContent = label;
        playerToggle.setAttribute('aria-label', label);
      }
      var playingSrc = pathOf(episodePlayer.currentSrc || episodePlayer.src);
      document.querySelectorAll('.episode-play').forEach(function (button) {
        var same = button.getAttribute('data-src') === playingSrc && playingSrc && !paused;
        button.textContent = same ? 'Pause' : 'Play';
        button.setAttribute('aria-label', same ? 'Pause' : 'Play');
      });
    }

    function seekPlayer(delta) {
      if (!episodePlayer) return;
      var duration = episodePlayer.readyState >= 1 ? episodePlayer.duration : NaN;
      var base = episodePlayer.readyState >= 1 && isFinite(episodePlayer.currentTime)
        ? episodePlayer.currentTime
        : Number(episodePlayer.dataset.pending || 0);
      var next = base + delta;
      if (next < 0) next = 0;
      if (isFinite(duration) && duration > 0 && next > duration) next = duration;
      if (episodePlayer.readyState >= 1) {
        try { episodePlayer.currentTime = next; } catch (e) { /* ignore */ }
      } else {
        episodePlayer.dataset.pending = String(next);
      }
      renderPlayer();
    }

    function selectedVoiceMeta() {
      if (!voiceSelect) return null;
      var path = voiceSelect.value;
      for (var i = 0; i < voicesCatalog.length; i++) {
        if (voicesCatalog[i].relative_path === path) return voicesCatalog[i];
      }
      return null;
    }

    function syncAudioControls() {
      var voice = selectedVoiceMeta();
      if (voiceReason) voiceReason.textContent = voice && voice.reason ? voice.reason : '';
      var voiceOk = !!(voice && voice.usability !== 'unusable');
      if (previewVoiceBtn) previewVoiceBtn.disabled = !voiceOk;
      if (generateAudioBtn) {
        generateAudioBtn.disabled = !(ttsReady && currentScriptId && voiceOk && !audioRunId);
      }
    }

    function refreshTtsStatus() {
      return fetch('/api/podcast/tts/status').then(function (res) {
        return res.json();
      }).then(function (data) {
        ttsReady = !!data.ready;
        if (ttsStatusMessage) ttsStatusMessage.textContent = data.message || '';
        if (ttsStartCommand) {
          if (data.start_command && !ttsReady) {
            ttsStartCommand.textContent = data.start_command;
            ttsStartCommand.hidden = false;
          } else {
            ttsStartCommand.hidden = true;
          }
        }
        if (ttsStatusPanel) {
          ttsStatusPanel.classList.toggle('is-ready', ttsReady);
          ttsStatusPanel.classList.toggle('is-down', !ttsReady);
        }
        if (ttsStatusChip) {
          ttsStatusChip.textContent = ttsReady ? 'Ready' : 'Not running';
          ttsStatusChip.className = 'chip ' + (ttsReady ? 'chip-ok' : 'chip-danger');
        }
        syncAudioControls();
      }).catch(function () {
        ttsReady = false;
        if (ttsStatusMessage) ttsStatusMessage.textContent = 'Could not reach the speech engine status endpoint.';
        if (ttsStatusPanel) {
          ttsStatusPanel.classList.remove('is-ready');
          ttsStatusPanel.classList.add('is-down');
        }
        if (ttsStatusChip) {
          ttsStatusChip.textContent = 'Not running';
          ttsStatusChip.className = 'chip chip-danger';
        }
        syncAudioControls();
      });
    }

    function loadVoices() {
      return fetch('/api/podcast/tts/voices').then(function (res) {
        return res.json();
      }).then(function (data) {
        voicesCatalog = data.voices || [];
        if (!voiceSelect) return;
        voiceSelect.replaceChildren();
        var placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = 'Select a voice…';
        voiceSelect.appendChild(placeholder);
        voicesCatalog.forEach(function (v) {
          var opt = document.createElement('option');
          opt.value = v.relative_path;
          var mark = v.usability === 'usable' ? '' : v.usability === 'warning' ? ' ⚠' : ' ✗';
          opt.textContent = v.label + mark;
          opt.disabled = v.usability === 'unusable';
          voiceSelect.appendChild(opt);
        });
        voiceSelect.disabled = voicesCatalog.length === 0;
        syncAudioControls();
      }).catch(function () {
        if (!voiceSelect) return;
        voiceSelect.replaceChildren();
        var opt = document.createElement('option');
        opt.textContent = 'Could not load voices';
        voiceSelect.appendChild(opt);
        voiceSelect.disabled = true;
      });
    }

    function collectOptions() {
      var options = {};
      document.querySelectorAll('.podcast-option').forEach(function (select) {
        options[select.dataset.optionKey] = select.value;
        if (select.value === customValue) {
          var custom = document.getElementById('option-' + select.dataset.optionKey + '-custom');
          options[select.dataset.optionKey + '_custom'] = custom ? custom.value : '';
        }
      });
      return options;
    }

    function loadScript(scriptId) {
      if (!scriptId) return Promise.resolve();
      return fetch('/api/podcast/' + encodeURIComponent(scriptId)).then(function (res) {
        return res.json().then(function (data) {
          if (!res.ok) throw new Error(data.detail || 'Failed to load script');
          var script = data.script || {};
          currentScriptId = scriptId;
          unsavedScript = null;
          var stored = script.source_material || {};
          showWarnings(stored.warnings, { unsaved: false });
          showSettings(script.options);
          applyScriptView(script, scriptId);
          loadedBodyId = scriptId;
          syncAudioControls();
          return loadFactcheck(scriptId, script.script || '');
        });
      }).catch(function (err) {
        if (statusEl) statusEl.textContent = 'Error loading script: ' + err.message;
        if (global.RunProgress) global.RunProgress.applyStatusClass(statusEl, 'run-danger');
      });
    }

    function onScriptUpdate(run) {
      if (!run || run.kind !== 'podcast_script' || !global.RunProgress) return;
      global.RunProgress.applyStatusClass(statusEl, run);
    }

    function onScriptFinished(run) {
      scriptRunId = null;
      if (generateBtn) generateBtn.disabled = false;
      if (!run) {
        if (statusEl) statusEl.textContent = 'Run no longer available.';
        if (progressBar) progressBar.hidden = true;
        return;
      }
      if (global.RunProgress) global.RunProgress.applyStatusClass(statusEl, run);
      if (run.status === 'failed' || run.status === 'cancelled' || run.status === 'interrupted') return;
      var result = run.result || {};
      if (result.saved === false) {
        stopFactcheckPoll();
        factcheckLoadToken += 1;
        currentScriptId = null;
        unsavedScript = result;
        showWarnings(result.warnings || [], { unsaved: true });
        showSettings(result.options);
        setDownloadHref(null);
        setCover('', episodeInitial(result.title));
        renderScript({
          title: result.title,
          script: result.script,
          word_count: result.word_count,
          key_points: result.key_points,
        });
        showStoredAudio(null);
        if (factcheckPanel) factcheckPanel.hidden = true;
        if (factcheckIssues) factcheckIssues.hidden = true;
        if (factcheckAudioWarning) factcheckAudioWarning.hidden = true;
        return;
      }
      if (result.script_id) {
        currentScriptId = result.script_id;
        unsavedScript = null;
        var warnings = result.warnings || [];
        var partial = warnings.length > 0;
        showWarnings(warnings, { unsaved: false });
        showSettings(result.options);
        loadScript(result.script_id).then(function () {
          if (partial && global.RunProgress) global.RunProgress.applyStatusClass(statusEl, 'run-warn');
        });
      }
    }

    function onAudioFinished(run) {
      audioRunId = null;
      syncAudioControls();
      if (run && run.result && run.result.audio_url) {
        var href = pathOf(run.result.audio_url);
        if (audioHref(run.result.file_name) || (href && href.indexOf('/api/podcast/audio/') === 0)) {
          currentAudioHref = audioHref(run.result.file_name) || href;
          assignSource(currentAudioHref);
        }
        if (playerNoAudio) playerNoAudio.hidden = true;
        if (playerToggle) playerToggle.disabled = !currentAudioHref;
        if (audioFilePath) audioFilePath.textContent = run.result.file_name || run.result.audio_url;
        if (audioFileHint) audioFileHint.hidden = false;
        if (audioRunStatus) audioRunStatus.textContent = 'Audio ready — press play above.';
        renderPlayer();
      } else if (run && run.status === 'failed' && audioRunStatus) {
        audioRunStatus.textContent = run.error || 'Audio generation failed.';
      }
    }

    function onAudioUpdate(run) {
      if (run && run.result && run.result.audio_url && global.RunProgress && global.RunProgress.isTerminal(run.status)) {
        onAudioFinished(run);
      }
    }

    function startScript(paperId, pillarId) {
      stopFactcheckPoll();
      factcheckLoadToken += 1;
      if (statusEl) statusEl.textContent = 'Starting podcast script generation…';
      if (global.RunProgress) global.RunProgress.applyStatusClass(statusEl, '');
      showWarnings([]);
      unsavedScript = null;
      currentScriptId = null;
      currentFactcheck = null;
      if (factcheckPanel) factcheckPanel.hidden = true;
      if (factcheckIssues) factcheckIssues.hidden = true;
      if (factcheckAudioWarning) factcheckAudioWarning.hidden = true;
      if (scriptArea) scriptArea.hidden = true;
      if (scriptStages) scriptStages.replaceChildren();
      if (progressBar) progressBar.hidden = false;
      if (generateBtn) generateBtn.disabled = true;
      return fetch('/api/podcast/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          paper_id: paperId,
          pillar_id: pillarId,
          options: collectOptions(),
        }),
      }).then(function (res) {
        return res.json().then(function (data) { return { res: res, data: data }; });
      }).then(function (out) {
        if (out.res.status === 409 && global.RunProgress) {
          return global.RunProgress.findActive().then(function (active) {
            if (active && active.kind === 'podcast_script') {
              scriptRunId = active.id;
              global.RunProgress.attach(active.id, scriptEls, onScriptFinished, onScriptUpdate);
              return;
            }
            throw new Error((out.data && out.data.detail) || 'A run is already in progress');
          });
        }
        if (!out.res.ok) throw new Error(out.data.detail || 'Generation failed');
        scriptRunId = out.data.run_id;
        if (global.RunProgress) global.RunProgress.attach(out.data.run_id, scriptEls, onScriptFinished, onScriptUpdate);
      }).catch(function (err) {
        if (statusEl) statusEl.textContent = 'Error: ' + err.message;
        if (global.RunProgress) global.RunProgress.applyStatusClass(statusEl, 'run-danger');
        if (progressBar) progressBar.hidden = true;
        if (generateBtn) generateBtn.disabled = false;
      });
    }

    function startAudio() {
      if (!currentScriptId || !voiceSelect || !voiceSelect.value) return;
      if (generateAudioBtn) generateAudioBtn.disabled = true;
      if (audioFileHint) audioFileHint.hidden = true;
      if (audioProgressBar) audioProgressBar.hidden = false;
      if (audioRunStatus) audioRunStatus.textContent = 'Starting audio generation…';
      fetch('/api/podcast/tts/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ script_id: currentScriptId, voice_path: voiceSelect.value }),
      }).then(function (res) {
        return res.json().then(function (data) { return { res: res, data: data }; });
      }).then(function (out) {
        if (out.res.status === 409 && global.RunProgress) {
          return global.RunProgress.findActive().then(function (active) {
            if (active && active.kind === 'podcast_audio') {
              audioRunId = active.id;
              global.RunProgress.attach(active.id, audioEls, onAudioFinished, onAudioUpdate);
              return;
            }
            throw new Error((out.data && out.data.detail) || 'A run is already in progress');
          });
        }
        if (!out.res.ok) throw new Error(out.data.detail || 'Could not start audio generation');
        audioRunId = out.data.run_id;
        if (global.RunProgress) global.RunProgress.attach(out.data.run_id, audioEls, onAudioFinished, onAudioUpdate);
      }).catch(function (err) {
        if (audioRunStatus) audioRunStatus.textContent = 'Error: ' + err.message;
        if (generateAudioBtn) generateAudioBtn.disabled = false;
        if (audioProgressBar) audioProgressBar.hidden = true;
      });
    }

    function appendText(parent, text) {
      parent.appendChild(document.createTextNode(text));
    }

    function paintRow(list, ep) {
      var li = document.createElement('li');
      li.className = 'episode-row';
      var cover = document.createElement('span');
      cover.className = 'episode-cover' + (/^[a-z0-9-]+$/.test(ep.pillar_id || '') ? ' pillar-' + ep.pillar_id : '');
      cover.setAttribute('aria-hidden', 'true');
      cover.textContent = ep.initial || episodeInitial(ep.title);
      li.appendChild(cover);

      var main = document.createElement('div');
      main.className = 'episode-main';
      var open = document.createElement('button');
      open.type = 'button';
      open.className = 'episode-open';
      open.setAttribute('data-script-id', ep.id);
      open.textContent = ep.title || 'Untitled episode';
      main.appendChild(open);
      var meta = document.createElement('p');
      meta.className = 'episode-meta';
      if (ep.paper_href) {
        var link = document.createElement('a');
        link.setAttribute('href', ep.paper_href);
        link.textContent = ep.paper_title || ep.paper_id || 'Paper';
        meta.appendChild(link);
      } else {
        appendText(meta, ep.paper_title || 'Unknown paper');
      }
      appendText(meta, ' · ' + (ep.word_count || 0) + ' words');
      if (ep.duration_label) appendText(meta, ' · ' + ep.duration_label);
      if (ep.created_label) appendText(meta, ' · ' + ep.created_label);
      main.appendChild(meta);
      li.appendChild(main);

      var actions = document.createElement('div');
      actions.className = 'episode-actions';
      if (ep.audio_href) {
        var play = document.createElement('button');
        play.type = 'button';
        play.className = 'btn btn-primary btn-sm episode-play';
        play.setAttribute('data-script-id', ep.id);
        play.setAttribute('data-src', ep.audio_href);
        play.setAttribute('data-title', ep.title || 'Episode');
        play.textContent = 'Play';
        actions.appendChild(play);
        var mp3 = document.createElement('a');
        mp3.className = 'btn btn-ghost btn-sm';
        mp3.setAttribute('href', ep.audio_href);
        mp3.setAttribute('download', '');
        mp3.textContent = 'MP3';
        actions.appendChild(mp3);
      } else if (ep.studio_href) {
        var make = document.createElement('a');
        make.className = 'btn btn-secondary btn-sm';
        make.setAttribute('href', ep.studio_href);
        make.textContent = 'Make audio';
        actions.appendChild(make);
      }
      if (ep.script_download_href) {
        var scriptLink = document.createElement('a');
        scriptLink.className = 'btn btn-ghost btn-sm';
        scriptLink.setAttribute('href', ep.script_download_href);
        scriptLink.textContent = 'Script';
        actions.appendChild(scriptLink);
      }
      li.appendChild(actions);
      list.appendChild(li);
    }

    function paintLibrary(rows) {
      if (!groupsEl) return;
      var groups = groupEpisodes(rows);
      groupsEl.replaceChildren();
      groups.forEach(function (group) {
        var section = document.createElement('section');
        section.className = 'episode-group' + (/^[a-z0-9-]+$/.test(group.pillar_id || '') ? ' pillar-' + group.pillar_id : '');
        section.setAttribute('data-pillar', group.pillar_id || '');
        var heading = document.createElement('h3');
        heading.className = 'episode-group-title';
        var mark = document.createElement('span');
        mark.className = 'episode-mark';
        mark.setAttribute('aria-hidden', 'true');
        heading.appendChild(mark);
        heading.appendChild(document.createTextNode(group.pillar_name || 'No pillar'));
        section.appendChild(heading);
        var list = document.createElement('ul');
        list.className = 'episode-list';
        group.episodes.forEach(function (ep) { paintRow(list, ep); });
        section.appendChild(list);
        groupsEl.appendChild(section);
      });
      if (libraryEmpty) libraryEmpty.hidden = rows.length > 0;
      if (libraryError && rows.length) libraryError.hidden = true;
      applyLibraryFilter();
      renderPlayer();
    }

    function applyLibraryFilter() {
      if (!groupsEl) return;
      var want = libraryPillar ? libraryPillar.value : '';
      var shown = 0;
      var total = 0;
      groupsEl.querySelectorAll('.episode-group').forEach(function (group) {
        total += 1;
        var hide = !!(want && group.getAttribute('data-pillar') !== want);
        group.hidden = hide;
        if (!hide) shown += 1;
      });
      if (filterEmpty) filterEmpty.hidden = !(want && total > 0 && shown === 0);
    }

    function refreshLibrary() {
      return fetch('/api/podcast/list?limit=20').then(function (res) {
        if (!res.ok) return null;
        return res.json();
      }).then(function (data) {
        if (!data || !Array.isArray(data.scripts)) return;
        return Promise.all(data.scripts.map(function (summary) {
          if (!summary || !summary.id) return Promise.resolve(null);
          return fetch('/api/podcast/' + encodeURIComponent(summary.id)).then(function (res) {
            if (!res.ok) return { summary: summary, detail: null };
            return res.json().then(function (detail) {
              return { summary: summary, detail: detail };
            });
          }).catch(function () {
            return { summary: summary, detail: null };
          });
        })).then(function (pairs) {
          var rows = [];
          pairs.forEach(function (pair) {
            if (!pair) return;
            var row = rowFromParts(pair.summary, pair.detail, catalog);
            if (row) rows.push(row);
          });
          paintLibrary(rows);
        });
      }).catch(function () { /* keep the server-rendered list */ });
    }

    function playEpisode(scriptId, src, title) {
      if (!src) {
        if (scriptId) loadScript(scriptId);
        return;
      }
      var current = episodePlayer ? pathOf(episodePlayer.currentSrc || episodePlayer.src) : '';
      if (episodePlayer && current === src && !episodePlayer.paused) {
        episodePlayer.pause();
        renderPlayer();
        return;
      }
      if (scriptId) currentScriptId = scriptId;
      if (title && scriptTitle) scriptTitle.textContent = title;
      currentAudioHref = src;
      assignSource(src);
      if (playerToggle) playerToggle.disabled = false;
      if (playerNoAudio) playerNoAudio.hidden = true;
      if (scriptArea) scriptArea.hidden = false;
      if (episodePlayer) {
        var pending = episodePlayer.play();
        if (pending && typeof pending.catch === 'function') pending.catch(function () {});
      }
      if (scriptId && scriptId !== loadedBodyId) loadScript(scriptId);
    }

    document.querySelectorAll('.podcast-option').forEach(function (select) {
      var custom = document.getElementById('option-' + select.dataset.optionKey + '-custom');
      if (!custom) return;
      var sync = function () {
        custom.hidden = select.value !== customValue;
        custom.required = !custom.hidden;
      };
      select.addEventListener('change', sync);
      sync();
    });

    if (pillarSelect && paperSelect) {
      pillarSelect.addEventListener('change', function () {
        var selectedPillar = pillarSelect.value;
        paperSelect.querySelectorAll('option').forEach(function (option) {
          if (option.value === '') return;
          var hide = !!(selectedPillar && option.dataset.pillar !== selectedPillar);
          option.hidden = hide;
          option.disabled = hide;
        });
        if (paperSelect.selectedOptions[0] && paperSelect.selectedOptions[0].disabled) {
          paperSelect.value = '';
        }
      });
    }

    if (generateForm) {
      generateForm.addEventListener('submit', function (event) {
        event.preventDefault();
        var paperId = paperSelect ? paperSelect.value : '';
        var pillarId = '';
        if (paperSelect && paperSelect.selectedOptions[0]) {
          pillarId = paperSelect.selectedOptions[0].dataset.pillar || '';
        }
        if (!pillarId && pillarSelect) pillarId = pillarSelect.value;
        if (!paperId) {
          if (paperError) paperError.hidden = false;
          if (paperField) paperField.classList.add('field-invalid');
          return;
        }
        if (paperError) paperError.hidden = true;
        if (paperField) paperField.classList.remove('field-invalid');
        if (!pillarId) pillarId = paperId;
        startScript(paperId, pillarId);
      });
    }

    if (paperSelect) {
      paperSelect.addEventListener('change', function () {
        if (paperSelect.value && paperError) paperError.hidden = true;
        if (paperSelect.value && paperField) paperField.classList.remove('field-invalid');
      });
    }

    if (cancelScriptBtn) {
      cancelScriptBtn.addEventListener('click', function () {
        if (!scriptRunId || !global.RunProgress) return;
        global.RunProgress.cancel(scriptRunId);
      });
    }

    if (downloadBtn) {
      downloadBtn.addEventListener('click', function (event) {
        if (currentScriptId) return;
        event.preventDefault();
        if (!unsavedScript) return;
        var header = '# ' + (unsavedScript.title || 'Untitled') + '\n\nWord Count: ' + unsavedScript.word_count + '\n'
          + 'NOT SAVED TO THE DATABASE - this is the only copy.\n\n'
          + '============================================================\n\n';
        var blob = new Blob([header + (unsavedScript.script || '')], { type: 'text/plain;charset=utf-8' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'unsaved_podcast.txt';
        a.click();
        URL.revokeObjectURL(url);
      });
    }

    if (voiceSelect) {
      voiceSelect.addEventListener('change', function () {
        if (voicePreview) voicePreview.hidden = true;
        syncAudioControls();
      });
    }
    if (factcheckBtn) {
      factcheckBtn.addEventListener('click', function () {
        if (factcheckButtonAction === 'reload') {
          loadFactcheck(currentScriptId, currentScriptText);
          return;
        }
        startFactcheck();
      });
    }
    if (showSupported) {
      showSupported.addEventListener('change', function () {
        paintAddressableScript(
          currentScriptText,
          currentFactcheck && Array.isArray(currentFactcheck.claims) ? currentFactcheck.claims : [],
        );
      });
    }
    if (previewVoiceBtn) {
      previewVoiceBtn.addEventListener('click', function () {
        if (!voiceSelect || !voiceSelect.value || !voicePreview) return;
        if (episodePlayer && !episodePlayer.paused) episodePlayer.pause();
        voicePreview.src = '/api/podcast/tts/voices/preview?path=' + encodeURIComponent(voiceSelect.value) + '&t=' + Date.now();
        voicePreview.hidden = false;
        voicePreview.load();
      });
    }
    if (generateAudioBtn) generateAudioBtn.addEventListener('click', startAudio);
    if (cancelAudioBtn) {
      cancelAudioBtn.addEventListener('click', function () {
        if (!audioRunId || !global.RunProgress) return;
        global.RunProgress.cancel(audioRunId);
      });
    }

    if (episodePlayer) {
      episodePlayer.addEventListener('play', function () {
        if (!global.NlpPlayer || typeof global.NlpPlayer.attach !== 'function') return;
        global.NlpPlayer.attach(episodePlayer, {
          script_id: currentScriptId || '',
          title: (scriptTitle && scriptTitle.textContent) || 'Episode',
          href: currentScriptId
            ? ('/podcast/?script=' + encodeURIComponent(currentScriptId))
            : '/podcast/',
        });
        renderPlayer();
      });
      ['pause', 'timeupdate', 'ended', 'ratechange', 'loadedmetadata'].forEach(function (name) {
        episodePlayer.addEventListener(name, function () {
          if (name === 'loadedmetadata' && episodePlayer.dataset.pending) {
            var target = Number(episodePlayer.dataset.pending);
            delete episodePlayer.dataset.pending;
            try { episodePlayer.currentTime = target; } catch (e) { /* ignore */ }
          }
          renderPlayer();
        });
      });
    }
    if (playerToggle) {
      playerToggle.addEventListener('click', function () {
        if (!currentAudioHref) return;
        playEpisode(currentScriptId, currentAudioHref, scriptTitle ? scriptTitle.textContent : 'Episode');
      });
    }
    if (playerBack) playerBack.addEventListener('click', function () { seekPlayer(-15); });
    if (playerForward) playerForward.addEventListener('click', function () { seekPlayer(30); });
    if (playerRange) {
      playerRange.addEventListener('pointerdown', function () { scrubbing = true; });
      playerRange.addEventListener('pointerup', function () { scrubbing = false; });
      playerRange.addEventListener('change', function () { scrubbing = false; });
      playerRange.addEventListener('input', function () {
        var next = Number(playerRange.value);
        if (episodePlayer && episodePlayer.readyState >= 1) {
          try { episodePlayer.currentTime = next; } catch (e) { /* ignore */ }
        }
        playerRange.setAttribute('aria-valuetext', formatValueText(next, episodePlayer && episodePlayer.duration));
        if (playerTime && episodePlayer && isFinite(episodePlayer.duration) && episodePlayer.duration > 0) {
          playerTime.textContent = formatClock(next) + ' / ' + formatClock(episodePlayer.duration);
        }
      });
    }
    if (playerRate) {
      playerRate.addEventListener('change', function () {
        var next = Number(playerRate.value);
        if (episodePlayer) {
          try { episodePlayer.playbackRate = next; } catch (e) { /* ignore */ }
        }
        var mini = document.getElementById('mini-player-rate');
        if (mini) mini.value = playerRate.value;
      });
    }

    if (groupsEl) {
      groupsEl.addEventListener('click', function (event) {
        var target = event.target;
        if (!target || !target.closest) return;
        var play = target.closest('.episode-play');
        if (play) {
          playEpisode(play.getAttribute('data-script-id'), play.getAttribute('data-src'), play.getAttribute('data-title'));
          return;
        }
        var open = target.closest('.episode-open');
        if (open) loadScript(open.getAttribute('data-script-id'));
      });
    }
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') closeFactcheckPopover(true);
    });
    document.addEventListener('click', function (event) {
      // A click anywhere else dismisses the evidence popover. Without this it
      // stays open over the script until Escape or the script is repainted.
      if (!activeFactcheckPopover) return;
      var node = activeFactcheckPopover.node;
      var trigger = activeFactcheckPopover.trigger;
      var target = event.target;
      if (target && (node.contains(target) || trigger.contains(target))) return;
      closeFactcheckPopover(false);
    });
    global.addEventListener('pagehide', stopFactcheckPoll);
    if (libraryPillar) libraryPillar.addEventListener('change', applyLibraryFilter);

    refreshTtsStatus();
    loadVoices();
    global.setInterval(refreshTtsStatus, 30000);

    var params = new URLSearchParams(global.location.search);
    var urlScriptId = params.get('script');
    if (urlScriptId) loadScript(urlScriptId);

    (async function reattach() {
      if (!global.RunProgress) return;
      var run = null;
      var fromUrl = params.get('run');
      var stored = null;
      try { stored = global.localStorage.getItem('nlp:activeRun'); } catch (e) { /* private */ }
      var candidate = fromUrl || stored;
      if (candidate) {
        try {
          var res = await fetch('/api/pipeline-runs/' + encodeURIComponent(candidate));
          if (res.ok) run = await res.json();
        } catch (e) { /* fall through */ }
      }
      if (!run || global.RunProgress.isTerminal(run.status)) {
        run = await global.RunProgress.findActive();
      }
      if (!run || global.RunProgress.isTerminal(run.status)) return;
      if (run.kind === 'podcast_script') {
        scriptRunId = run.id;
        if (generateBtn) generateBtn.disabled = true;
        if (progressBar) progressBar.hidden = false;
        global.RunProgress.attach(run.id, scriptEls, onScriptFinished, onScriptUpdate);
      } else if (run.kind === 'podcast_audio') {
        audioRunId = run.id;
        if (run.result && run.result.script_id) await loadScript(run.result.script_id);
        if (audioSection) audioSection.hidden = false;
        if (generateAudioBtn) generateAudioBtn.disabled = true;
        if (audioProgressBar) audioProgressBar.hidden = false;
        global.RunProgress.attach(run.id, audioEls, onAudioFinished, onAudioUpdate);
      }
    })();

    refreshLibrary();
  }

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
    else boot();
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      episodeInitial: episodeInitial,
      paperPageHref: paperPageHref,
      audioHref: audioHref,
      formatDuration: formatDuration,
      formatCreated: formatCreated,
      formatClock: formatClock,
      formatValueText: formatValueText,
      scriptSegments: scriptSegments,
      scriptLines: scriptLines,
      claimTone: claimTone,
      claimBucket: claimBucket,
      claimPresentation: claimPresentation,
      resolveSpan: resolveSpan,
      visibleClaimSpans: visibleClaimSpans,
      claimSegments: claimSegments,
      strongestClaim: strongestClaim,
      issueClaims: issueClaims,
      factcheckSummary: factcheckSummary,
      isFactcheckStale: isFactcheckStale,
      requestedLine: requestedLine,
      groupEpisodes: groupEpisodes,
      viewMode: viewMode,
      rowFromParts: rowFromParts,
    };
  }
})(typeof window !== 'undefined' ? window : globalThis);
