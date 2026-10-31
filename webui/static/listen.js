/*
 * Listen: episode library and studio.
 *
 * The library is newest-first and grouped by pillar. Playing is done by the
 * one player in player-core.js (the bar's <audio>); every transport on this
 * page is a view of it, and the episode on screen only follows the audio when
 * it is the episode the player holds.
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

  var Core = global.ListenCore ||
    (typeof require === 'function' ? require('./listen-core.js') : {});
  var formatClock = Core.formatClock;
  var formatValueText = Core.formatValueText;
  var scriptLines = Core.scriptLines;
  var resolveSpan = Core.resolveSpan;
  var visibleClaimSpans = Core.visibleClaimSpans;
  var claimSegments = Core.claimSegments;
  var claimsByLine = Core.claimsByLine;
  var groupTokens = Core.groupTokens;

  var RA = global.ReadAlong ||
    (typeof require === 'function' ? require('./readalong.js') : null);
  var PC = global.NlpPlayerCore ||
    (typeof require === 'function' ? require('./player-core.js') : null);
  // Where in the reading surface the spoken line settles: a little above the
  // middle, the way a transcript keeps what is coming in view.
  var READ_ANCHOR = 0.38;

  var Factcheck = global.Factcheck ||
    (typeof require === 'function' ? require('./factcheck.js') : {});
  var claimTone = Factcheck.claimTone;
  var claimBucket = Factcheck.claimBucket;
  var claimPresentation = Factcheck.claimPresentation;
  var strongestClaim = Factcheck.strongestClaim;
  var issueClaims = Factcheck.issueClaims;
  var factcheckSummary = Factcheck.factcheckSummary;
  var isFactcheckStale = Factcheck.isFactcheckStale;
  var canDismiss = Factcheck.canDismiss;
  var checkDismissalNote = Factcheck.checkDismissalNote;
  var DISMISSAL_NOTE_MAX = Factcheck.DISMISSAL_NOTE_MAX;

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
    var readEl = document.getElementById('readalong');
    var readSurface = document.getElementById('readalong-surface');
    var readNoteEl = document.getElementById('readalong-note');
    var readNowBtn = document.getElementById('readalong-now');
    var readToggleBtn = document.getElementById('readalong-toggle');
    var readBackBtn = document.getElementById('readalong-back');
    var readForwardBtn = document.getElementById('readalong-forward');
    var readTimeEl = document.getElementById('readalong-time');
    var viewSwitch = document.getElementById('script-view-switch');
    var viewReadBtn = document.getElementById('view-readalong');
    var viewScriptBtn = document.getElementById('view-script');
    var downloadBtn = document.getElementById('download-btn');
    var warningsEl = document.getElementById('script-warnings');
    var settingsEl = document.getElementById('script-settings');
    var settingsTextEl = document.getElementById('script-settings-text');
    var requestVoiceSelect = document.getElementById('request-voice-select');
    var requestVoiceReason = document.getElementById('request-voice-reason');
    var requestPreviewVoiceBtn = document.getElementById('request-preview-voice-btn');
    var requestVoicePreview = document.getElementById('request-voice-preview');
    var requestVoiceError = document.getElementById('request-voice-error');
    var autoAudioCheckbox = document.getElementById('auto-audio');
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
    // Episode playback uses the shared bar controller. The voice preview audio
    // elements remain independent and never replace the episode's track.
    var episodePlayer = document.getElementById('mini-player-audio');
    var controller = global.NlpPlayer && typeof global.NlpPlayer.get === 'function'
      ? global.NlpPlayer.get() : null;
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
    var idleDuration = 0;
    var landing = null;
    var groupsEl = document.getElementById('episode-groups');
    var libraryPillar = document.getElementById('library-pillar');
    var libraryEmpty = document.getElementById('listen-empty');
    var libraryError = document.getElementById('listen-scripts-error');
    var filterEmpty = document.getElementById('listen-filter-empty');

    var currentTimeline = null;
    var readModel = null;
    var readLineEls = {};
    var readWordEls = {};
    var readActive = { line: -1, word: -1 };
    var readFollowing = true;
    var readRaf = 0;
    var readProgrammaticUntil = 0;
    var lastClaims = [];
    var scriptView = 'readalong';
    try {
      // A deep link to one line is a request for the line-numbered script.
      if (global.localStorage.getItem('nlp:scriptView') === 'script') scriptView = 'script';
    } catch (e) { /* private window: the default stands */ }
    if (new URLSearchParams(global.location.search).get('line') != null) scriptView = 'script';
    // The bar's "Read along" link says it came to follow the audio.
    if (new URLSearchParams(global.location.search).get('view') === 'readalong') scriptView = 'readalong';

    var currentScriptId = null;
    var loadedBodyId = null;
    var currentAudioHref = '';
    var unsavedScript = null;
    var ttsReady = false;
    var voicesCatalog = [];
    var lastVoicePath = null;
    var episodeVoicePath = null;
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

    /* The player's snapshot, but only while it holds the episode on screen. */
    function liveSnapshot() {
      if (!controller || !currentAudioHref) return null;
      var snap = controller.snapshot();
      return snap.track && snap.track.src === currentAudioHref ? snap : null;
    }

    function episodeTitle() {
      return (scriptTitle && scriptTitle.textContent) || 'Episode';
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
      appendDismissControl(parent, claim);
    }

    /*
     * Dismiss a flag with a note, or restore it. The note is required and is
     * kept (with every restore) in podcast_claim_reviews: a dismissal is a
     * person saying the verifier was wrong, which is a calibration label. The
     * flag itself is not changed, only marked.
     */
    function appendDismissControl(parent, claim) {
      if (!canDismiss(claim)) return;
      var box = document.createElement('div');
      box.className = 'factcheck-dismiss';
      var failure = document.createElement('p');
      failure.className = 'factcheck-dismiss-error';
      failure.setAttribute('role', 'alert');
      failure.hidden = true;

      function save(dismissed, note, button) {
        var scriptId = currentScriptId;
        button.disabled = true;
        failure.hidden = true;
        fetch('/api/podcast/' + encodeURIComponent(scriptId) + '/factcheck/claims/' +
              encodeURIComponent(claim.id) + '/dismissal', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dismissed: dismissed, note: note }),
        }).then(function (res) {
          return res.json().catch(function () { return {}; }).then(function (data) {
            if (!res.ok) {
              throw new Error(typeof data.detail === 'string' ? data.detail : 'HTTP ' + res.status);
            }
            return data;
          });
        }).then(function (updated) {
          if (scriptId !== currentScriptId) return;
          claim.dismissed_at = updated.dismissed_at;
          claim.dismissed_note = updated.dismissed_note;
          refreshFactcheckClaims();
        }).catch(function (err) {
          failure.textContent = 'Could not save: ' + err.message;
          failure.hidden = false;
          button.disabled = false;
        });
      }

      if (claim.dismissed_at) {
        var kept = document.createElement('p');
        kept.className = 'factcheck-dismissed';
        kept.textContent = 'Dismissed: “' + (claim.dismissed_note || '') + '”';
        box.appendChild(kept);
        var restore = document.createElement('button');
        restore.type = 'button';
        restore.className = 'btn btn-ghost btn-sm';
        restore.textContent = 'Restore flag';
        restore.addEventListener('click', function () { save(false, '', restore); });
        box.appendChild(restore);
      } else {
        var form = document.createElement('form');
        form.className = 'factcheck-dismiss-form';
        form.hidden = true;
        var label = document.createElement('label');
        label.textContent = 'Why is this flag fine?';
        var field = document.createElement('textarea');
        field.rows = 2;
        field.maxLength = DISMISSAL_NOTE_MAX;
        field.required = true;
        label.appendChild(field);
        form.appendChild(label);
        var confirm = document.createElement('button');
        confirm.type = 'submit';
        confirm.className = 'btn btn-sm';
        confirm.textContent = 'Dismiss flag';
        var cancel = document.createElement('button');
        cancel.type = 'button';
        cancel.className = 'btn btn-ghost btn-sm';
        cancel.textContent = 'Cancel';
        form.appendChild(confirm);
        form.appendChild(cancel);
        var open = document.createElement('button');
        open.type = 'button';
        open.className = 'btn btn-ghost btn-sm';
        open.textContent = 'Dismiss…';
        open.setAttribute('aria-expanded', 'false');
        function toggle(show) {
          form.hidden = !show;
          open.hidden = show;
          open.setAttribute('aria-expanded', show ? 'true' : 'false');
          failure.hidden = true;
          if (show) field.focus(); else open.focus();
        }
        open.addEventListener('click', function () { toggle(true); });
        cancel.addEventListener('click', function () { toggle(false); });
        form.addEventListener('submit', function (event) {
          event.preventDefault();
          var checked = checkDismissalNote(field.value);
          if (!checked.ok) {
            failure.textContent = checked.error;
            failure.hidden = false;
            return;
          }
          save(true, checked.note, confirm);
        });
        box.appendChild(open);
        box.appendChild(form);
      }
      box.appendChild(failure);
      parent.appendChild(box);
    }

    // Repaint everything that shows claims after one of them changed. The
    // popover is closed because the node it hangs off is rebuilt.
    function refreshFactcheckClaims() {
      if (!currentFactcheck) return;
      var claims = Array.isArray(currentFactcheck.claims) ? currentFactcheck.claims : [];
      closeFactcheckPopover(false);
      if (factcheckSummaryText) factcheckSummaryText.textContent = factcheckSummary(claims);
      paintAddressableScript(currentScriptText, claims);
      renderFactcheckIssues(claims);
      factcheckHighCount = claims.filter(function (claim) {
        return claim && claim.severity === 'high' && !claim.dismissed_at;
      }).length;
      updateAudioWarning();
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
      if (lineRow && lineRow.classList && lineRow.classList.contains('readalong-line')) {
        addPlayFromHere(popover, Number(lineRow.getAttribute('data-line')));
      }
      trigger.setAttribute('aria-expanded', 'true');
      trigger.setAttribute('aria-controls', popoverId);
      lineRow.appendChild(popover);
      if (lineRow.classList && lineRow.classList.contains('readalong-line')) {
        // Reading the evidence is the reader's choice: let go of auto-scroll
        // so the line they are on does not move, and bring the popover into view.
        setFollowing(false);
        readProgrammaticUntil = Date.now() + 900;
        if (popover.scrollIntoView) popover.scrollIntoView({ block: 'nearest' });
      }
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

    function jumpToReadAlongLine(line) {
      var target = document.getElementById('readalong-line-' + line);
      if (!target) return false;
      document.querySelectorAll('.readalong-line.is-target').forEach(function (row) {
        row.classList.remove('is-target');
      });
      target.classList.add('is-target');
      currentTargetLine = line;
      // Looking at one line is the reader's choice, not the audio's.
      setFollowing(false);
      scrollReadTo(target, false);
      if (readSurface && readSurface.scrollIntoView && !reducedMotion()) {
        var box = readSurface.getBoundingClientRect();
        if (box.top < 0 || box.bottom > global.innerHeight) readSurface.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
      return true;
    }

    function jumpToScriptLine(line, updateUrl) {
      if (readVisible() && jumpToReadAlongLine(line)) return;
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
      var byLine = claimsByLine(claims, lines.length);
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
          mark.className = 'factcheck-mark is-' + presentation.tone + (segment.claims.every(function (c) { return c.dismissed_at; }) ? ' is-dismissed' : '');
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
      lastClaims = claims || [];
      paintReadAlong();
      followRequestedLine();
    }

    /*
     * Read along.
     *
     * The same script, set as a page to read: serif text in one column, the
     * line being spoken at full strength and the rest quieter, and a soft chip
     * on the word being said. The words and times come from the episode's
     * stored timeline (readalong.js); the audio element stays the one source
     * of time. Everything is built with createElement + textContent.
     *
     * Two things are easy to get wrong here. Auto-scroll must stop the moment
     * the reader scrolls themselves, or the page fights their thumb; so a
     * scroll the page did not start (no recent programmatic scroll) detaches
     * the view and offers "Jump to now". And a click on a word seeks, but a
     * click on a fact-check mark opens its evidence, so seeking from a flagged
     * claim is a button inside the popover.
     */
    function reducedMotion() {
      return !!(global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches);
    }

    function readAudioMatches() {
      return !!liveSnapshot();
    }

    function readVisible() {
      return !!(readEl && !readEl.hidden);
    }

    function scrollReadTo(el, instant) {
      if (!el || !readSurface) return;
      var surfaceRect = readSurface.getBoundingClientRect();
      var rect = el.getBoundingClientRect();
      var offset = rect.top - surfaceRect.top + readSurface.scrollTop;
      var top = RA.scrollTopFor(offset, rect.height, readSurface.clientHeight, readSurface.scrollHeight, READ_ANCHOR);
      readProgrammaticUntil = Date.now() + 900;
      if (instant || reducedMotion() || !readSurface.scrollTo) {
        readSurface.scrollTop = top;
      } else {
        readSurface.scrollTo({ top: top, behavior: 'smooth' });
      }
    }

    /* "Jump to now" is for a reader who has scrolled away, not one still looking at it. */
    function refreshNowButton() {
      if (!readNowBtn) return;
      if (readFollowing || !readModel || readActive.line === -1 || !readSurface) {
        readNowBtn.hidden = true;
        return;
      }
      var lineEl = readLineEls[readModel.lines[readActive.line].line];
      if (!lineEl) {
        readNowBtn.hidden = true;
        return;
      }
      var box = readSurface.getBoundingClientRect();
      var rect = lineEl.getBoundingClientRect();
      readNowBtn.hidden = rect.bottom > box.top + 16 && rect.top < box.bottom - 16;
    }

    function setFollowing(on) {
      readFollowing = on;
      refreshNowButton();
    }

    function readNote(text) {
      if (!readNoteEl) return;
      readNoteEl.textContent = text || '';
      readNoteEl.hidden = !text;
    }

    function readStatusNote() {
      if (!currentAudioHref) return '';
      var state = RA.timelineState(currentTimeline, currentScriptHash);
      if (state === 'missing') {
        return 'No read-along timing for this episode yet. It still plays, but the text will not follow the audio. ' +
          'Episodes recorded before this feature can be timed with: python -m nlp_pillars.cli timeline backfill';
      }
      if (state === 'stale') {
        return 'The script was edited after this audio was recorded, so the highlight is off. Record the episode again to bring it back.';
      }
      if (readModel && !readModel.wordLevel) {
        return 'Word timing was weak for this episode, so only the line being spoken is highlighted.';
      }
      return '';
    }

    function clearReadActive() {
      if (readActive.line !== -1 && readModel) {
        var oldLine = readLineEls[readModel.lines[readActive.line].line];
        if (oldLine) oldLine.classList.remove('is-current');
        var oldWords = readWordEls[readModel.lines[readActive.line].line];
        if (oldWords && oldWords[readActive.word]) oldWords[readActive.word].classList.remove('is-now');
      }
      readActive = { line: -1, word: -1 };
    }

    /* Move the highlight to wherever the audio is. Cheap when nothing changed. */
    function updateReadPosition(force) {
      if (!readModel || !readVisible()) return;
      var pos = { line: -1, word: -1 };
      var live = liveSnapshot();
      if (live && isFinite(live.time)) pos = RA.positionAt(readModel, live.time * 1000);
      if (!force && pos.line === readActive.line && pos.word === readActive.word) return;
      var lineChanged = pos.line !== readActive.line;
      if (readActive.line !== -1) {
        var prev = readModel.lines[readActive.line].line;
        if (lineChanged && readLineEls[prev]) readLineEls[prev].classList.remove('is-current');
        var prevWords = readWordEls[prev];
        if (prevWords && prevWords[readActive.word]) prevWords[readActive.word].classList.remove('is-now');
      }
      readActive = pos;
      if (pos.line === -1) {
        refreshNowButton();
        return;
      }
      var lineNo = readModel.lines[pos.line].line;
      var lineEl = readLineEls[lineNo];
      if (lineEl) lineEl.classList.add('is-current');
      var wordEls = readWordEls[lineNo];
      if (wordEls && pos.word !== -1 && wordEls[pos.word]) wordEls[pos.word].classList.add('is-now');
      if (lineChanged || force) {
        if (readFollowing) scrollReadTo(lineEl, false);
        else refreshNowButton();
      } else if (readFollowing && wordEls && wordEls[pos.word]) {
        // A long line wraps; keep the word in view, not only the line's top.
        var surfaceRect = readSurface.getBoundingClientRect();
        var wordRect = wordEls[pos.word].getBoundingClientRect();
        if (wordRect.bottom > surfaceRect.bottom - 24 || wordRect.top < surfaceRect.top + 24) {
          scrollReadTo(wordEls[pos.word], false);
        }
      }
    }

    function readLoop() {
      readRaf = 0;
      updateReadPosition(false);
      var live = liveSnapshot();
      if (live && live.playing && !(episodePlayer && episodePlayer.ended) && readVisible()) {
        readRaf = global.requestAnimationFrame(readLoop);
      }
    }

    function startReadLoop() {
      if (!readRaf && readVisible()) readRaf = global.requestAnimationFrame(readLoop);
    }

    /* Seek to a time and play from there; reattach auto-scroll. */
    function playFrom(seconds) {
      if (!controller || !currentAudioHref || seconds == null) return;
      controller.load({
        src: currentAudioHref,
        script_id: currentScriptId || '',
        title: episodeTitle(),
      }, { at: seconds, play: true });
      setFollowing(true);
      updateReadPosition(true);
    }

    function paintReadAlong() {
      if (!readEl || !readSurface || !RA) return;
      var keepScroll = readSurface.scrollTop;
      readProgrammaticUntil = Date.now() + 400;
      readSurface.replaceChildren();
      readLineEls = {};
      readWordEls = {};
      readActive = { line: -1, word: -1 };
      var lines = scriptLines(currentScriptText);
      readModel = RA.buildModel(currentTimeline, lines, currentScriptHash);
      readSurface.classList.toggle('is-timed', !!readModel);
      var showAll = !!(showSupported && showSupported.checked);
      var byLine = claimsByLine(lastClaims, lines.length);
      lines.forEach(function (textLine, lineNo) {
        var tokens = RA.tokenize(textLine);
        if (!tokens) {
          if (/^\s*\[(?:MUSIC|SFX|TRANSITION)\]\s*$/i.test(textLine)) {
            var rule = document.createElement('div');
            rule.className = 'readalong-break';
            rule.setAttribute('aria-hidden', 'true');
            rule.textContent = '• • •';
            readSurface.appendChild(rule);
          }
          return;
        }
        var row = document.createElement('div');
        row.className = 'readalong-line';
        row.id = 'readalong-line-' + lineNo;
        row.setAttribute('data-line', String(lineNo));
        var timed = readModel && readModel.byLine[lineNo] != null ? readModel.lines[readModel.byLine[lineNo]] : null;
        if (timed) {
          var stamp = document.createElement('button');
          stamp.type = 'button';
          stamp.className = 'readalong-seek';
          stamp.setAttribute('data-line', String(lineNo));
          stamp.setAttribute('aria-label', 'Play from ' + RA.clockLabel(timed.start));
          stamp.textContent = RA.clockLabel(timed.start);
          row.appendChild(stamp);
        }
        var body = document.createElement('p');
        body.className = 'readalong-text';
        var segments = claimSegments(textLine, byLine[lineNo] || [], showAll);
        var words = [];
        groupTokens(tokens, segments).forEach(function (group) {
          var parent = body;
          var mark = null;
          if (group.segment) {
            // A <button> is atomic inline content in Chrome: a mark that wraps
            // would drop to its own line as one block. A span that behaves as
            // a button wraps with the words around it.
            mark = document.createElement('span');
            mark.setAttribute('role', 'button');
            mark.setAttribute('tabindex', '0');
            if (group.words[0].index > 0) body.appendChild(document.createTextNode(' '));
            body.appendChild(mark);
            parent = mark;
          }
          group.words.forEach(function (entry, position) {
            if ((mark && position > 0) || (!mark && entry.index > 0)) parent.appendChild(document.createTextNode(' '));
            var span = document.createElement('span');
            span.className = 'ra-w';
            span.setAttribute('data-w', String(entry.index));
            span.textContent = entry.token.text;
            if (timed && timed.words && !timed.words[entry.index].heard) span.classList.add('is-guess');
            words.push(span);
            parent.appendChild(span);
          });
          if (!mark) return;
          var claims = group.segment.claims;
          var presentation = claimPresentation(strongestClaim(claims) || claims[0]);
          mark.className = 'factcheck-mark is-' + presentation.tone + (claims.every(function (c) { return c.dismissed_at; }) ? ' is-dismissed' : '');
          mark.setAttribute('aria-expanded', 'false');
          mark.setAttribute('aria-label', presentation.label + ': ' + mark.textContent);
          mark.addEventListener('click', function () { openFactcheckPopover(mark, claims, row); });
          mark.addEventListener('keydown', function (event) {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            openFactcheckPopover(mark, claims, row);
          });
        });
        row.appendChild(body);
        readLineEls[lineNo] = row;
        readWordEls[lineNo] = words;
        readSurface.appendChild(row);
      });
      readSurface.scrollTop = keepScroll;
      readNote(readStatusNote());
      updateReadPosition(true);
    }

    function updateViewSwitch() {
      var hasAudio = !!currentAudioHref;
      if (viewSwitch) viewSwitch.hidden = !hasAudio;
      var readAlong = hasAudio && scriptView === 'readalong';
      if (readEl) readEl.hidden = !readAlong;
      if (scriptContent) scriptContent.hidden = readAlong;
      if (viewReadBtn) viewReadBtn.setAttribute('aria-pressed', readAlong ? 'true' : 'false');
      if (viewScriptBtn) viewScriptBtn.setAttribute('aria-pressed', readAlong ? 'false' : 'true');
      if (readAlong) {
        readNote(readStatusNote());
        updateReadPosition(true);
        startReadLoop();
      }
    }

    function setScriptView(view, remember) {
      scriptView = view === 'script' ? 'script' : 'readalong';
      if (remember) {
        try { global.localStorage.setItem('nlp:scriptView', scriptView); } catch (e) { /* private window */ }
      }
      closeFactcheckPopover(false);
      updateViewSwitch();
    }

    /*
     * Land where the audio is.
     *
     * Used when the episode on screen is the one the player holds and the
     * reader came to follow it: the read-along view, the spoken line settled
     * in its box, and the box itself on screen above the bar. A different
     * episode is left alone: looking at it does not take over the one playing.
     */
    function pageScrollToRead() {
      if (!readEl || readEl.hidden) return;
      var rect = readEl.getBoundingClientRect();
      var bar = document.getElementById('mini-player');
      var reserved = bar && !bar.hidden ? Math.max(0, global.innerHeight - bar.getBoundingClientRect().top) : 0;
      var visible = global.innerHeight - reserved;
      var gap = rect.height >= visible ? 16 : (visible - rect.height) / 2;
      global.scrollTo({ top: Math.max(0, global.pageYOffset + rect.top - gap), behavior: 'auto' });
    }

    /* An episode that is not playing: open the page at its card, not at the top of the library. */
    function scrollToEpisode() {
      if (!scriptArea || scriptArea.hidden) return;
      var rect = scriptArea.getBoundingClientRect();
      global.scrollTo({ top: Math.max(0, global.pageYOffset + rect.top - 16), behavior: 'auto' });
    }

    function landAtPlayback(forceReadAlong) {
      if (!controller) return false;
      var arrival = PC.landingFor(controller.snapshot(), {
        script_id: currentScriptId,
        src: currentAudioHref,
      });
      if (!arrival.attached) {
        scrollToEpisode();
        return false;
      }
      if (forceReadAlong && scriptView !== 'readalong') setScriptView('readalong', false);
      if (!readVisible()) {
        scrollToEpisode();
        return true;
      }
      setFollowing(true);
      updateReadPosition(true);
      if (readModel && readActive.line !== -1) {
        // The word, not the line: a paragraph can be taller than the box.
        var lineNo = readModel.lines[readActive.line].line;
        var words = readWordEls[lineNo];
        var target = (words && readActive.word !== -1 && words[readActive.word]) || readLineEls[lineNo];
        if (target) scrollReadTo(target, true);
      }
      pageScrollToRead();
      return true;
    }

    /* The reader scrolling is the end of the landing: nothing below moves the page again. */
    function cancelLanding() {
      if (landing) landing.live = false;
    }

    /* The bar's title and Read along button, answered in place so the audio never restarts. */
    function openPlayingEpisode(detail) {
      var id = detail && detail.script_id;
      var src = detail && detail.src;
      var here = id ? (id === currentScriptId && id === loadedBodyId) : (!!src && src === currentAudioHref);
      if (!id && !here) return false;
      try {
        var url = new URL(global.location.href);
        if (id) url.searchParams.set('script', id);
        url.searchParams.set('view', 'readalong');
        url.searchParams.delete('line');
        global.history.replaceState({}, '', url);
      } catch (e) { /* the landing works without a shareable URL */ }
      if (here) {
        landing = { force: true, live: true };
        landAtPlayback(true);
      } else {
        loadScript(id, { land: true, force: true });
      }
      return true;
    }

    /*
     * The code panel beside the transcript (follow-along.js). It reads the same
     * timeline and the same player as the read-along; this only tells it which
     * episode is on screen.
     */
    var followPanel = null;
    function syncFollow() {
      if (!followPanel && global.FollowAlong) {
        followPanel = global.FollowAlong.mount({
          root: document.getElementById('follow-panel'),
          live: liveSnapshot,
          controller: controller,
        });
      }
      if (!followPanel) return;
      followPanel.setEpisode(currentScriptId ? {
        scriptId: currentScriptId,
        scriptText: currentScriptText,
        scriptSha: currentScriptHash,
        timeline: currentTimeline,
        hasAudio: !!currentAudioHref,
      } : null);
    }

    /* The timeline can arrive after the page did: recording finishes while it is open. */
    function refreshTimeline() {
      if (!currentScriptId) return Promise.resolve();
      var id = currentScriptId;
      return fetch('/api/podcast/' + encodeURIComponent(id)).then(function (res) {
        return res.ok ? res.json() : null;
      }).then(function (data) {
        if (!data || !data.script || id !== currentScriptId) return;
        var audio = data.script.audio_metadata || {};
        currentTimeline = audio.timeline || null;
        paintReadAlong();
        updateViewSwitch();
        syncFollow();
      }).catch(function () { /* the highlight is a nicety; the audio is unaffected */ });
    }

    function addPlayFromHere(popover, lineNo) {
      if (!readModel || readModel.byLine[lineNo] == null) return;
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'btn btn-ghost btn-sm factcheck-playfrom';
      button.textContent = 'Play from here';
      button.addEventListener('click', function () {
        playFrom(RA.seekSecondsForLine(readModel, lineNo));
      });
      popover.appendChild(button);
    }

    function bindReadAlong() {
      if (!readEl || !readSurface || !RA) return;
      readSurface.addEventListener('click', function (event) {
        var target = event.target;
        if (!target || !target.closest || !readModel) return;
        if (target.closest('.factcheck-mark') || target.closest('.factcheck-popover')) return;
        var selection = global.getSelection && global.getSelection();
        if (selection && String(selection).length > 0) return;
        var stamp = target.closest('.readalong-seek');
        var word = target.closest('.ra-w');
        var row = target.closest('.readalong-line');
        if (!row) return;
        var lineNo = Number(row.getAttribute('data-line'));
        if (stamp) {
          playFrom(RA.seekSecondsForLine(readModel, lineNo));
        } else if (word) {
          playFrom(RA.seekSecondsForWord(readModel, lineNo, Number(word.getAttribute('data-w'))));
        } else {
          playFrom(RA.seekSecondsForLine(readModel, lineNo));
        }
      });
      // A scroll the page did not start is the reader's. Let go of them.
      readSurface.addEventListener('scroll', function () {
        if (Date.now() < readProgrammaticUntil) return;
        if (readFollowing) setFollowing(false);
        else refreshNowButton();
      }, { passive: true });
      ['wheel', 'touchmove'].forEach(function (name) {
        readSurface.addEventListener(name, function () {
          if (readFollowing) setFollowing(false);
        }, { passive: true });
      });
      if (readNowBtn) {
        readNowBtn.addEventListener('click', function () {
          setFollowing(true);
          if (readActive.line !== -1 && readModel) {
            scrollReadTo(readLineEls[readModel.lines[readActive.line].line], false);
          }
        });
      }
      if (readToggleBtn && playerToggle) readToggleBtn.addEventListener('click', function () { playerToggle.click(); });
      if (readBackBtn) readBackBtn.addEventListener('click', function () { seekPlayer(-15); });
      if (readForwardBtn) readForwardBtn.addEventListener('click', function () { seekPlayer(30); });
      if (viewReadBtn) viewReadBtn.addEventListener('click', function () { setScriptView('readalong', true); });
      if (viewScriptBtn) viewScriptBtn.addEventListener('click', function () { setScriptView('script', true); });
      if (controller) {
        controller.subscribe(function (snap) {
          if (snap.playing) startReadLoop();
          updateReadPosition(false);
        });
      }
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
        card.className = 'factcheck-issue is-' + presentation.tone + (claim.dismissed_at ? ' is-dismissed' : '');
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
        appendDismissControl(card, claim);
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
      idleDuration = Number(meta && meta.duration_seconds) || 0;
      if (!href) {
        if (playerNoAudio) playerNoAudio.hidden = false;
        if (playerToggle) playerToggle.disabled = true;
        if (audioFileHint) audioFileHint.hidden = true;
        renderPlayer();
        return;
      }
      if (playerNoAudio) playerNoAudio.hidden = true;
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
      currentTimeline = audio.timeline || null;
      renderScript(script);
      showStoredAudio(audio);
      episodeVoicePath = audio.voice_path || (requestVoiceSelect && requestVoiceSelect.value) || lastVoicePath;
      if (voiceSelect && episodeVoicePath) voiceSelect.value = episodeVoicePath;
      syncAudioControls();
      setFollowing(true);
      updateViewSwitch();
      syncFollow();
    }

    /*
     * Paint this page's transports from the player. They show the live
     * position only while the player holds the episode on screen; otherwise
     * they sit at the start of it and Play loads it.
     */
    function renderPlayer() {
      var live = liveSnapshot();
      var time = live ? live.time : 0;
      var duration = live ? live.duration : idleDuration;
      var paused = live ? live.paused : true;
      if (playerRange && !scrubbing) {
        var max = duration > 0 ? duration : 0;
        playerRange.max = String(max);
        playerRange.value = String(Math.min(time, max || time));
        playerRange.disabled = !(live && live.duration > 0);
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
      var label = paused ? 'Play' : 'Pause';
      [playerToggle, readToggleBtn].forEach(function (button) {
        if (!button) return;
        button.textContent = label;
        button.setAttribute('aria-label', label);
      });
      // Skipping needs a place to skip from.
      [playerBack, playerForward, readBackBtn, readForwardBtn].forEach(function (button) {
        if (button) button.disabled = !live;
      });
      if (playerRate && controller) {
        var rate = String(controller.snapshot().rate);
        if (playerRate.value !== rate) playerRate.value = rate;
      }
      if (readTimeEl && playerTime) readTimeEl.textContent = playerTime.textContent;
      var held = controller ? controller.snapshot() : null;
      var playingSrc = held && held.track ? held.track.src : '';
      var playingNow = !!(held && held.playing);
      document.querySelectorAll('.episode-play').forEach(function (button) {
        var same = !!playingSrc && button.getAttribute('data-src') === playingSrc && playingNow;
        button.textContent = same ? 'Pause' : 'Play';
        button.setAttribute('aria-label', same ? 'Pause' : 'Play');
      });
    }

    function seekPlayer(delta) {
      if (controller && liveSnapshot()) controller.seekBy(delta);
    }

    function selectedVoiceMeta(select) {
      if (!select) return null;
      var path = select.value;
      for (var i = 0; i < voicesCatalog.length; i++) {
        if (voicesCatalog[i].relative_path === path) return voicesCatalog[i];
      }
      return null;
    }

    function syncAudioControls() {
      var voice = selectedVoiceMeta(voiceSelect);
      var requestVoice = selectedVoiceMeta(requestVoiceSelect);
      if (voiceReason) voiceReason.textContent = voice && voice.reason ? voice.reason : '';
      if (requestVoiceReason) requestVoiceReason.textContent = requestVoice && requestVoice.reason ? requestVoice.reason : '';
      var voiceOk = !!(voice && voice.usability !== 'unusable');
      var requestVoiceOk = !!(requestVoice && requestVoice.usability !== 'unusable');
      if (previewVoiceBtn) previewVoiceBtn.disabled = !voiceOk;
      if (requestPreviewVoiceBtn) requestPreviewVoiceBtn.disabled = !requestVoiceOk;
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
        lastVoicePath = data.last_voice_path || null;
        function populate(select, preferred) {
          if (!select) return;
          select.replaceChildren();
          var placeholder = document.createElement('option');
          placeholder.value = '';
          placeholder.textContent = 'Select a voice…';
          select.appendChild(placeholder);
          voicesCatalog.forEach(function (v) {
            var opt = document.createElement('option');
            opt.value = v.relative_path;
            var mark = v.usability === 'usable' ? '' : v.usability === 'warning' ? ' ⚠' : ' ✗';
            opt.textContent = v.label + mark;
            opt.disabled = v.usability === 'unusable';
            select.appendChild(opt);
          });
          select.disabled = voicesCatalog.length === 0;
          var candidate = preferred || lastVoicePath;
          var valid = voicesCatalog.find(function (v) {
            return v.relative_path === candidate && v.usability !== 'unusable';
          });
          if (!valid) valid = voicesCatalog.find(function (v) { return v.usability !== 'unusable'; });
          select.value = valid ? valid.relative_path : '';
        }
        populate(requestVoiceSelect, lastVoicePath);
        populate(voiceSelect, episodeVoicePath || lastVoicePath);
        syncAudioControls();
      }).catch(function () {
        [requestVoiceSelect, voiceSelect].forEach(function (select) {
          if (!select) return;
          select.replaceChildren();
          var opt = document.createElement('option');
          opt.textContent = 'Could not load voices';
          select.appendChild(opt);
          select.disabled = true;
        });
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

    function loadScript(scriptId, opts) {
      if (!scriptId) return Promise.resolve();
      opts = opts || {};
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
          var mine = null;
          if (opts.land) {
            mine = landing = { force: !!opts.force, live: true };
            landAtPlayback(mine.force);
          }
          return loadFactcheck(scriptId, script.script || '').then(function () {
            // The fact check changes the height of what sits above the script;
            // land again so the read-along is still where the reader was sent.
            if (!mine || landing !== mine) return;
            if (mine.live) landAtPlayback(mine.force);
            landing = null;
          });
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
        syncFollow();
        if (factcheckPanel) factcheckPanel.hidden = true;
        if (factcheckIssues) factcheckIssues.hidden = true;
        if (factcheckAudioWarning) factcheckAudioWarning.hidden = true;
        return;
      }
      if (result.script_id) {
        currentScriptId = result.script_id;
        episodeVoicePath = result.voice_path || (requestVoiceSelect && requestVoiceSelect.value) || lastVoicePath;
        if (voiceSelect && episodeVoicePath) voiceSelect.value = episodeVoicePath;
        syncAudioControls();
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
        }
        if (playerNoAudio) playerNoAudio.hidden = true;
        if (playerToggle) playerToggle.disabled = !currentAudioHref;
        if (audioFilePath) audioFilePath.textContent = run.result.file_name || run.result.audio_url;
        if (audioFileHint) audioFileHint.hidden = false;
        if (audioRunStatus) audioRunStatus.textContent = 'Audio ready — press play above.';
        renderPlayer();
        updateViewSwitch();
        refreshTimeline();
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
      syncFollow();
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
          voice_path: requestVoiceSelect ? requestVoiceSelect.value || null : null,
          auto_audio: !!(autoAudioCheckbox && autoAudioCheckbox.checked),
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
      if (scriptId) currentScriptId = scriptId;
      if (title && scriptTitle) scriptTitle.textContent = title;
      currentAudioHref = src;
      if (playerToggle) playerToggle.disabled = false;
      if (playerNoAudio) playerNoAudio.hidden = true;
      if (scriptArea) scriptArea.hidden = false;
      if (controller) {
        if (liveSnapshot()) {
          controller.toggle();
        } else {
          controller.load({ src: src, script_id: scriptId || '', title: title || episodeTitle() }, { play: true });
        }
      }
      renderPlayer();
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
        var autoAudio = !!(autoAudioCheckbox && autoAudioCheckbox.checked);
        var requestedVoice = selectedVoiceMeta(requestVoiceSelect);
        var usableVoice = !!(requestedVoice && requestedVoice.usability !== 'unusable');
        if (requestVoiceError) requestVoiceError.hidden = !autoAudio || usableVoice;
        if (autoAudio && !usableVoice) return;
        episodeVoicePath = requestVoiceSelect ? requestVoiceSelect.value || null : null;
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
    if (requestVoiceSelect) {
      requestVoiceSelect.addEventListener('change', function () {
        if (requestVoicePreview) requestVoicePreview.hidden = true;
        if (requestVoiceError) requestVoiceError.hidden = true;
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
        if (controller) controller.pause();
        voicePreview.src = '/api/podcast/tts/voices/preview?path=' + encodeURIComponent(voiceSelect.value) + '&t=' + Date.now();
        voicePreview.hidden = false;
        voicePreview.load();
      });
    }
    if (requestPreviewVoiceBtn) {
      requestPreviewVoiceBtn.addEventListener('click', function () {
        if (!requestVoiceSelect || !requestVoiceSelect.value || !requestVoicePreview) return;
        if (controller) controller.pause();
        requestVoicePreview.src = '/api/podcast/tts/voices/preview?path=' + encodeURIComponent(requestVoiceSelect.value) + '&t=' + Date.now();
        requestVoicePreview.hidden = false;
        requestVoicePreview.load();
      });
    }
    if (generateAudioBtn) generateAudioBtn.addEventListener('click', startAudio);
    if (cancelAudioBtn) {
      cancelAudioBtn.addEventListener('click', function () {
        if (!audioRunId || !global.RunProgress) return;
        global.RunProgress.cancel(audioRunId);
      });
    }

    if (controller) controller.subscribe(renderPlayer);
    bindReadAlong();
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
        var live = liveSnapshot();
        if (!live) return;
        controller.seekTo(next);
        playerRange.setAttribute('aria-valuetext', formatValueText(next, live.duration));
        if (playerTime && live.duration > 0) {
          playerTime.textContent = formatClock(next) + ' / ' + formatClock(live.duration);
        }
      });
    }
    if (playerRate) {
      playerRate.addEventListener('change', function () {
        if (controller) controller.setRate(playerRate.value);
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
        if (open) loadScript(open.getAttribute('data-script-id'), { land: true });
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
    ['wheel', 'touchstart', 'keydown', 'mousedown'].forEach(function (name) {
      global.addEventListener(name, cancelLanding, { passive: true });
    });
    document.addEventListener('nlp:player-open', function (event) {
      if (openPlayingEpisode(event.detail)) event.preventDefault();
    });
    if (urlScriptId) {
      var cameToFollow = params.get('view') === 'readalong';
      // A deep link to one line is the reader's own destination; only the bar's link overrides it.
      var inStudio = viewMode(params) === 'studio';
      loadScript(urlScriptId, {
        land: !inStudio && (cameToFollow || params.get('line') == null),
        force: cameToFollow,
      });
    }

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
