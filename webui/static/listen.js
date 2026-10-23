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

  function paintScript(container, text) {
    if (!container) return;
    container.replaceChildren();
    scriptSegments(text).forEach(function (seg) {
      if (seg.kind === 'text') {
        container.appendChild(document.createTextNode(seg.text));
        return;
      }
      var el = document.createElement(seg.kind === 'host' ? 'strong' : 'em');
      el.className = seg.kind === 'host' ? 'cue-host' : 'cue-mark';
      el.textContent = seg.text;
      container.appendChild(el);
    });
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
      paintScript(scriptContent, script.script || '');
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
      if (statusEl) statusEl.textContent = 'Starting podcast script generation…';
      if (global.RunProgress) global.RunProgress.applyStatusClass(statusEl, '');
      showWarnings([]);
      unsavedScript = null;
      currentScriptId = null;
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
      groupEpisodes: groupEpisodes,
      viewMode: viewMode,
      rowFromParts: rowFromParts,
      paintScript: paintScript,
    };
  }
})(typeof window !== 'undefined' ? window : globalThis);
