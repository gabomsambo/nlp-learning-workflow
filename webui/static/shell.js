/*
 * Shell extras: command palette, shortcut registry, global mini player.
 *
 * Page scripts extend the ? overlay by calling NlpShortcuts.register. Pass
 * match + run and the shell dispatches the key (letter shortcuts are ignored
 * while a text field is focused). Omit match and run to list a shortcut the
 * page handles itself:
 *
 *   NlpShortcuts.register({
 *     id: 'review-reveal',
 *     group: 'Review',
 *     label: 'Reveal the answer',
 *     keys: 'Space',
 *     match: function (event) { return event.key === ' '; },
 *     run: function () {}
 *   });
 *
 * The mini player resumes from localStorage['nlp:player'] and does not
 * autoplay. The Listen page calls NlpPlayer.attach(audio, meta) when its
 * episode element starts.
 *
 * Nodes are createElement + textContent. Nothing here concatenates markup.
 */
(function (global) {
  'use strict';

  var PLAYER_KEY = 'nlp:player';
  var PAPERS_URL = '/api/papers/?limit=200';
  var CHORD_MS = 1000;
  var PAPER_LIMIT = 8;
  var RATES = [0.75, 1, 1.25, 1.5, 1.75, 2];

  var NAV_COMMANDS = [
    { id: 'today', label: 'Today', href: '/', sequence: ['g', 't'], keys: 'G then T' },
    { id: 'review', label: 'Review', href: '/quiz/', sequence: ['g', 'r'], keys: 'G then R' },
    { id: 'library', label: 'Library', href: '/papers/', sequence: ['g', 'l'], keys: 'G then L' },
    { id: 'lessons', label: 'Lessons', href: '/lessons/', sequence: ['g', 'e'], keys: 'G then E' },
    { id: 'listen', label: 'Listen', href: '/podcast/', sequence: ['g', 'i'], keys: 'G then I' },
    { id: 'add', label: 'Add papers', href: '/discovery/', sequence: ['g', 'a'], keys: 'G then A' },
    { id: 'pillars', label: 'Pillars', href: '/pillars/', sequence: ['g', 'p'], keys: 'G then P' },
    { id: 'activity', label: 'Activity', href: '/activity/', sequence: ['g', 'v'], keys: 'G then V' },
    { id: 'analytics', label: 'Analytics', href: '/analytics/', sequence: ['g', 'n'], keys: 'G then N' },
    { id: 'settings', label: 'Settings', action: 'settings', sequence: ['g', 's'], keys: 'G then S' },
  ];

  var TEXT_INPUTS = {
    text: 1, search: 1, email: 1, url: 1, password: 1, tel: 1, number: 1,
    date: 1, 'datetime-local': 1, time: 1, month: 1, week: 1,
  };

  function isTextField(el) {
    if (!el || !el.tagName) return false;
    var tag = String(el.tagName).toUpperCase();
    if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
    if (el.isContentEditable) return true;
    if (tag === 'INPUT') {
      var type = String(el.type || 'text').toLowerCase();
      return !!TEXT_INPUTS[type];
    }
    return false;
  }

  function authorText(authors) {
    if (Array.isArray(authors)) {
      return authors.filter(function (name) {
        return typeof name === 'string' && name.trim();
      }).join(', ');
    }
    if (typeof authors === 'string') return authors.trim();
    return '';
  }

  function paperHref(id) {
    if (typeof id !== 'string' || !id || id.length > 200) return null;
    return '/papers/?paper=' + encodeURIComponent(id);
  }

  function termsOf(query) {
    return String(query || '').trim().toLowerCase().split(/\s+/).filter(Boolean);
  }

  function titleScore(paper, terms) {
    var title = String((paper && paper.title) || '').toLowerCase();
    var hits = 0;
    for (var i = 0; i < terms.length; i++) {
      if (title.indexOf(terms[i]) !== -1) hits += 1;
    }
    return hits;
  }

  function filterPalette(query, papers) {
    var terms = termsOf(query);
    var commands = NAV_COMMANDS.filter(function (command) {
      if (!terms.length) return true;
      var hay = command.label.toLowerCase();
      return terms.every(function (term) { return hay.indexOf(term) !== -1; });
    });
    var matched = [];
    if (terms.length) {
      matched = (Array.isArray(papers) ? papers : []).filter(function (paper) {
        if (!paper || typeof paper !== 'object') return false;
        var hay = (String(paper.title || '') + ' ' + authorText(paper.authors)).toLowerCase();
        return terms.every(function (term) { return hay.indexOf(term) !== -1; });
      });
      matched.sort(function (a, b) {
        return titleScore(b, terms) - titleScore(a, terms);
      });
      matched = matched.slice(0, PAPER_LIMIT);
    }
    return { papers: matched, commands: commands };
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

  function clampSeek(time, delta, duration) {
    var next = (isFinite(time) ? time : 0) + delta;
    if (next < 0) next = 0;
    if (isFinite(duration) && duration > 0 && next > duration) next = duration;
    return next;
  }

  function normalizeRate(rate) {
    var n = Number(rate);
    for (var i = 0; i < RATES.length; i++) {
      if (Math.abs(RATES[i] - n) < 0.01) return RATES[i];
    }
    return 1;
  }

  function isSafeAudioSrc(src) {
    if (typeof src !== 'string' || !src) return false;
    if (src.indexOf('..') !== -1 || src.indexOf('\\') !== -1) return false;
    if (src.indexOf('\n') !== -1 || src.indexOf('\r') !== -1) return false;
    return src.indexOf('/api/podcast/audio/') === 0;
  }

  function isSafeAppHref(href) {
    if (typeof href !== 'string' || href.charAt(0) !== '/') return false;
    if (href.indexOf('//') !== -1 || href.indexOf('..') !== -1) return false;
    if (href.indexOf('\\') !== -1) return false;
    return true;
  }

  function toStoredSrc(src, base) {
    if (typeof src !== 'string' || !src) return '';
    if (src.indexOf('//') === -1 && src.charAt(0) === '/') {
      var hash = src.indexOf('#');
      return hash === -1 ? src : src.slice(0, hash);
    }
    try {
      var url = new URL(src, base || 'http://localhost');
      if (url.protocol !== 'http:' && url.protocol !== 'https:') return '';
      return url.pathname + url.search;
    } catch (e) {
      return '';
    }
  }

  function cleanTitle(title) {
    if (typeof title !== 'string') return 'Episode';
    var text = title.replace(/\s+/g, ' ').trim();
    if (!text) return 'Episode';
    return text.slice(0, 180);
  }

  function fallbackHref(scriptId) {
    if (typeof scriptId === 'string' && /^[A-Za-z0-9-]+$/.test(scriptId)) {
      return '/podcast/?script=' + encodeURIComponent(scriptId);
    }
    return '/podcast/';
  }

  function parsePlayerState(raw) {
    if (!raw) return null;
    var data;
    try {
      data = typeof raw === 'string' ? JSON.parse(raw) : raw;
    } catch (e) {
      return null;
    }
    if (!data || typeof data !== 'object') return null;
    var src = toStoredSrc(data.src, 'http://localhost');
    if (!isSafeAudioSrc(src)) return null;
    var time = Number(data.time);
    if (!isFinite(time) || time < 0) time = 0;
    if (time > 48 * 3600) time = 48 * 3600;
    var scriptId = typeof data.script_id === 'string' ? data.script_id.slice(0, 80) : '';
    return {
      script_id: scriptId,
      src: src,
      time: time,
      rate: normalizeRate(data.rate),
      title: cleanTitle(data.title),
      href: isSafeAppHref(data.href) ? data.href : fallbackHref(scriptId),
    };
  }

  function handoffTime(shellTime, shellPlaying, pageTime) {
    if (shellPlaying && isFinite(shellTime)) return shellTime;
    return isFinite(pageTime) && pageTime > 0 ? pageTime : 0;
  }

  function createRegistry() {
    var entries = [];
    return {
      register: function (entry) {
        if (!entry || typeof entry.id !== 'string' || !entry.id) {
          throw new Error('shortcut id required');
        }
        if (typeof entry.label !== 'string' || typeof entry.keys !== 'string') {
          throw new Error('shortcut label and keys required');
        }
        entries = entries.filter(function (item) { return item.id !== entry.id; });
        entries.push(entry);
        return entry;
      },
      unregister: function (id) {
        entries = entries.filter(function (item) { return item.id !== id; });
      },
      list: function () { return entries.slice(); },
      grouped: function () { return groupShortcuts(entries); },
    };
  }

  function groupShortcuts(entries) {
    var map = {};
    var order = [];
    (entries || []).forEach(function (entry) {
      var name = entry.group || 'Other';
      if (!map[name]) {
        map[name] = [];
        order.push(name);
      }
      map[name].push(entry);
    });
    order.sort(function (a, b) {
      if (a === 'Global') return -1;
      if (b === 'Global') return 1;
      if (a < b) return -1;
      if (a > b) return 1;
      return 0;
    });
    return order.map(function (name) { return { name: name, items: map[name] }; });
  }

  function matchSearch(event) {
    if (!event) return false;
    var key = event.key;
    return (key === 'k' || key === 'K') &&
      (event.metaKey || event.ctrlKey) &&
      !event.altKey &&
      !event.shiftKey;
  }

  function matchHelp(event) {
    if (!event) return false;
    return event.key === '?' && !event.metaKey && !event.ctrlKey && !event.altKey;
  }

  function startsSequence(entries, key) {
    for (var i = 0; i < entries.length; i++) {
      var sequence = entries[i].sequence;
      if (sequence && sequence[0] === key) return true;
    }
    return false;
  }

  function resolveShortcut(entries, pending, event, now, timeout) {
    var limit = timeout || CHORD_MS;
    if (!event || event.repeat) return { pending: pending, firedId: null };
    var inField = isTextField(event.target);
    var i;
    var entry;
    for (i = 0; i < entries.length; i++) {
      entry = entries[i];
      if (typeof entry.match !== 'function') continue;
      if (inField && !entry.allowInField) continue;
      if (entry.match(event)) return { pending: null, firedId: entry.id };
    }
    if (inField || event.metaKey || event.ctrlKey || event.altKey || event.dialogOpen) {
      return { pending: null, firedId: null };
    }
    var key = typeof event.key === 'string' && event.key.length === 1 ? event.key.toLowerCase() : '';
    if (!key) return { pending: pending, firedId: null };
    var armed = pending;
    if (armed && now - armed.at > limit) armed = null;
    if (armed) {
      for (i = 0; i < entries.length; i++) {
        entry = entries[i];
        if (!entry.sequence || entry.sequence.length !== 2) continue;
        if (entry.sequence[0] === armed.key && entry.sequence[1] === key) {
          return { pending: null, firedId: entry.id };
        }
      }
      if (startsSequence(entries, key)) return { pending: { key: key, at: now }, firedId: null };
      return { pending: null, firedId: null };
    }
    if (startsSequence(entries, key)) return { pending: { key: key, at: now }, firedId: null };
    return { pending: null, firedId: null };
  }

  function moveIndex(index, delta, length) {
    if (!length) return -1;
    if (index < 0) return delta < 0 ? length - 1 : 0;
    var next = index + delta;
    if (next < 0) return length - 1;
    if (next >= length) return 0;
    return next;
  }

  var shortcuts = createRegistry();
  var chord = null;
  var shellAudio = null;
  var pageAudio = null;
  var pendingSeek = null;
  var lastPersist = 0;
  var papersCache = null;
  var papersFlight = null;
  var paletteItems = [];
  var paletteIndex = 0;
  var scrubbing = false;

  function readStored() {
    try {
      return parsePlayerState(global.localStorage.getItem(PLAYER_KEY));
    } catch (e) {
      return null;
    }
  }

  function writeStored(state) {
    try {
      if (!state) global.localStorage.removeItem(PLAYER_KEY);
      else global.localStorage.setItem(PLAYER_KEY, JSON.stringify(state));
    } catch (e) {}
  }

  function activeAudio() {
    if (pageAudio && pageAudio.isConnected) return pageAudio;
    return shellAudio;
  }

  function runCommand(command) {
    if (!command) return;
    if (command.action === 'settings') {
      var button = document.querySelector('[data-open-settings]');
      if (button) button.click();
      return;
    }
    if (command.href) global.location.assign(command.href);
  }

  function openModal(dialog, focusEl) {
    if (!dialog) return;
    var others = document.querySelectorAll('dialog[open]');
    for (var i = 0; i < others.length; i++) {
      if (others[i] !== dialog) others[i].close();
    }
    var previous = document.activeElement;
    dialog.addEventListener('close', function () {
      if (previous && typeof previous.focus === 'function' && document.contains(previous)) {
        previous.focus();
      }
    }, { once: true });
    if (typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
    if (focusEl && typeof focusEl.focus === 'function') focusEl.focus();
  }

  function togglePalette() {
    var dialog = document.getElementById('command-palette');
    if (!dialog) return;
    if (dialog.open) {
      dialog.close();
      return;
    }
    openPalette();
  }

  function toggleHelp() {
    var dialog = document.getElementById('shortcuts-dialog');
    if (!dialog) return;
    if (dialog.open) {
      dialog.close();
      return;
    }
    renderHelp();
    var closeBtn = dialog.querySelector('[data-dialog-close]');
    openModal(dialog, closeBtn);
  }

  function renderHelp() {
    var body = document.getElementById('shortcuts-body');
    if (!body) return;
    body.replaceChildren();
    groupShortcuts(shortcuts.list()).forEach(function (group) {
      var section = document.createElement('section');
      var heading = document.createElement('h3');
      heading.textContent = group.name;
      section.appendChild(heading);
      var list = document.createElement('ul');
      list.className = 'shortcut-list';
      group.items.forEach(function (item) {
        var row = document.createElement('li');
        var label = document.createElement('span');
        label.textContent = item.label;
        var keys = document.createElement('kbd');
        keys.textContent = item.keys;
        row.appendChild(label);
        row.appendChild(keys);
        list.appendChild(row);
      });
      section.appendChild(list);
      body.appendChild(section);
    });
    var note = document.createElement('p');
    note.className = 'shortcut-note';
    note.textContent = 'Letter shortcuts pause while you are typing in a field. Each of these also has a button or a link.';
    body.appendChild(note);
  }

  function setPaletteStatus(text) {
    var status = document.getElementById('palette-status');
    if (status) status.textContent = text || '';
  }

  function renderPalette(query, papers, statusText) {
    var list = document.getElementById('palette-list');
    var input = document.getElementById('palette-input');
    if (!list || !input) return;
    var filtered = filterPalette(query, papers);
    list.replaceChildren();
    paletteItems = [];

    function addGroup(name, rows, kind) {
      if (!rows.length) return;
      var label = document.createElement('li');
      label.className = 'palette-group';
      label.setAttribute('role', 'presentation');
      label.textContent = name;
      list.appendChild(label);
      rows.forEach(function (row) {
        var index = paletteItems.length;
        var option = document.createElement('li');
        option.className = 'palette-option';
        option.setAttribute('role', 'option');
        option.setAttribute('id', 'palette-opt-' + index);
        option.setAttribute('data-index', String(index));
        var main = document.createElement('span');
        main.className = 'palette-option-main';
        var title = document.createElement('span');
        title.className = 'palette-option-title';
        var meta = document.createElement('span');
        meta.className = 'palette-option-meta';
        if (kind === 'paper') {
          var href = paperHref(row.id);
          if (!href) return;
          title.textContent = row.title ? String(row.title) : 'Untitled paper';
          var bits = [];
          var authors = authorText(row.authors);
          if (authors) bits.push(authors);
          if (row.year) bits.push(String(row.year));
          meta.textContent = bits.join(' · ');
          paletteItems.push({ kind: 'paper', href: href });
        } else {
          title.textContent = row.label;
          var keys = document.createElement('kbd');
          keys.textContent = row.keys || '';
          option.appendChild(title);
          option.appendChild(keys);
          paletteItems.push({
            kind: 'command',
            href: row.href || null,
            action: row.action || null,
          });
          list.appendChild(option);
          return;
        }
        main.appendChild(title);
        if (meta.textContent) main.appendChild(meta);
        option.appendChild(main);
        list.appendChild(option);
      });
    }

    if (termsOf(query).length) addGroup('Papers', filtered.papers, 'paper');
    addGroup('Go to', filtered.commands, 'command');
    if (!paletteItems.length) {
      setPaletteStatus(statusText || 'No matches');
    } else if (statusText) {
      setPaletteStatus(statusText);
    } else if (!termsOf(query).length) {
      setPaletteStatus('Type a title or an author, or jump to a page.');
    } else {
      setPaletteStatus('');
    }
    paletteIndex = paletteItems.length ? 0 : -1;
    markActive();
  }

  function markActive() {
    var input = document.getElementById('palette-input');
    var options = document.querySelectorAll('#palette-list .palette-option');
    for (var i = 0; i < options.length; i++) {
      var on = i === paletteIndex;
      options[i].classList.toggle('is-active', on);
      options[i].setAttribute('aria-selected', on ? 'true' : 'false');
    }
    if (input) {
      input.setAttribute('aria-activedescendant', paletteIndex >= 0 ? 'palette-opt-' + paletteIndex : '');
    }
    var active = paletteIndex >= 0 ? document.getElementById('palette-opt-' + paletteIndex) : null;
    if (active && active.scrollIntoView) active.scrollIntoView({ block: 'nearest' });
  }

  function activatePalette(index) {
    var item = paletteItems[index];
    if (!item) return;
    var dialog = document.getElementById('command-palette');
    if (dialog && dialog.open) dialog.close();
    if (item.action === 'settings') {
      var button = document.querySelector('[data-open-settings]');
      if (button) button.click();
      return;
    }
    if (item.href) global.location.assign(item.href);
  }

  function loadPapers() {
    if (papersCache) return Promise.resolve(papersCache);
    if (papersFlight) return papersFlight;
    papersFlight = fetch(PAPERS_URL, { headers: { Accept: 'application/json' } })
      .then(function (response) {
        if (!response.ok) throw new Error('status ' + response.status);
        return response.json();
      })
      .then(function (body) {
        papersCache = body && Array.isArray(body.papers) ? body.papers : [];
        return papersCache;
      })
      .catch(function () {
        papersFlight = null;
        return Promise.reject(new Error('Papers could not be loaded. You can still jump to a page.'));
      });
    return papersFlight;
  }

  function openPalette() {
    var dialog = document.getElementById('command-palette');
    var input = document.getElementById('palette-input');
    if (!dialog || !input) return;
    input.value = '';
    renderPalette('', papersCache || [], '');
    openModal(dialog, input);
    input.select();
    if (papersCache) return;
    setPaletteStatus('Loading papers…');
    loadPapers().then(function (rows) {
      if (!dialog.open) return;
      renderPalette(input.value, rows, '');
    }).catch(function (error) {
      if (!dialog.open) return;
      renderPalette(input.value, [], error.message);
    });
  }

  function onPaletteKey(event) {
    var key = event.key;
    if (key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      var dialog = document.getElementById('command-palette');
      if (dialog) dialog.close();
      return;
    }
    if (key === 'ArrowDown' || key === 'ArrowUp' || key === 'Home' || key === 'End') {
      event.preventDefault();
      if (!paletteItems.length) return;
      if (key === 'Home') paletteIndex = 0;
      else if (key === 'End') paletteIndex = paletteItems.length - 1;
      else paletteIndex = moveIndex(paletteIndex, key === 'ArrowDown' ? 1 : -1, paletteItems.length);
      markActive();
      return;
    }
    if (key === 'Enter') {
      event.preventDefault();
      activatePalette(paletteIndex);
    }
  }

  function wirePalette() {
    var input = document.getElementById('palette-input');
    var list = document.getElementById('palette-list');
    var dialog = document.getElementById('command-palette');
    if (!input || !list || !dialog) return;
    input.addEventListener('input', function () {
      var query = input.value;
      if (papersCache) {
        renderPalette(query, papersCache, '');
        return;
      }
      renderPalette(query, [], '');
      loadPapers().then(function (rows) {
        if (input.value !== query) return;
        renderPalette(query, rows, '');
      }).catch(function (error) {
        if (input.value !== query) return;
        renderPalette(query, [], error.message);
      });
    });
    input.addEventListener('keydown', onPaletteKey);
    list.addEventListener('mousedown', function (event) {
      var option = event.target.closest ? event.target.closest('[data-index]') : null;
      if (option) event.preventDefault();
    });
    list.addEventListener('click', function (event) {
      var option = event.target.closest ? event.target.closest('[data-index]') : null;
      if (!option) return;
      activatePalette(Number(option.getAttribute('data-index')));
    });
    dialog.querySelectorAll('[data-dialog-close]').forEach(function (button) {
      button.addEventListener('click', function () { dialog.close(); });
    });
    document.querySelectorAll('[data-open-palette]').forEach(function (button) {
      button.addEventListener('click', function () {
        var drawer = document.getElementById('nav-drawer');
        if (drawer && drawer.open) drawer.close();
        openPalette();
      });
    });
  }

  function wireHelp() {
    var dialog = document.getElementById('shortcuts-dialog');
    if (!dialog) return;
    dialog.querySelectorAll('[data-dialog-close]').forEach(function (button) {
      button.addEventListener('click', function () { dialog.close(); });
    });
    document.querySelectorAll('[data-open-shortcuts]').forEach(function (button) {
      button.addEventListener('click', function () {
        var drawer = document.getElementById('nav-drawer');
        if (drawer && drawer.open) drawer.close();
        if (dialog.open) dialog.close();
        else {
          renderHelp();
          openModal(dialog, dialog.querySelector('[data-dialog-close]'));
        }
      });
    });
  }

  function onKey(event) {
    var result = resolveShortcut(shortcuts.list(), chord, {
      key: event.key,
      metaKey: event.metaKey,
      ctrlKey: event.ctrlKey,
      altKey: event.altKey,
      shiftKey: event.shiftKey,
      repeat: event.repeat,
      target: event.target,
      dialogOpen: !!document.querySelector('dialog[open]'),
    }, Date.now(), CHORD_MS);
    chord = result.pending;
    if (!result.firedId) return;
    event.preventDefault();
    var entry = null;
    shortcuts.list().forEach(function (item) {
      if (item.id === result.firedId) entry = item;
    });
    if (entry && typeof entry.run === 'function') entry.run();
  }

  function syncPlayerChrome(bar) {
    if (!bar || bar.hidden) {
      document.documentElement.removeAttribute('data-player');
      return;
    }
    var expanded = bar.classList.contains('is-expanded');
    document.documentElement.setAttribute('data-player', expanded ? 'expanded' : 'on');
  }

  function showPlayer(state) {
    var bar = document.getElementById('mini-player');
    var title = document.getElementById('mini-player-title');
    var rate = document.getElementById('mini-player-rate');
    if (!bar || !state) return;
    bar.hidden = false;
    syncPlayerChrome(bar);
    if (title) {
      title.textContent = state.title || 'Episode';
      title.setAttribute('href', state.href || '/podcast/');
    }
    if (rate) rate.value = String(state.rate || 1);
    renderTransport();
  }

  function renderTransport() {
    var stored = readStored();
    var audio = activeAudio();
    var time = stored ? stored.time : 0;
    var duration = 0;
    var paused = true;
    if (audio && audio.readyState >= 1) {
      if (isFinite(audio.currentTime)) time = audio.currentTime;
      if (isFinite(audio.duration)) duration = audio.duration;
      paused = !!audio.paused;
    }
    var range = document.getElementById('mini-player-range');
    var clock = document.getElementById('mini-player-time');
    var toggle = document.getElementById('mini-player-toggle');
    if (range && !scrubbing) {
      var max = duration > 0 ? duration : 0;
      range.max = String(max);
      range.value = String(Math.min(time, max || time));
      range.disabled = !(duration > 0);
      range.setAttribute('aria-valuetext', formatValueText(time, duration));
    }
    if (clock) {
      clock.textContent = formatClock(time) + (duration > 0 ? ' / ' + formatClock(duration) : '');
    }
    if (toggle) {
      toggle.textContent = paused ? 'Play' : 'Pause';
      toggle.setAttribute('aria-label', paused ? 'Play' : 'Pause');
    }
  }

  function persistActive(force) {
    var stored = readStored();
    if (!stored) return;
    var audio = activeAudio();
    if (!audio || audio.readyState < 1) return;
    var src = toStoredSrc(audio.currentSrc || audio.src || '', global.location.href);
    if (src !== stored.src) return;
    var now = Date.now();
    if (!force && now - lastPersist < 1000) {
      renderTransport();
      return;
    }
    lastPersist = now;
    stored.time = isFinite(audio.currentTime) ? audio.currentTime : stored.time;
    stored.rate = normalizeRate(audio.playbackRate);
    writeStored(stored);
    renderTransport();
  }

  function ensureShellSource(state) {
    if (!shellAudio || !state) return;
    var src = toStoredSrc(shellAudio.currentSrc || shellAudio.src || '', global.location.href);
    if (src === state.src) return;
    pendingSeek = state.time;
    shellAudio.preload = 'metadata';
    shellAudio.src = state.src;
    try { shellAudio.playbackRate = state.rate; } catch (e) {}
  }

  function togglePlay() {
    var stored = readStored();
    var audio = activeAudio();
    if (!stored || !audio) return;
    if (!audio.paused) {
      audio.pause();
      persistActive(true);
      return;
    }
    var src = toStoredSrc(audio.currentSrc || audio.src || '', global.location.href);
    if (src !== stored.src) {
      pendingSeek = stored.time;
      audio.preload = 'metadata';
      audio.src = stored.src;
    }
    try { audio.playbackRate = stored.rate; } catch (e) {}
    var pending = audio.play();
    if (pending && typeof pending.catch === 'function') pending.catch(function () {});
  }

  function seekBy(delta) {
    var stored = readStored();
    var audio = activeAudio();
    if (!stored || !audio) return;
    var duration = audio.readyState >= 1 ? audio.duration : NaN;
    var base = audio.readyState >= 1 && isFinite(audio.currentTime) ? audio.currentTime : stored.time;
    var next = clampSeek(base, delta, duration);
    if (audio.readyState >= 1) {
      try { audio.currentTime = next; } catch (e) {}
    } else {
      pendingSeek = next;
    }
    stored.time = next;
    writeStored(stored);
    renderTransport();
  }

  function bindAudio(audio) {
    if (!audio || audio.__nlpPlayerBound) return;
    audio.__nlpPlayerBound = true;
    ['play', 'pause', 'seeked', 'ratechange', 'ended'].forEach(function (name) {
      audio.addEventListener(name, function () { persistActive(true); });
    });
    audio.addEventListener('timeupdate', function () { persistActive(false); });
  }

  function attach(audio, meta) {
    if (!audio) return null;
    var src = toStoredSrc(audio.currentSrc || audio.src || '', global.location.href);
    if (!isSafeAudioSrc(src)) return null;
    meta = meta || {};
    var stored = readStored();
    var shellSrc = shellAudio
      ? toStoredSrc(shellAudio.currentSrc || shellAudio.src || '', global.location.href)
      : '';
    var shellSame = !!(shellAudio && shellSrc === src);
    var time = handoffTime(
      shellSame ? shellAudio.currentTime : 0,
      shellSame && !shellAudio.paused,
      audio.currentTime
    );
    if (shellAudio && shellAudio !== audio) shellAudio.pause();
    pageAudio = audio;
    var rate = stored && stored.src === src ? stored.rate : normalizeRate(audio.playbackRate);
    try { audio.playbackRate = rate; } catch (e) {}
    if (isFinite(time) && Math.abs((audio.currentTime || 0) - time) > 0.35) {
      try { audio.currentTime = time; } catch (e2) {}
    }
    var scriptId = typeof meta.script_id === 'string' ? meta.script_id.slice(0, 80) : '';
    var state = {
      script_id: scriptId,
      src: src,
      time: isFinite(audio.currentTime) ? audio.currentTime : time,
      rate: rate,
      title: cleanTitle(meta.title),
      href: isSafeAppHref(meta.href) ? meta.href : fallbackHref(scriptId),
    };
    writeStored(state);
    showPlayer(state);
    bindAudio(audio);
    return state;
  }

  function play(meta) {
    meta = meta || {};
    var src = toStoredSrc(meta.src || '', global.location && global.location.href);
    if (!isSafeAudioSrc(src)) return null;
    if (!shellAudio && typeof document !== 'undefined') {
      shellAudio = document.getElementById('mini-player-audio');
    }
    if (pageAudio && pageAudio !== shellAudio && pageAudio.pause) pageAudio.pause();
    pageAudio = null;
    var stored = readStored();
    var same = !!(stored && stored.src === src);
    var scriptId = typeof meta.script_id === 'string' ? meta.script_id.slice(0, 80) : '';
    var state = {
      script_id: scriptId,
      src: src,
      time: same ? stored.time : 0,
      rate: same ? stored.rate : normalizeRate(meta.rate),
      title: cleanTitle(meta.title),
      href: isSafeAppHref(meta.href) ? meta.href : fallbackHref(scriptId),
    };
    writeStored(state);
    showPlayer(state);
    if (!shellAudio) return state;
    var current = toStoredSrc(shellAudio.currentSrc || shellAudio.src || '', global.location.href);
    if (current !== src) {
      pendingSeek = state.time;
      shellAudio.preload = 'auto';
      shellAudio.src = src;
      try { shellAudio.playbackRate = state.rate; } catch (e) {}
    }
    var pending = shellAudio.play();
    if (pending && typeof pending.catch === 'function') pending.catch(function () {});
    return state;
  }

  function wirePlayer() {
    shellAudio = document.getElementById('mini-player-audio');
    var back = document.getElementById('mini-player-back');
    var toggle = document.getElementById('mini-player-toggle');
    var forward = document.getElementById('mini-player-forward');
    var range = document.getElementById('mini-player-range');
    var rate = document.getElementById('mini-player-rate');
    if (!shellAudio) return;
    bindAudio(shellAudio);
    shellAudio.addEventListener('loadedmetadata', function () {
      if (pendingSeek == null) return;
      var target = pendingSeek;
      pendingSeek = null;
      try {
        if (isFinite(shellAudio.duration) && target > shellAudio.duration) target = shellAudio.duration;
        shellAudio.currentTime = target;
      } catch (e) {}
      renderTransport();
    });
    if (back) back.addEventListener('click', function () { seekBy(-15); });
    if (forward) forward.addEventListener('click', function () { seekBy(30); });
    if (toggle) toggle.addEventListener('click', togglePlay);
    if (range) {
      range.addEventListener('pointerdown', function () { scrubbing = true; });
      range.addEventListener('pointerup', function () { scrubbing = false; persistActive(true); });
      range.addEventListener('change', function () { scrubbing = false; persistActive(true); });
      range.addEventListener('input', function () {
        var audio = activeAudio();
        var next = Number(range.value);
        if (audio && audio.readyState >= 1) {
          try { audio.currentTime = next; } catch (e) {}
        }
        range.setAttribute('aria-valuetext', formatValueText(next, audio && audio.duration));
        var clock = document.getElementById('mini-player-time');
        if (clock && audio && isFinite(audio.duration) && audio.duration > 0) {
          clock.textContent = formatClock(next) + ' / ' + formatClock(audio.duration);
        }
      });
    }
    var expand = document.getElementById('mini-player-expand');
    var bar = document.getElementById('mini-player');
    if (expand && bar) {
      expand.addEventListener('click', function () {
        var open = bar.classList.toggle('is-expanded');
        expand.setAttribute('aria-expanded', open ? 'true' : 'false');
        expand.textContent = open ? 'Less' : 'More';
        expand.setAttribute('aria-label', open ? 'Hide playback controls' : 'Show playback controls');
        syncPlayerChrome(bar);
      });
    }
    if (rate) {
      rate.addEventListener('change', function () {
        var next = normalizeRate(rate.value);
        var audio = activeAudio();
        if (audio) {
          try { audio.playbackRate = next; } catch (e) {}
        }
        var stored = readStored();
        if (stored) {
          stored.rate = next;
          writeStored(stored);
        }
      });
    }
    global.addEventListener('pagehide', function () { persistActive(true); });
    var state = readStored();
    if (!state) return;
    showPlayer(state);
    if (!(pageAudio && pageAudio.isConnected)) ensureShellSource(state);
  }

  function labelSearchKeys() {
    var mac = false;
    try {
      mac = /Mac|iPhone|iPad/.test(navigator.platform || '') || /Mac/.test(navigator.userAgent || '');
    } catch (e) {}
    if (!mac) return;
    document.querySelectorAll('[data-search-kbd]').forEach(function (node) {
      node.textContent = '⌘K';
    });
  }

  function installGlobals() {
    shortcuts.register({
      id: 'search',
      group: 'Global',
      label: 'Search papers and commands',
      keys: '⌘K or Ctrl+K',
      allowInField: true,
      match: matchSearch,
      run: togglePalette,
    });
    shortcuts.register({
      id: 'help',
      group: 'Global',
      label: 'Show keyboard shortcuts',
      keys: '?',
      match: matchHelp,
      run: toggleHelp,
    });
    NAV_COMMANDS.forEach(function (command) {
      shortcuts.register({
        id: 'go-' + command.id,
        group: 'Global',
        label: command.action === 'settings' ? 'Open Settings' : 'Go to ' + command.label,
        keys: command.keys,
        sequence: command.sequence,
        run: function () { runCommand(command); },
      });
    });
  }

  function boot() {
    labelSearchKeys();
    wirePalette();
    wireHelp();
    wirePlayer();
    document.addEventListener('keydown', onKey);
  }

  installGlobals();
  global.NlpShortcuts = shortcuts;
  global.NlpPlayer = {
    attach: attach,
    play: play,
    read: readStored,
  };

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
    else boot();
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      NAV_COMMANDS: NAV_COMMANDS,
      isTextField: isTextField,
      authorText: authorText,
      paperHref: paperHref,
      filterPalette: filterPalette,
      formatClock: formatClock,
      formatValueText: formatValueText,
      clampSeek: clampSeek,
      normalizeRate: normalizeRate,
      isSafeAudioSrc: isSafeAudioSrc,
      isSafeAppHref: isSafeAppHref,
      toStoredSrc: toStoredSrc,
      parsePlayerState: parsePlayerState,
      handoffTime: handoffTime,
      createRegistry: createRegistry,
      groupShortcuts: groupShortcuts,
      matchSearch: matchSearch,
      matchHelp: matchHelp,
      resolveShortcut: resolveShortcut,
      moveIndex: moveIndex,
    };
  }
})(typeof window !== 'undefined' ? window : globalThis);
