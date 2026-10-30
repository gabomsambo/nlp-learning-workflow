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
 * The bar is one view of the single player in player-core.js. It restores
 * localStorage['nlp:player'] paused at the saved place (and carries on playing
 * only when the page before it was playing a moment ago). The Listen page and
 * the Today page drive the same controller, NlpPlayer.get().
 *
 * Nodes are createElement + textContent. Nothing here concatenates markup.
 */
(function (global) {
  'use strict';

  var PC = global.NlpPlayerCore ||
    (typeof require === 'function' ? require('./player-core.js') : null);
  var PAPERS_URL = '/api/papers/?limit=200';
  var CHORD_MS = 1000;
  var PAPER_LIMIT = 8;

  var NAV_COMMANDS = [
    { id: 'today', label: 'Today', href: '/', sequence: ['g', 't'], keys: 'G then T' },
    { id: 'review', label: 'Review', href: '/quiz/', sequence: ['g', 'r'], keys: 'G then R' },
    { id: 'library', label: 'Library', href: '/papers/', sequence: ['g', 'l'], keys: 'G then L' },
    { id: 'map', label: 'Paper map', href: '/papers/map', sequence: ['g', 'm'], keys: 'G then M' },
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

  var formatClock = PC.formatClock;
  var formatValueText = PC.formatValueText;
  var clampSeek = PC.clampSeek;
  var normalizeRate = PC.normalizeRate;
  var isSafeAudioSrc = PC.isSafeAudioSrc;
  var isSafeAppHref = PC.isSafeAppHref;
  var toStoredSrc = PC.toStoredSrc;
  var parsePlayerState = PC.parsePlayerState;

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
  var player = null;
  var papersCache = null;
  var papersFlight = null;
  var paletteItems = [];
  var paletteIndex = 0;
  var scrubbing = false;

  function readStored() {
    try {
      return parsePlayerState(global.localStorage.getItem(PC.PLAYER_KEY));
    } catch (e) {
      return null;
    }
  }

  /* The one controller, built on the bar's own <audio>. Null where there is no bar. */
  function getPlayer() {
    if (player) return player;
    if (typeof document === 'undefined') return null;
    var audio = document.getElementById('mini-player-audio');
    if (!audio) return null;
    var storage = null;
    try { storage = global.localStorage; } catch (e) { storage = null; }
    player = PC.createPlayer({ audio: audio, storage: storage, base: global.location.href });
    return player;
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

  /* "1:43 / 10:54" as two spans, so a phone's one-row bar can keep the first and drop the second. */
  function paintClock(clock, time, duration) {
    if (!clock) return;
    var now = clock.querySelector('.mini-time-now');
    var total = clock.querySelector('.mini-time-total');
    if (!now || !total) {
      clock.replaceChildren();
      now = document.createElement('span');
      now.className = 'mini-time-now';
      total = document.createElement('span');
      total.className = 'mini-time-total';
      clock.appendChild(now);
      clock.appendChild(total);
    }
    now.textContent = formatClock(time);
    total.textContent = duration > 0 ? ' / ' + formatClock(duration) : '';
  }

  /* Paint the bar from the player's snapshot. Every control on it is a view of that. */
  function renderBar(snap) {
    var bar = document.getElementById('mini-player');
    if (!bar || !snap) return;
    if (!snap.track) {
      bar.hidden = true;
      syncPlayerChrome(bar);
      return;
    }
    bar.hidden = false;
    syncPlayerChrome(bar);
    var href = PC.readAlongHref(snap.track.script_id, snap.track.href);
    var title = document.getElementById('mini-player-title');
    if (title) {
      if (title.textContent !== snap.track.title) title.textContent = snap.track.title;
      title.setAttribute('href', href);
    }
    var open = document.getElementById('mini-player-open');
    if (open) open.setAttribute('href', href);
    var rate = document.getElementById('mini-player-rate');
    if (rate && rate.value !== String(snap.rate)) rate.value = String(snap.rate);
    var range = document.getElementById('mini-player-range');
    var clock = document.getElementById('mini-player-time');
    var toggle = document.getElementById('mini-player-toggle');
    if (range && !scrubbing) {
      var max = snap.duration > 0 ? snap.duration : 0;
      range.max = String(max);
      range.value = String(Math.min(snap.time, max || snap.time));
      range.disabled = !(snap.duration > 0);
      range.setAttribute('aria-valuetext', formatValueText(snap.time, snap.duration));
    }
    paintClock(clock, snap.time, snap.duration);
    if (toggle) {
      var label = snap.paused ? 'Play' : 'Pause';
      if (toggle.textContent !== label) toggle.textContent = label;
      toggle.setAttribute('aria-label', label);
    }
  }

  /*
   * The bar's title and its Read along button open the episode that is
   * playing. Pages that can show it in place (the Listen page) answer the
   * nlp:player-open event and cancel it, so the click moves the view to the
   * spot being spoken instead of reloading the page under a running audio.
   * Everywhere else the link is a normal navigation to the episode's page.
   */
  function onOpenClick(event) {
    if (event.defaultPrevented || event.button > 0) return;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    var snap = player && player.snapshot();
    if (!snap || !snap.track || typeof CustomEvent !== 'function') return;
    var proceed = document.dispatchEvent(new CustomEvent('nlp:player-open', {
      cancelable: true,
      detail: { script_id: snap.track.script_id, src: snap.track.src },
    }));
    if (!proceed) event.preventDefault();
  }

  /* Start an episode in the bar (the Today page's Play buttons). */
  function play(meta) {
    var controller = getPlayer();
    if (!controller) return null;
    meta = meta || {};
    return controller.load({
      src: meta.src,
      script_id: meta.script_id,
      title: meta.title,
      href: meta.href,
    }, { play: true });
  }

  function wirePlayer() {
    var controller = getPlayer();
    if (!controller) return;
    var back = document.getElementById('mini-player-back');
    var toggle = document.getElementById('mini-player-toggle');
    var forward = document.getElementById('mini-player-forward');
    var range = document.getElementById('mini-player-range');
    var rate = document.getElementById('mini-player-rate');
    controller.subscribe(renderBar);
    if (back) back.addEventListener('click', function () { controller.seekBy(-15); });
    if (forward) forward.addEventListener('click', function () { controller.seekBy(30); });
    if (toggle) toggle.addEventListener('click', function () { controller.toggle(); });
    ['mini-player-title', 'mini-player-open'].forEach(function (id) {
      var link = document.getElementById(id);
      if (link) link.addEventListener('click', onOpenClick);
    });
    if (range) {
      range.addEventListener('pointerdown', function () { scrubbing = true; });
      range.addEventListener('pointerup', function () { scrubbing = false; renderBar(controller.snapshot()); });
      range.addEventListener('change', function () { scrubbing = false; renderBar(controller.snapshot()); });
      range.addEventListener('input', function () {
        var next = Number(range.value);
        controller.seekTo(next);
        range.setAttribute('aria-valuetext', formatValueText(next, controller.snapshot().duration));
        paintClock(document.getElementById('mini-player-time'), next, controller.snapshot().duration);
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
    if (rate) rate.addEventListener('change', function () { controller.setRate(rate.value); });
    // Leaving the page ends this <audio>. The saved place carries to the next one,
    // and a stamp asks it to carry on playing if this one was.
    global.addEventListener('pagehide', function () { controller.suspend(true); });
    global.addEventListener('pageshow', function (event) {
      // Back/forward can hand over this page from the cache with its audio
      // stopped at an older place; the saved one is newer.
      if (event && event.persisted) controller.restore();
    });
    controller.restore();
    renderBar(controller.snapshot());
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
    get: getPlayer,
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
      createRegistry: createRegistry,
      groupShortcuts: groupShortcuts,
      matchSearch: matchSearch,
      matchHelp: matchHelp,
      resolveShortcut: resolveShortcut,
      moveIndex: moveIndex,
    };
  }
})(typeof window !== 'undefined' ? window : globalThis);
