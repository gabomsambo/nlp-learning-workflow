/*
 * Follow along with code, on the Listen page (docs/PLAYGROUND.md, "Follow-along").
 *
 * While an episode plays, the code card the narrator is talking about sits
 * beside the transcript (a drawer under the player on phones). The chain is
 * time -> script line (the episode's stored read-along timeline, readalong.js)
 * -> span (the link's stored mapping) -> card (the playground's own cells and
 * study guide, from GET /api/playgrounds).
 *
 * The top half is pure and covered by node --test. The bottom half is the
 * panel: built with createElement + textContent, and the card's code and
 * saved outputs are rendered by playground.js's own renderCode/renderOutput,
 * so the sanitizer is the Playground tab's, not a copy.
 */
(function (global) {
  'use strict';

  var RA = global.ReadAlong ||
    (typeof require === 'function' ? require('./readalong.js') : null);
  var Core = global.ListenCore ||
    (typeof require === 'function' ? require('./listen-core.js') : {});

  var NARROW = '(max-width: 859px)';
  var DRAWER_KEY = 'nlp:followDrawerOpen';
  var TIMELINE_HELP = 'python -m nlp_pillars.cli timeline backfill';

  /* ------------------------------------------------------------ pure helpers */

  /* Cell index -> { step, number } for every cell of the study guide. */
  function guideIndex(steps) {
    var map = {};
    (steps || []).forEach(function (step, i) {
      (step && step.cells || []).forEach(function (cell) {
        if (cell && Number.isInteger(cell.cell_index) && !map[cell.cell_index]) {
          map[cell.cell_index] = { step: step, number: i + 1, guide: cell };
        }
      });
    });
    return map;
  }

  /*
   * The stored mapping, checked again against the playground as it is now: a
   * re-explained notebook can renumber its steps or drop a cell, and a span
   * pointing at a cell that is gone must not show some other code. Sorted by
   * line; an overlapping span is dropped.
   */
  function normalizeSpans(mapping, steps) {
    var guide = guideIndex(steps);
    var spans = (Array.isArray(mapping) ? mapping : []).filter(function (s) {
      if (!s || !Number.isInteger(s.line_start) || !Number.isInteger(s.line_end)) return false;
      if (s.line_start < 0 || s.line_end < s.line_start) return false;
      var hit = guide[s.cell_index];
      return !!hit && hit.step.id === s.step_id;
    }).map(function (s) {
      return { line_start: s.line_start, line_end: s.line_end, cell_index: s.cell_index, step_id: s.step_id };
    });
    spans.sort(function (a, b) { return a.line_start - b.line_start || a.line_end - b.line_end; });
    var out = [];
    spans.forEach(function (s) {
      var last = out[out.length - 1];
      if (!last || s.line_start > last.line_end) out.push(s);
    });
    return out;
  }

  /*
   * The span at a script line: the last span starting at or before it, and
   * whether the line is inside it (otherwise the card is kept, dimmed).
   * { index: -1 } before the first span.
   */
  function spanIndexAt(spans, lineNo) {
    var lo = 0;
    var hi = (spans || []).length - 1;
    var found = -1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (spans[mid].line_start <= lineNo) {
        found = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    if (found === -1) return { index: -1, inside: false };
    return { index: found, inside: lineNo <= spans[found].line_end };
  }

  /*
   * Time -> line -> span. `model` is ReadAlong.buildModel's. Pure in `ms`, so
   * seeking (forwards or back) lands on the same answer as playing there.
   */
  function followAt(model, spans, ms) {
    if (!RA || !model || !spans || !spans.length) return { line: -1, index: -1, inside: false };
    var pos = RA.positionAt(model, ms);
    if (pos.line === -1) return { line: -1, index: -1, inside: false };
    var lineNo = model.lines[pos.line].line;
    var hit = spanIndexAt(spans, lineNo);
    return { line: lineNo, index: hit.index, inside: hit.inside };
  }

  /* What one span shows: the step, its number, the cell's guide entry and the cell itself. */
  function cardFor(span, playground) {
    if (!span || !playground) return null;
    var hit = guideIndex(playground.steps)[span.cell_index];
    if (!hit) return null;
    var cell = null;
    (playground.cells || []).forEach(function (c) { if (c.index === span.cell_index) cell = c; });
    if (!cell || cell.type !== 'code') return null;
    return { step: hit.step, number: hit.number, guide: hit.guide, cell: cell };
  }

  /* The paper's Playground tab, scrolled to one cell. Paper ids carry ':'. */
  function cellHref(paperId, cellIndex) {
    if (!paperId) return null;
    return '/papers/' + encodeURIComponent(paperId) + '?cell=' + Number(cellIndex) + '#playground';
  }

  /*
   * Which link to follow: the one asked for, else the newest ready one, else
   * the newest. `stale` when its mapping indexes another version of the script.
   */
  function chooseLink(links, scriptSha, preferredPlaygroundId) {
    var list = Array.isArray(links) ? links : [];
    var pick = null;
    if (preferredPlaygroundId) {
      list.forEach(function (l) { if (!pick && l.playground_id === preferredPlaygroundId) pick = l; });
    }
    if (!pick) list.forEach(function (l) { if (!pick && l.status === 'ready') pick = l; });
    if (!pick) pick = list[0] || null;
    if (!pick) return { link: null, stale: false };
    var stale = !!(pick.status === 'ready' && scriptSha && pick.script_sha256 && pick.script_sha256 !== scriptSha);
    return { link: pick, stale: stale };
  }

  /*
   * What the panel can do for this episode.
   *
   *   off      no follow-along is possible; `reason` says why
   *   pick     offer the picker (with `reason` when a previous pass failed or went stale)
   *   ready    follow the chosen link
   */
  function availability(opts) {
    var o = opts || {};
    if (!o.hasAudio) {
      return { mode: 'off', reason: 'The code follows the audio, and this episode has none yet. Record it first.' };
    }
    var state = RA ? RA.timelineState(o.timeline, o.scriptSha) : 'missing';
    if (state === 'missing') {
      return {
        mode: 'off',
        reason: 'This episode has no read-along timing, so the code cannot follow it. ' +
          'Time it with: ' + TIMELINE_HELP,
      };
    }
    if (state === 'stale') {
      return {
        mode: 'off',
        reason: 'The script was edited after this audio was recorded, so its timing is off and the code cannot follow it. Record the episode again.',
      };
    }
    var chosen = chooseLink(o.links, o.scriptSha, o.preferred);
    var link = chosen.link;
    if (link && link.status === 'ready' && !chosen.stale) return { mode: 'ready', link: link };
    if (link && chosen.stale) {
      return { mode: 'pick', link: link, reason: 'The script changed after it was matched to this notebook. Match it again.' };
    }
    if (link && link.status === 'failed') {
      return { mode: 'pick', link: link, reason: 'The last match failed: ' + (link.error || 'no reason recorded') + ' Try again.' };
    }
    return { mode: 'pick', link: link };
  }

  /* Picker rows matching a search, title or paper, case-insensitive. */
  function filterPlaygrounds(list, query) {
    var q = String(query || '').trim().toLowerCase();
    var rows = Array.isArray(list) ? list : [];
    if (!q) return rows.slice();
    return rows.filter(function (row) {
      return [row.paper_title, row.title, row.paper_id].some(function (v) {
        return String(v || '').toLowerCase().indexOf(q) !== -1;
      });
    });
  }

  /* The episode's own paper's playground if it has one, else the newest. */
  function defaultPlaygroundId(own, list) {
    if (own && own.id) return own.id;
    var rows = Array.isArray(list) ? list : [];
    return rows.length ? rows[0].id : null;
  }

  function pickerLabel(row) {
    var paper = row.paper_title || row.paper_id || 'Unknown paper';
    var bits = [paper];
    if (row.title && row.title !== paper) bits.push(row.title);
    if (row.status === 'imported') bits.push('no study guide yet');
    return bits.join(' · ');
  }

  /* -------------------------------------------------------------------- panel */

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function readJson(res) {
    return res.json().catch(function () { return {}; }).then(function (body) {
      return { res: res, body: body };
    });
  }

  /*
   * opts: { root, live: () => snapshot of the player when it holds this
   * episode, else null; controller: the shared player (for subscribe) }.
   */
  function mount(opts) {
    var root = opts && opts.root;
    if (!root || typeof document === 'undefined' || !RA) return null;
    var live = opts.live || function () { return null; };
    var controller = opts.controller || null;
    var PG = global.Playground || {};

    var drawer = root.querySelector('#follow-drawer');
    var statusEl = root.querySelector('#follow-status');
    var sourceEl = root.querySelector('#follow-source');
    var pinBtn = root.querySelector('#follow-pin');
    var changeBtn = root.querySelector('#follow-change');
    var slot = root.querySelector('#follow-card-slot');
    var form = root.querySelector('#follow-pick');
    var search = root.querySelector('#follow-search');
    var select = root.querySelector('#follow-select');
    var startBtn = root.querySelector('#follow-start');
    var pickNote = root.querySelector('#follow-pick-note');
    var runCard = root.querySelector('#follow-run-card');
    var runEls = {
      status: root.querySelector('#follow-run-status'),
      stages: root.querySelector('#follow-run-stages'),
      progress: root.querySelector('#follow-run-progress'),
      cancel: root.querySelector('#follow-run-cancel'),
    };
    var home = root.parentNode;
    var homeNext = root.nextSibling;
    var narrowSlot = document.getElementById('follow-slot-narrow');
    var layout = root.closest ? root.closest('.factcheck-layout') : null;

    var ep = null;          // the episode on screen
    var data = null;        // GET /api/follow-along
    var model = null;       // readalong model for this episode
    var spans = [];
    var playground = null;  // the followed playground (cells + steps)
    var shown = { index: -2, inside: false };
    var pinned = false;
    var picking = false;
    var preferred = null;
    var catalog = null;     // picker rows
    var loadToken = 0;
    var runId = null;

    function say(text) {
      if (statusEl) statusEl.textContent = text || '';
    }

    function setVisible(on) {
      root.hidden = !on;
      if (layout) layout.classList.toggle('has-follow', on);
    }

    /* Beside the transcript when there is room; a drawer under the player when not. */
    function place() {
      var narrow = !!(global.matchMedia && global.matchMedia(NARROW).matches);
      if (narrow && narrowSlot && root.parentNode !== narrowSlot) {
        narrowSlot.appendChild(root);
      } else if (!narrow && home && root.parentNode !== home) {
        home.insertBefore(root, homeNext && homeNext.parentNode === home ? homeNext : null);
      }
      root.classList.toggle('is-drawer', narrow);
    }

    function clearCard() {
      if (slot) slot.replaceChildren();
      shown = { index: -2, inside: false };
    }

    function renderCard(span, inside) {
      var card = cardFor(span, playground);
      slot.replaceChildren();
      if (!card) {
        slot.appendChild(el('p', 'follow-empty', 'That cell is no longer in the study guide.'));
        return;
      }
      var href = cellHref(playground.paper_id, card.cell.index);
      var link = el('a', 'follow-card' + (inside ? '' : ' is-dimmed'));
      if (href) link.setAttribute('href', href);
      link.setAttribute('aria-label', 'Open cell ' + card.cell.index + ' on the Playground tab: ' + (card.step.title || ''));
      var count = card.cell.execution_count != null ? 'In [' + card.cell.execution_count + ']' : 'Cell ' + card.cell.index;
      link.appendChild(el('p', 'follow-kicker', 'Step ' + card.number + ' · ' + count));
      link.appendChild(el('h4', 'follow-step', card.step.title || 'Step ' + card.number));
      if (card.guide && card.guide.explanation) link.appendChild(el('p', 'follow-explain', card.guide.explanation));
      if (PG.renderCode) {
        // No line notes here: a note is a button, and this card is one link.
        link.appendChild(PG.renderCode(card.cell.source, []));
      } else {
        link.appendChild(el('pre', 'pg-code', card.cell.source || ''));
      }
      var outputs = card.cell.outputs || [];
      if (outputs.length && PG.renderOutput) {
        var box = el('div', 'pg-outputs');
        box.appendChild(el('p', 'pg-label', 'Saved output'));
        outputs.forEach(function (o) { box.appendChild(PG.renderOutput(o)); });
        link.appendChild(box);
      }
      link.appendChild(el('span', 'follow-open', 'Open in Playground →'));
      slot.appendChild(link);
    }

    function placeholder(text) {
      slot.replaceChildren(el('p', 'follow-empty', text));
    }

    /* Show the card for wherever the audio is. Cheap when nothing changed. */
    function update(force) {
      if (!ep || !playground || !model || !slot) return;
      if (pinned && !force) return;
      var snap = live();
      var ms = snap && isFinite(snap.time) ? snap.time * 1000 : 0;
      var at = snap ? followAt(model, spans, ms) : { index: -1, inside: false };
      if (!force && at.index === shown.index && at.inside === shown.inside) return;
      if (at.index === -1) {
        if (shown.index !== -1 || force) {
          placeholder(snap
            ? 'The code shows up here when the narrator gets to it.'
            : 'Press play. The code the narrator is talking about shows up here.');
        }
      } else if (at.index === shown.index && slot.firstChild && slot.firstChild.classList) {
        slot.firstChild.classList.toggle('is-dimmed', !at.inside);
      } else {
        renderCard(spans[at.index], at.inside);
      }
      shown = { index: at.index, inside: at.inside };
    }

    function paintPin() {
      if (!pinBtn) return;
      pinBtn.setAttribute('aria-pressed', pinned ? 'true' : 'false');
      pinBtn.textContent = pinned ? 'Pinned' : 'Pin';
      pinBtn.title = pinned
        ? 'The card stays put. Press again to follow the audio.'
        : 'Keep this card while the episode plays on';
    }

    function paintPicker(reason) {
      if (!form || !select) return;
      form.hidden = false;
      var rows = filterPlaygrounds(catalog || [], search ? search.value : '');
      var keep = select.value;
      select.replaceChildren();
      rows.forEach(function (row) {
        var option = el('option', null, pickerLabel(row));
        option.value = row.id;
        if (row.status === 'imported') option.disabled = true;
        select.appendChild(option);
      });
      var wanted = keep || defaultPlaygroundId(data && data.own_playground, catalog || []);
      if (wanted) select.value = wanted;
      if (!select.value && select.options.length) {
        for (var i = 0; i < select.options.length; i++) {
          if (!select.options[i].disabled) { select.value = select.options[i].value; break; }
        }
      }
      startBtn.disabled = !select.value || !!runId;
      if (pickNote) {
        pickNote.textContent = !catalog
          ? 'Loading papers with a playground…'
          : !catalog.length
            ? 'No paper has a playground yet. Import a notebook on a paper’s Playground tab first.'
            : !rows.length ? 'No paper matches that search.' : '';
        pickNote.hidden = !pickNote.textContent;
      }
      if (reason) say(reason);
    }

    function loadCatalog() {
      if (catalog) return Promise.resolve(catalog);
      return fetch('/api/follow-along/playgrounds').then(readJson).then(function (out) {
        if (!out.res.ok) throw new Error(out.body.detail || 'Could not list playgrounds (HTTP ' + out.res.status + ').');
        catalog = out.body.playgrounds || [];
        return catalog;
      });
    }

    function showPicker(reason) {
      picking = true;
      clearCard();
      if (pinBtn) pinBtn.hidden = true;
      if (changeBtn) changeBtn.hidden = true;
      if (sourceEl) sourceEl.textContent = '';
      paintPicker(reason);
      loadCatalog().then(function () { paintPicker(reason); }).catch(function (err) {
        catalog = [];
        paintPicker(err.message);
      });
    }

    function hidePicker() {
      picking = false;
      if (form) form.hidden = true;
    }

    function follow(link) {
      hidePicker();
      var pg = link.playground || {};
      if (sourceEl) sourceEl.textContent = 'Following ' + (pg.paper_title || pg.title || 'a notebook');
      if (changeBtn) changeBtn.hidden = false;
      say('Loading the notebook…');
      var token = loadToken;
      return fetch('/api/playgrounds?paper_id=' + encodeURIComponent(pg.paper_id || '')).then(readJson).then(function (out) {
        if (token !== loadToken) return;
        if (!out.res.ok) throw new Error(out.body.detail || 'Could not load the notebook (HTTP ' + out.res.status + ').');
        var got = out.body.playground;
        if (!got || got.id !== link.playground_id) throw new Error('That notebook is no longer on its paper. Pick another.');
        playground = got;
        spans = normalizeSpans(link.mapping, playground.steps);
        if (pinBtn) pinBtn.hidden = !spans.length;
        if (!spans.length) {
          say(link.error || 'No line of this episode is clearly about a cell of this notebook.');
          placeholder('Nothing to follow in this episode.');
          return;
        }
        var dropped = (link.mapping || []).length - spans.length;
        say(spans.length + ' code card' + (spans.length === 1 ? '' : 's') + ' follow this episode' +
          (dropped > 0 ? ' (' + dropped + ' no longer match the study guide)' : '') + '.');
        update(true);
      }).catch(function (err) {
        if (token !== loadToken) return;
        playground = null;
        showPicker(err.message);
      });
    }

    function decide() {
      if (!ep || !data) return;
      var avail = availability({
        hasAudio: ep.hasAudio,
        timeline: ep.timeline,
        scriptSha: ep.scriptSha,
        links: data.links,
        preferred: preferred,
      });
      playground = null;
      spans = [];
      clearCard();
      if (avail.mode === 'off') {
        hidePicker();
        if (pinBtn) pinBtn.hidden = true;
        if (changeBtn) changeBtn.hidden = true;
        if (sourceEl) sourceEl.textContent = '';
        say(avail.reason);
        return;
      }
      if (avail.mode === 'ready') {
        follow(avail.link);
        return;
      }
      showPicker(avail.reason || '');
    }

    /*
     * Follow the alignment run in this panel's own run card. Not
     * RunProgress.attach(): that tracks one run per page and rewrites ?run=,
     * and the Listen page already uses it for script and audio runs. Same
     * rendering (RunProgress.preview), same recursive setTimeout, never
     * setInterval.
     */
    var pollTimer = null;
    function stopPoll() {
      if (pollTimer) global.clearTimeout(pollTimer);
      pollTimer = null;
      runId = null;
    }

    function watchRun(id) {
      if (!id) return;
      stopPoll();
      runId = id;
      var episode = ep && ep.scriptId;
      var delay = 1000;
      if (runCard) runCard.hidden = false;
      if (startBtn) startBtn.disabled = true;
      function finished() {
        stopPoll();
        if (startBtn) startBtn.disabled = false;
        reload();
      }
      function tick() {
        pollTimer = null;
        if (runId !== id || !ep || ep.scriptId !== episode) return;
        fetch('/api/pipeline-runs/' + encodeURIComponent(id)).then(function (res) {
          if (res.status === 404) return null;
          if (!res.ok) throw new Error('HTTP ' + res.status);
          return res.json();
        }).then(function (run) {
          if (runId !== id) return;
          if (!run) {
            if (global.RunProgress) global.RunProgress.preview(null, runEls);
            finished();
            return;
          }
          delay = 1000;
          if (global.RunProgress) global.RunProgress.preview(run, runEls);
          var done = global.RunProgress ? global.RunProgress.isTerminal(run.status)
            : ['succeeded', 'failed', 'cancelled', 'interrupted'].indexOf(run.status) !== -1;
          if (done) {
            finished();
            return;
          }
          pollTimer = global.setTimeout(tick, delay);
        }).catch(function () {
          if (runId !== id) return;
          delay = Math.min(delay * 2, 15000);
          if (runEls.status) runEls.status.textContent = 'Lost contact with the server — retrying. The run is still going.';
          pollTimer = global.setTimeout(tick, delay);
        });
      }
      tick();
    }

    function reload() {
      if (!ep || !ep.scriptId) return Promise.resolve();
      var token = ++loadToken;
      say('Checking for a linked notebook…');
      return fetch('/api/follow-along?script_id=' + encodeURIComponent(ep.scriptId)).then(readJson).then(function (out) {
        if (token !== loadToken) return;
        if (!out.res.ok) {
          data = null;
          hidePicker();
          clearCard();
          say(out.body.detail || 'Follow-along could not be loaded (HTTP ' + out.res.status + ').');
          return;
        }
        data = out.body;
        decide();
        var run = data.latest_run;
        if (run && (run.status === 'pending' || run.status === 'running') && !runId) watchRun(run.id);
      }).catch(function (err) {
        if (token !== loadToken) return;
        say('Follow-along could not be loaded: ' + err.message);
      });
    }

    function start(playgroundId) {
      if (!ep || !playgroundId) return;
      preferred = playgroundId;
      startBtn.disabled = true;
      say('Starting…');
      // spend-guard.js wraps fetch and asks before this paid call.
      fetch('/api/follow-along', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ script_id: ep.scriptId, playground_id: playgroundId }),
      }).then(readJson).then(function (out) {
        if ((out.res.status === 202 || out.res.status === 409) && out.body.run_id) {
          say(out.res.status === 409 ? 'Already matching — following that run.' : 'Matching the episode to the notebook…');
          watchRun(out.body.run_id);
          return;
        }
        throw new Error(out.body.detail || out.body.error || 'HTTP ' + out.res.status);
      }).catch(function (err) {
        say(err.message);
        startBtn.disabled = false;
      });
    }

    if (pinBtn) {
      pinBtn.addEventListener('click', function () {
        pinned = !pinned;
        paintPin();
        if (!pinned) update(true);
      });
    }
    if (changeBtn) changeBtn.addEventListener('click', function () { showPicker(''); });
    if (search) search.addEventListener('input', function () { paintPicker(''); });
    if (select) select.addEventListener('change', function () { startBtn.disabled = !select.value || !!runId; });
    if (form) {
      form.addEventListener('submit', function (event) {
        event.preventDefault();
        if (select && select.value) start(select.value);
      });
    }
    if (runEls.cancel) {
      runEls.cancel.addEventListener('click', function () {
        if (runId && global.RunProgress) global.RunProgress.cancel(runId).catch(function () { /* the poll shows the truth */ });
      });
    }
    if (drawer) {
      try {
        var stored = global.localStorage.getItem(DRAWER_KEY);
        if (stored === 'false') drawer.open = false;
      } catch (e) { /* default open */ }
      drawer.addEventListener('toggle', function () {
        try { global.localStorage.setItem(DRAWER_KEY, drawer.open ? 'true' : 'false'); } catch (e) { /* this view only */ }
      });
    }
    if (global.matchMedia) {
      var mq = global.matchMedia(NARROW);
      if (mq.addEventListener) mq.addEventListener('change', place);
      else if (mq.addListener) mq.addListener(place);
    }
    if (controller && controller.subscribe) controller.subscribe(function () { update(false); });
    place();
    paintPin();

    return {
      /*
       * ep: { scriptId, paperId, scriptText, scriptSha, timeline, hasAudio }.
       * Called when an episode opens and again when its timeline arrives.
       */
      setEpisode: function (next) {
        if (!next || !next.scriptId) {
          ep = null;
          data = null;
          loadToken += 1;
          setVisible(false);
          return;
        }
        var sameScript = ep && ep.scriptId === next.scriptId;
        ep = next;
        model = RA.buildModel(next.timeline, Core.scriptLines ? Core.scriptLines(next.scriptText) : [], next.scriptSha);
        setVisible(true);
        place();
        if (!sameScript) {
          pinned = false;
          preferred = null;
          paintPin();
          reload();
        } else if (data) {
          decide();
        }
      },
      update: update,
    };
  }

  var api = {
    guideIndex: guideIndex,
    normalizeSpans: normalizeSpans,
    spanIndexAt: spanIndexAt,
    followAt: followAt,
    cardFor: cardFor,
    cellHref: cellHref,
    chooseLink: chooseLink,
    availability: availability,
    filterPlaygrounds: filterPlaygrounds,
    defaultPlaygroundId: defaultPlaygroundId,
    pickerLabel: pickerLabel,
    mount: mount,
  };
  global.FollowAlong = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
