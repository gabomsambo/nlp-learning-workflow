/*
 * Paper map page: fetches GET /api/map, draws Timeline and Clusters on a
 * Canvas 2D surface with d3 scales and zoom, and fills the side panel.
 *
 * Every node of the panel is createElement + textContent; no HTML string ever
 * carries a title, an author or a sentence from the snapshot. What the page
 * says about a connection is worded by paper-map-core.js from stored evidence.
 * The rendering sits behind this one file so another renderer could replace it.
 */
(function (global) {
  'use strict';

  if (typeof document === 'undefined') return;
  var Core = global.PaperMapCore;
  var d3 = global.d3;

  var page = document.getElementById('map-page');
  if (!page || !Core) return;

  var PILLAR = page.getAttribute('data-pillar') || '';
  var PILLAR_NAME = page.getAttribute('data-pillar-name') || PILLAR;
  var PILLAR_CLASS = /^[a-z0-9-]+$/;
  var PILLAR_NAMES = {};
  Array.prototype.forEach.call(document.querySelectorAll('.map-chips [data-pillar]'), function (chip) {
    PILLAR_NAMES[chip.getAttribute('data-pillar')] = chip.getAttribute('data-name') || '';
  });

  function pillarName(id) {
    return PILLAR_NAMES[id] || id;
  }

  var els = {
    views: document.getElementById('map-views'),
    notices: document.getElementById('map-notices'),
    stateBox: document.getElementById('map-state'),
    insights: document.getElementById('map-insights'),
    work: document.getElementById('map-work'),
    find: document.getElementById('map-find'),
    yaxis: document.getElementById('map-yaxis'),
    nsugg: document.getElementById('map-nsugg'),
    ncontext: document.getElementById('map-ncontext'),
    cites: document.getElementById('map-cites'),
    lineage: document.getElementById('map-lineage'),
    align: document.getElementById('map-align'),
    count: document.getElementById('map-count'),
    wrap: document.getElementById('map-canvas-wrap'),
    canvas: document.getElementById('map-canvas'),
    list: document.getElementById('map-listwrap'),
    pill: document.getElementById('map-focus-pill'),
    pillName: document.getElementById('map-focus-name'),
    pillExit: document.getElementById('map-focus-exit'),
    hover: document.getElementById('map-hover'),
    zoomIn: document.getElementById('map-zin'),
    zoomOut: document.getElementById('map-zout'),
    zoomFit: document.getElementById('map-zfit'),
    legend: document.getElementById('map-legend'),
    tabs: document.getElementById('map-tabs'),
    panel: document.getElementById('map-panel'),
    foot: document.getElementById('map-foot'),
  };

  var query = Core.parseQuery(global.location.search);
  var state = {
    model: null,
    view: query.view,
    yaxis: 'citations',
    nSugg: Core.DEFAULT_SUGGESTIONS,
    nContext: Core.DEFAULT_CONTEXT,
    contextTouched: false,
    cites: true,
    lineage: true,
    align: true,
    selectedId: null,
    hoverId: null,
    focusId: null,
    find: '',
    matches: {},
    tab: 'suggested',
    direction: '',
    sort: { key: 'role', dir: 'asc' },
    queued: {},
    forced: {},
    transform: d3 ? d3.zoomIdentity : null,
    visible: [],
    layout: null,
    screen: [],
    size: { w: 0, h: 0 },
  };

  var ctx = els.canvas.getContext ? els.canvas.getContext('2d') : null;
  var colorsCache = null;
  var zoom = null;
  var probes = null;
  var pillarColors = {};

  /* ------------------------------------------------------------------ DOM kit */

  function h(tag, props, kids) {
    var el = document.createElement(tag);
    if (props) {
      Object.keys(props).forEach(function (key) {
        var value = props[key];
        if (value === undefined || value === null || value === false) return;
        if (key === 'class') el.className = value;
        else if (key === 'text') el.textContent = value;
        else if (key === 'on') {
          Object.keys(value).forEach(function (name) {
            el.addEventListener(name, value[name]);
          });
        } else el.setAttribute(key, value === true ? '' : String(value));
      });
    }
    (kids || []).forEach(function (kid) {
      if (kid === null || kid === undefined || kid === false) return;
      el.appendChild(typeof kid === 'string' ? document.createTextNode(kid) : kid);
    });
    return el;
  }

  function clear(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function link(text, onClick, extra) {
    return h('button', Object.assign({ type: 'button', class: 'map-link', text: text, on: { click: onClick } }, extra || {}));
  }

  function badge(text, tone) {
    return h('span', { class: 'map-badge map-badge-' + (tone || 'neutral'), text: text });
  }

  /* ------------------------------------------------------------------- colors */

  function readColors() {
    var root = getComputedStyle(document.documentElement);
    var stage = getComputedStyle(els.wrap);
    function v(name, fallback) {
      var value = (stage.getPropertyValue(name) || root.getPropertyValue(name) || '').trim();
      return value || fallback;
    }
    return {
      text: v('--text', '#1E1C19'),
      muted: v('--muted', '#6B665E'),
      border: v('--border', '#E6E2D9'),
      strong: v('--border-strong', '#8F887C'),
      surface: v('--surface', '#FFFFFF'),
      bg: v('--map-bg', v('--bg', '#FBFAF7')),
      accent: v('--accent', '#2F5D50'),
      font: getComputedStyle(document.body).fontFamily || 'sans-serif',
    };
  }

  function colors() {
    if (!colorsCache) colorsCache = readColors();
    return colorsCache;
  }

  function pillarColor(pillarId) {
    if (!pillarId || !PILLAR_CLASS.test(pillarId)) return colors().accent;
    if (pillarColors[pillarId]) return pillarColors[pillarId];
    if (!probes) {
      probes = h('div', { class: 'map-probes', 'aria-hidden': 'true' });
      els.wrap.appendChild(probes);
    }
    var probe = h('span', { class: 'pillar-' + pillarId });
    probes.appendChild(probe);
    var value = getComputedStyle(probe).getPropertyValue('--p').trim();
    probes.removeChild(probe);
    // Do not cache a miss: the pillar stylesheet may not have applied yet.
    if (!value) return colors().accent;
    pillarColors[pillarId] = value;
    return value;
  }

  function alpha(color, a) {
    var hex = /^#([0-9a-f]{6})$/i.exec(color);
    if (hex) {
      var n = parseInt(hex[1], 16);
      return 'rgba(' + (n >> 16) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
    }
    var short = /^#([0-9a-f]{3})$/i.exec(color);
    if (short) {
      var s = short[1];
      return 'rgba(' + parseInt(s[0] + s[0], 16) + ',' + parseInt(s[1] + s[1], 16) + ',' + parseInt(s[2] + s[2], 16) + ',' + a + ')';
    }
    var rgb = /^rgba?\(([^)]+)\)/.exec(color);
    if (rgb) {
      var parts = rgb[1].split(',').slice(0, 3).map(function (p) { return p.trim(); });
      return 'rgba(' + parts.join(',') + ',' + a + ')';
    }
    return color;
  }

  function nodeColor(node) {
    if (node.role === 'yours' || node.role === 'suggested') return pillarColor(PILLAR);
    if (node.role === 'other') return pillarColor(node.pillar);
    return colors().muted;
  }

  /* ------------------------------------------------------------------- loading */

  function setBusy(busy) {
    els.stateBox.hidden = !busy;
    els.work.hidden = true;
    els.insights.hidden = true;
    els.foot.hidden = true;
    if (busy) {
      clear(els.stateBox);
      els.stateBox.appendChild(
        h('div', { class: 'map-skeleton', role: 'status' }, [
          h('p', { class: 'map-skeleton-title', text: 'Loading the map' }),
          h('span', { class: 'skeleton-line' }),
          h('span', { class: 'skeleton-line' }),
          h('span', { class: 'skeleton-line' }),
        ])
      );
    }
  }

  function load() {
    if (!PILLAR) {
      showState({
        kind: 'empty',
        title: 'No pillar to map',
        message: 'The map is built per pillar, and there is no pillar to open it on.',
      });
      return;
    }
    setBusy(true);
    var done = false;
    fetch('/api/map?scope=' + encodeURIComponent(PILLAR), { headers: { Accept: 'application/json' } })
      .then(function (response) {
        return response
          .json()
          .catch(function () {
            return null;
          })
          .then(function (body) {
            return { status: response.status, body: body };
          });
      })
      .then(function (result) {
        done = true;
        var verdict = Core.classify(result.status, result.body, null);
        if (verdict.kind !== 'ready') return showState(verdict);
        start(result.body);
      })
      .catch(function (error) {
        if (done) {
          // A bug after the data arrived must not look like an empty map.
          showState({ kind: 'error', title: 'The map could not be drawn', message: String(error && error.message ? error.message : error) });
          if (global.console) console.error(error);
          return;
        }
        showState(Core.classify(0, null, error && error.message ? error.message : error));
      });
  }

  function showState(verdict) {
    els.work.hidden = true;
    els.insights.hidden = true;
    els.foot.hidden = true;
    clear(els.notices);
    clear(els.stateBox);
    els.stateBox.hidden = false;
    var box = h('div', { class: 'empty map-state-box map-state-' + verdict.kind, role: verdict.kind === 'error' || verdict.kind === 'migration' ? 'alert' : 'status' });
    box.appendChild(h('p', { class: 'empty-title', text: verdict.title }));
    box.appendChild(h('p', { class: 'empty-message', text: verdict.message }));
    if (verdict.command) {
      box.appendChild(h('code', { class: 'map-command', text: verdict.command }));
      box.appendChild(
        h('p', { class: 'map-state-hint', text: 'Run it where the stack runs. It reads stored data only and spends nothing.' })
      );
    }
    if (verdict.migration) {
      box.appendChild(h('code', { class: 'map-command', text: verdict.migration }));
    }
    var actions = h('div', { class: 'map-state-actions' });
    if (verdict.kind === 'error' || verdict.kind === 'migration') {
      actions.appendChild(h('button', { type: 'button', class: 'btn btn-primary btn-md', text: 'Try again', on: { click: load } }));
    }
    if (verdict.kind === 'empty') {
      actions.appendChild(h('a', { class: 'btn btn-primary btn-md', href: '/discovery/' + (PILLAR ? '?pillar=' + encodeURIComponent(PILLAR) : ''), text: 'Add papers' }));
    }
    actions.appendChild(h('a', { class: 'btn btn-secondary btn-md', href: '/papers/' + (PILLAR ? '?pillar=' + encodeURIComponent(PILLAR) : ''), text: 'Open the library list' }));
    box.appendChild(actions);
    els.stateBox.appendChild(box);
  }

  /* ------------------------------------------------------------------- start */

  function start(response) {
    var model = Core.prepare(response, PILLAR);
    state.model = model;
    state.selectedId = null;
    state.focusId = null;
    state.forced = {};
    var availability = Core.citationAvailability(model);
    var haveCounts = Core.hasCitationCounts(model.nodes);
    state.yaxis = haveCounts ? 'citations' : 'topic';
    state.nSugg = Math.min(Core.DEFAULT_SUGGESTIONS, model.suggestions.length) || 0;

    els.stateBox.hidden = true;
    els.work.hidden = false;
    els.insights.hidden = false;
    els.foot.hidden = false;
    buildControls(model, haveCounts);
    renderNotices(model, availability, haveCounts);
    renderInsights();
    renderFoot();
    applyDefaultContext();
    applyViewButtons();
    renderLegend();
    ensureZoom();
    if (query.select) {
      var wanted = findNodeByRef(query.select);
      if (wanted) {
        state.forced[wanted.id] = true;
        state.selectedId = wanted.id;
      }
    }
    refresh(true);
    renderPanel();
  }

  function findNodeByRef(ref) {
    var model = state.model;
    if (model.byId[ref]) return model.byId[ref];
    for (var i = 0; i < model.nodes.length; i++) {
      if (model.nodes[i].paper_ids.indexOf(ref) >= 0) return model.nodes[i];
    }
    return null;
  }

  function buildControls(model, haveCounts) {
    clear(els.yaxis);
    Object.keys(Core.Y_AXES).forEach(function (key) {
      var option = h('option', { value: key, text: Core.Y_AXES[key].label });
      if (key === 'citations' && !haveCounts) {
        option.disabled = true;
        option.textContent += ' (no counts)';
      }
      els.yaxis.appendChild(option);
    });
    els.yaxis.value = state.yaxis;

    clear(els.nsugg);
    var max = model.suggestions.length;
    var choices = Core.SUGGESTION_CHOICES.filter(function (n) { return n < max; });
    choices.push(max);
    choices.forEach(function (n, i) {
      if (i > 0 && n === choices[i - 1]) return;
      els.nsugg.appendChild(h('option', { value: String(n), text: n === 0 ? 'None' : String(n) }));
    });
    els.nsugg.value = String(state.nSugg);

    clear(els.ncontext);
    Core.CONTEXT_CHOICES.forEach(function (n) {
      els.ncontext.appendChild(h('option', { value: String(n), text: n === 0 ? 'None' : String(n) }));
    });
    els.ncontext.value = String(state.nContext);
    els.cites.checked = state.cites;
    els.lineage.checked = state.lineage;
    els.align.checked = state.align;
  }

  function renderNotices(model, availability, haveCounts) {
    clear(els.notices);
    function callout(kind, title, text) {
      els.notices.appendChild(
        h('aside', { class: 'callout callout-' + kind, role: 'status' }, [
          h('p', { class: 'callout-title', text: title }),
          h('div', { class: 'callout-body', text: text }),
        ])
      );
    }
    if (availability.unavailable) callout('warn', 'Citation data unavailable', availability.message);
    if (!haveCounts) callout('warn', 'Citation counts unavailable', availability.countsMessage + ' Papers are placed by community instead.');
    if (model.warnings.length) {
      els.notices.appendChild(
        h('details', { class: 'map-warnings' }, [
          h('summary', { text: 'Build notes (' + model.warnings.length + ')' }),
          h('ul', {}, model.warnings.map(function (w) { return h('li', { text: w }); })),
        ])
      );
    }
  }

  function renderFoot() {
    var model = state.model;
    var bits = [];
    var built = Core.formatBuiltAt(model.builtAt);
    if (built) bits.push('Built ' + built + ' from stored data. Nothing here was computed by a model.');
    if (model.stats.works) {
      bits.push(model.stats.works + ' works, ' + (model.stats.citations || 0) + ' citation links.');
    }
    if (model.lineage.note) bits.push('Lineage is a sketch: ' + model.lineage.note.charAt(0).toLowerCase() + model.lineage.note.slice(1));
    if (!model.clustered) bits.push('Too few works to find communities, so the pillar is one community.');
    els.foot.textContent = bits.join(' ');
  }

  /* ------------------------------------------------------------------ insights */

  function renderInsights() {
    var cards = Core.insightCards(state.model, 4);
    clear(els.insights);
    els.insights.hidden = cards.length === 0;
    cards.forEach(function (card) {
      var button = h(
        'button',
        {
          type: 'button',
          class: 'map-insight',
          on: {
            click: function () {
              if (card.tab) {
                state.selectedId = null;
                state.tab = card.tab;
                renderPanel();
                return;
              }
              if (card.ids.length) {
                if (card.ids.length > 1) {
                  state.forced[card.ids[0]] = true;
                  state.forced[card.ids[1]] = true;
                }
                select(card.ids[0], true);
              }
            },
          },
        },
        [h('span', { class: 'map-insight-k', text: card.title }), h('span', { class: 'map-insight-v', text: card.text })]
      );
      els.insights.appendChild(button);
    });
  }

  /* ------------------------------------------------------------------- layout */

  function visibleOptions() {
    var force = Object.assign({}, state.forced);
    if (state.selectedId) force[state.selectedId] = true;
    return {
      suggestions: state.nSugg,
      context: state.nContext,
      lineage: state.lineage && state.view !== 'list',
      force: force,
    };
  }

  function neighboursOf(id) {
    var out = {};
    var model = state.model;
    (model.cites[id] || []).forEach(function (o) { out[o] = true; });
    (model.citedBy[id] || []).forEach(function (o) { out[o] = true; });
    (model.evidence[id] || []).forEach(function (row) { out[row.other] = true; });
    return out;
  }

  function computeVisible() {
    var model = state.model;
    var visible = Core.visibleNodes(model, visibleOptions());
    if (state.focusId) {
      var ring = neighboursOf(state.focusId);
      ring[state.focusId] = true;
      visible = visible.filter(function (node) { return ring[node.id]; });
    }
    return visible;
  }

  function resize() {
    var rect = els.wrap.getBoundingClientRect();
    var w = Math.max(200, Math.round(rect.width));
    var hgt = Math.max(200, Math.round(rect.height));
    var dpr = global.devicePixelRatio || 1;
    if (state.size.w === w && state.size.h === hgt && state.size.dpr === dpr) return false;
    state.size = { w: w, h: hgt, dpr: dpr };
    els.canvas.width = Math.round(w * dpr);
    els.canvas.height = Math.round(hgt * dpr);
    return true;
  }

  /**
   * World positions for the visible works and the scales that put them on the
   * canvas. The zoom transform is applied on top of these in `project()`.
   */
  function computeLayout() {
    var nodes = state.visible;
    var size = state.size;
    var pad = { l: 58, r: 26, t: 26, b: 46 };
    var layout = { kind: state.view, world: {}, unplaced: [], sx: null, sy: null, pad: pad, gutter: null, axes: null };

    if (state.view === 'clusters') {
      var placed = nodes.filter(function (n) { return typeof n.x === 'number' && typeof n.y === 'number'; });
      layout.unplaced = nodes.filter(function (n) { return !(typeof n.x === 'number' && typeof n.y === 'number'); });
      var xs = Core.extent(placed.map(function (n) { return n.x; }), 0.08);
      var ys = Core.extent(placed.map(function (n) { return n.y; }), 0.08);
      var cpad = 44;
      var scale = Math.min((size.w - 2 * cpad) / (xs[1] - xs[0]), (size.h - 2 * cpad) / (ys[1] - ys[0]));
      var cx = (xs[0] + xs[1]) / 2;
      var cy = (ys[0] + ys[1]) / 2;
      layout.sx = d3.scaleLinear().domain([cx - size.w / 2 / scale, cx + size.w / 2 / scale]).range([0, size.w]);
      layout.sy = d3.scaleLinear().domain([cy - size.h / 2 / scale, cy + size.h / 2 / scale]).range([size.h, 0]);
      placed.forEach(function (n) { layout.world[n.id] = [n.x, n.y]; });
      return layout;
    }

    // Timeline
    var dated = nodes.filter(function (n) { return Core.timelineX(n) !== null; });
    var undated = nodes.filter(function (n) { return Core.timelineX(n) === null; });
    var years = dated.map(Core.timelineX);
    var xdom = Core.extent(years.length ? years : [2000, 2025], 0.04);
    if (xdom[1] - xdom[0] < 4) { xdom[0] -= 2; xdom[1] += 2; }
    var gutterWidth = undated.length ? (xdom[1] - xdom[0]) * 0.07 : 0;
    var xmin = xdom[0] - gutterWidth;
    layout.sx = d3.scaleLinear().domain([xmin, xdom[1]]).range([pad.l, size.w - pad.r]);
    if (undated.length) layout.gutter = { x: xdom[0] - gutterWidth / 2, edge: xdom[0] - gutterWidth * 0.02, count: undated.length };

    var values = nodes.map(function (n) { return Core.yValue(n, state.yaxis, state.model); });
    var known = values.filter(function (v) { return v !== null; });
    var ymax = known.length ? Math.max.apply(null, known) : 1;
    var ydom;
    if (state.yaxis === 'cohort') ydom = [0, 100];
    else if (state.yaxis === 'topic') ydom = Core.extent(known.length ? known : [0, 1], 0.12);
    else ydom = [0, Math.max(1, ymax * 1.1)];
    layout.sy = d3.scaleLinear().domain(ydom).range([size.h - pad.b, pad.t]);

    nodes.forEach(function (n, index) {
      var tx = Core.timelineX(n);
      if (tx === null) tx = layout.gutter.x + (Core.hash01(n.id) - 0.5) * gutterWidth * 0.6;
      var ty = values[index];
      if (ty === null) ty = state.yaxis === 'topic' ? ydom[0] : 0;
      layout.world[n.id] = [tx, ty];
    });
    layout.axes = { yaxis: state.yaxis, ymax: ymax };
    return layout;
  }

  function project(id) {
    var w = state.layout && state.layout.world[id];
    if (!w) return null;
    var t = state.transform;
    var x = state.layout.sx(w[0]);
    var y = state.layout.sy(w[1]);
    return [t.applyX(x), t.applyY(y)];
  }

  /* ------------------------------------------------------------------ refresh */

  function refresh(resetZoom) {
    if (!state.model) return;
    colorsCache = null;
    var wasResized = resize();
    state.matches = Core.findMatches(state.model.nodes, state.find);
    var isList = state.view === 'list';
    els.canvas.hidden = isList;
    els.list.hidden = !isList;
    els.zoomIn.parentNode.hidden = isList;
    els.wrap.classList.toggle('is-list', isList);
    page.setAttribute('data-view', state.view);
    state.visible = computeVisible();
    var timelineOnly = els.work.querySelectorAll('[data-only="timeline"]');
    Array.prototype.forEach.call(timelineOnly, function (el) { el.hidden = state.view !== 'timeline'; });
    var focusNode = state.focusId && state.model.byId[state.focusId];
    els.pill.hidden = !focusNode;
    if (focusNode) els.pillName.textContent = focusNode.label || focusNode.title;
    renderLegend();
    if (isList) {
      els.count.textContent = countText();
      renderList();
      return;
    }
    state.layout = computeLayout();
    els.count.textContent = countText();
    if (resetZoom || wasResized) {
      state.transform = d3.zoomIdentity;
      if (zoom) d3.select(els.canvas).call(zoom.transform, d3.zoomIdentity);
    }
    draw();
  }

  function countText() {
    var n = state.visible.length;
    var yours = state.visible.filter(function (x) { return x.role === 'yours'; }).length;
    var text = Core.plural(n, 'paper') + ' shown, ' + yours + ' yours';
    if (state.view !== 'list' && state.layout && state.layout.unplaced.length) {
      text += ' · ' + state.layout.unplaced.length + ' not placed';
    }
    return text;
  }

  /* --------------------------------------------------------------------- draw */

  function draw() {
    if (!ctx || state.view === 'list' || !state.layout) return;
    var c = colors();
    var dpr = state.size.dpr || 1;
    var w = state.size.w;
    var hgt = state.size.h;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, hgt);
    ctx.fillStyle = c.bg;
    ctx.fillRect(0, 0, w, hgt);

    var model = state.model;
    var screen = state.visible
      .map(function (node) {
        var p = project(node.id);
        if (!p) return null;
        return { node: node, x: p[0], y: p[1], r: Core.radiusFor(node.citation_count, node.role) };
      })
      .filter(Boolean);
    var at = {};
    screen.forEach(function (s) { at[s.node.id] = s; });
    state.screen = screen;

    if (state.view === 'timeline') drawAxes(c);
    if (state.view === 'clusters' && model.clustered) drawRegions(c, screen);

    var focusId = state.selectedId;
    var ring = focusId ? neighboursOf(focusId) : null;
    if (state.align) drawAlignment(c, at, focusId);
    if (state.cites) drawCitations(c, at, focusId);
    if (state.lineage) drawLineage(c, at);
    drawNodes(c, screen, focusId, ring);
    drawLabels(c, screen, focusId, ring);
    if (state.view === 'clusters' && model.clustered) drawRegionLabels(c, screen);
    if (state.layout.gutter && state.view === 'timeline') drawGutter(c);
    ctx.globalAlpha = 1;
  }

  function drawAxes(c) {
    var layout = state.layout;
    var t = state.transform;
    var size = state.size;
    var pad = layout.pad;
    var sx = t.rescaleX(layout.sx);
    var sy = t.rescaleY(layout.sy);
    ctx.save();
    ctx.font = '11px ' + c.font;
    ctx.fillStyle = c.muted;
    ctx.strokeStyle = alpha(c.text, 0.06);
    ctx.lineWidth = 1;
    // Year ticks
    var d = sx.domain();
    var ticks = Core.yearTicks(Math.max(d[0], 1900), d[1], Math.max(3, Math.floor(size.w / 90)));
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ticks.forEach(function (year) {
      var x = Math.round(sx(year)) + 0.5;
      if (x < pad.l - 4 || x > size.w - pad.r + 4) return;
      ctx.beginPath();
      ctx.moveTo(x, pad.t);
      ctx.lineTo(x, size.h - pad.b);
      ctx.stroke();
      ctx.fillText(String(year), x, size.h - pad.b + 6);
    });
    // Y ticks
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    var yticks = [];
    if (layout.axes.yaxis === 'citations') {
      yticks = Core.citationTicks(sy.domain()[1] > 0 ? Math.max(sy.domain()[1], 0) : 1).map(function (tk) { return { v: tk.value, label: Core.compactCount(tk.count) }; });
    } else if (layout.axes.yaxis === 'cohort') {
      yticks = [0, 25, 50, 75, 100].map(function (v) { return { v: v, label: v + '%' }; });
    } else if (layout.axes.yaxis === 'local') {
      yticks = [0, 1, 2, 5, 10, 20, 50].map(function (n) { return { v: Math.log10(1 + n), label: String(n) }; });
    } else {
      yticks = [];
    }
    yticks.forEach(function (tk) {
      var y = Math.round(sy(tk.v)) + 0.5;
      if (y < pad.t - 2 || y > size.h - pad.b + 2) return;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(size.w - pad.r, y);
      ctx.stroke();
      ctx.fillText(tk.label, pad.l - 8, y);
    });
    // Captions
    ctx.fillStyle = alpha(c.muted, 0.85);
    ctx.font = '600 10px ' + c.font;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText('MORE RECENTLY PUBLISHED →', size.w - pad.r, size.h - 4);
    ctx.save();
    ctx.translate(14, pad.t + 4);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText(Core.Y_AXES[layout.axes.yaxis].caption + ' →', 0, 0);
    ctx.restore();
    ctx.restore();
  }

  function drawGutter(c) {
    var layout = state.layout;
    var t = state.transform;
    var x = t.applyX(layout.sx(layout.gutter.edge));
    ctx.save();
    ctx.strokeStyle = alpha(c.text, 0.18);
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(x, layout.pad.t);
    ctx.lineTo(x, state.size.h - layout.pad.b);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = c.muted;
    ctx.font = '600 10px ' + c.font;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    var cx = t.applyX(layout.sx(layout.gutter.x));
    ctx.fillText('NO DATE', cx, layout.pad.t - 18);
    ctx.restore();
  }

  function arrowHead(x, y, angle, size) {
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x - size * Math.cos(angle - 0.45), y - size * Math.sin(angle - 0.45));
    ctx.lineTo(x - size * Math.cos(angle + 0.45), y - size * Math.sin(angle + 0.45));
    ctx.closePath();
    ctx.fill();
  }

  /** One curve from `from` to `to`, leaving horizontally on the Timeline. */
  function curve(from, to, kind) {
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    var angle;
    if (kind === 'timeline') {
      var dx = Math.max(24, Math.abs(from.x - to.x) * 0.5);
      var dir = to.x <= from.x ? -1 : 1;
      ctx.bezierCurveTo(from.x + dir * dx, from.y, to.x - dir * dx, to.y, to.x, to.y);
      angle = dir > 0 ? 0 : Math.PI;
    } else {
      var mx = (from.x + to.x) / 2;
      var my = (from.y + to.y) / 2;
      var nx = -(to.y - from.y) * 0.12;
      var ny = (to.x - from.x) * 0.12;
      ctx.quadraticCurveTo(mx + nx, my + ny, to.x, to.y);
      angle = Math.atan2(to.y - (my + ny), to.x - (mx + nx));
    }
    return angle;
  }

  function endPoint(s, angle, gap) {
    return { x: s.x - Math.cos(angle) * (s.r + gap), y: s.y - Math.sin(angle) * (s.r + gap) };
  }

  function drawCitations(c, at, focusId) {
    var model = state.model;
    var kind = state.view;
    var weak = focusId ? 0.05 : state.screen.length > 120 ? 0.1 : 0.17;
    ctx.save();
    ctx.lineWidth = 1;
    var drawn = [];
    model.citations.forEach(function (cite) {
      var a = at[cite.citing];
      var b = at[cite.cited];
      if (!a || !b) return;
      var hot = focusId && (cite.citing === focusId || cite.cited === focusId);
      if (hot) drawn.push({ a: a, b: b, cite: cite });
      else {
        ctx.strokeStyle = alpha(c.text, cite.influential ? Math.min(0.32, weak * 1.8) : weak);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.lineWidth = cite.influential ? 1.4 : 1;
        var angle = curve(a, b, kind);
        ctx.stroke();
        if (!focusId) {
          var tip = endPoint(b, angle, 1.5);
          arrowHead(tip.x, tip.y, angle, 4.5);
        }
      }
    });
    drawn.forEach(function (row) {
      var color = nodeColor(model.byId[focusId]);
      if (model.byId[focusId].role === 'context') color = c.text;
      ctx.strokeStyle = alpha(color, 0.9);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = 1.8;
      var angle = curve(row.a, row.b, kind);
      ctx.stroke();
      var tip = endPoint(row.b, angle, 1.5);
      arrowHead(tip.x, tip.y, angle, 7);
    });
    ctx.restore();
  }

  function drawAlignment(c, at, focusId) {
    var model = state.model;
    var seen = {};
    var pairs = [];
    var max = 0;
    state.visible.forEach(function (node) {
      if (node.role !== 'yours') return;
      (model.evidence[node.id] || []).forEach(function (row) {
        var other = model.byId[row.other];
        if (!other || other.role !== 'yours' || !at[other.id]) return;
        var key = node.id < other.id ? node.id + '|' + other.id : other.id + '|' + node.id;
        if (seen[key]) return;
        seen[key] = true;
        pairs.push({ a: at[node.id], b: at[other.id], w: row.weight });
        max = Math.max(max, row.weight);
      });
    });
    if (!pairs.length) return;
    ctx.save();
    ctx.setLineDash([6, 5]);
    pairs.forEach(function (pair) {
      var strength = max > 0 ? pair.w / max : 0;
      var hot = focusId && (pair.a.node.id === focusId || pair.b.node.id === focusId);
      ctx.strokeStyle = alpha(pillarColor(PILLAR), (focusId && !hot ? 0.07 : 0.25) + 0.4 * strength * (focusId && !hot ? 0.3 : 1));
      ctx.lineWidth = 1 + 3 * strength;
      ctx.beginPath();
      ctx.moveTo(pair.a.x, pair.a.y);
      ctx.lineTo(pair.b.x, pair.b.y);
      ctx.stroke();
    });
    ctx.restore();
  }

  function drawLineage(c, at) {
    var arcs = Core.lineageArcs(state.model);
    if (!arcs.length) return;
    ctx.save();
    ctx.lineCap = 'round';
    arcs.forEach(function (arc) {
      var a = at[arc.from];
      var b = at[arc.to];
      if (!a || !b) return;
      // A path runs oldest to newest, so the arrow marks the newer end.
      ctx.strokeStyle = alpha(c.accent, 0.78);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = 3;
      ctx.setLineDash(arc.direct ? [] : [2, 6]);
      var angle = curve(a, b, state.view);
      ctx.stroke();
      ctx.setLineDash([]);
      var tip = endPoint(b, angle, 3);
      arrowHead(tip.x, tip.y, angle, 8);
    });
    ctx.restore();
  }

  function drawNodes(c, screen, focusId, ring) {
    var order = { context: 0, other: 1, suggested: 2, yours: 3 };
    var sorted = screen.slice().sort(function (p, q) {
      return order[p.node.role] - order[q.node.role];
    });
    var hasFind = Object.keys(state.matches).length > 0 || state.find.trim() !== '';
    sorted.forEach(function (s) {
      var node = s.node;
      var dim = false;
      if (focusId && node.id !== focusId && !(ring && ring[node.id])) dim = true;
      if (hasFind && !state.matches[node.id]) dim = true;
      ctx.globalAlpha = dim ? 0.18 : 1;
      var color = nodeColor(node);
      if (node.id === focusId) {
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r + 7, 0, Math.PI * 2);
        ctx.fillStyle = alpha(color, 0.22);
        ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      if (node.role === 'yours') {
        ctx.fillStyle = color;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = c.surface;
        ctx.stroke();
        if (node.kind === 'document') {
          ctx.beginPath();
          ctx.arc(s.x, s.y, s.r + 3, 0, Math.PI * 2);
          ctx.setLineDash([2, 2.5]);
          ctx.lineWidth = 1.2;
          ctx.strokeStyle = color;
          ctx.stroke();
          ctx.setLineDash([]);
        }
      } else if (node.role === 'other') {
        ctx.fillStyle = alpha(color, 0.38);
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = color;
        ctx.stroke();
      } else if (node.role === 'suggested') {
        ctx.fillStyle = c.bg;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = color;
        ctx.stroke();
        ctx.fillStyle = c.text;
        ctx.font = '600 ' + (node.position > 9 ? 9 : 10) + 'px ' + c.font;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(node.position), s.x, s.y + 0.5);
      } else {
        ctx.fillStyle = alpha(c.muted, 0.55);
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = alpha(c.surface, 0.9);
        ctx.stroke();
      }
      if (state.lineage && node.on_lineage && node.role !== 'context') {
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r + 3.2, 0, Math.PI * 2);
        ctx.lineWidth = 1.6;
        ctx.strokeStyle = alpha(c.accent, 0.85);
        ctx.stroke();
      }
      if (node.id === focusId || node.id === state.hoverId) {
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r + (node.id === focusId ? 4 : 3), 0, Math.PI * 2);
        ctx.lineWidth = 2;
        ctx.strokeStyle = c.text;
        ctx.stroke();
      }
    });
    ctx.globalAlpha = 1;
  }

  function drawLabels(c, screen, focusId, ring) {
    ctx.save();
    var size = state.size;
    var items = screen
      .filter(function (s) {
        return s.x > -20 && s.x < size.w + 20 && s.y > -20 && s.y < size.h + 20;
      })
      .map(function (s) {
        var expanded = s.node.id === focusId;
        var text = Core.nodeCaption(s.node, expanded);
        ctx.font = (expanded || s.node.role === 'yours' ? '600 ' : '500 ') + '11px ' + c.font;
        return {
          id: s.node.id,
          node: s.node,
          x: s.x,
          y: s.y,
          r: s.r + (expanded ? 5 : 1),
          w: Math.ceil(ctx.measureText(text).width) + 6,
          h: 14,
          text: text,
          pri: Core.labelPriority(s.node, { selectedId: focusId, hoverId: state.hoverId, matches: state.matches, lineage: state.lineage, neighbours: ring }),
        };
      })
      .sort(function (a, b) { return b.pri - a.pri; });
    var placed = Core.placeLabels(items, { x: 0, y: 0, w: size.w, h: size.h });
    var byId = {};
    items.forEach(function (item) { byId[item.id] = item; });
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    placed.forEach(function (box) {
      var item = byId[box.id];
      var dim = focusId && item.node.id !== focusId && !(ring && ring[item.node.id]);
      ctx.globalAlpha = dim ? 0.3 : 1;
      ctx.font = (item.node.id === focusId || item.node.role === 'yours' ? '600 ' : '500 ') + '11px ' + c.font;
      ctx.lineWidth = 3;
      ctx.strokeStyle = alpha(c.bg, 0.92);
      ctx.lineJoin = 'round';
      ctx.strokeText(item.text, box.x + 3, box.y + 1);
      ctx.fillStyle = item.node.role === 'context' ? c.muted : c.text;
      ctx.fillText(item.text, box.x + 3, box.y + 1);
    });
    ctx.restore();
  }

  /**
   * The communities the Clusters view outlines: regions (the coarse level)
   * when the build found at least three, else topics. One region around
   * everything says nothing, and the topics are the level that separates.
   */
  function clusterLevel() {
    var model = state.model;
    var regions = Object.keys(model.regions).length;
    return regions >= 3 ? { key: 'region', index: model.regions } : { key: 'topic', index: model.topics };
  }

  function regionGroups(screen) {
    var level = clusterLevel();
    var groups = {};
    screen.forEach(function (s) {
      var id = s.node[level.key];
      if (id === null || id === undefined) return;
      (groups[id] = groups[id] || []).push(s);
    });
    return Object.keys(groups).map(function (id) {
      return { id: id, region: level.index[id], points: groups[id] };
    });
  }

  function drawRegions(c, screen) {
    ctx.save();
    regionGroups(screen).forEach(function (group) {
      if (group.points.length < 4) return;
      var padded = [];
      group.points.forEach(function (s) {
        for (var k = 0; k < 8; k++) {
          var a = (k / 8) * Math.PI * 2;
          padded.push([s.x + Math.cos(a) * (s.r + 22), s.y + Math.sin(a) * (s.r + 22)]);
        }
      });
      var hull = d3.polygonHull(padded);
      if (!hull) return;
      ctx.beginPath();
      var path = d3.line().curve(d3.curveCatmullRomClosed.alpha(0.6)).context(ctx);
      path(hull);
      ctx.fillStyle = alpha(c.text, 0.035);
      ctx.fill();
      ctx.setLineDash([4, 5]);
      ctx.lineWidth = 1;
      ctx.strokeStyle = alpha(c.text, 0.16);
      ctx.stroke();
    });
    ctx.restore();
  }

  function drawRegionLabels(c, screen) {
    var model = state.model;
    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    var placed = [];
    var groups = regionGroups(screen)
      .filter(function (g) { return g.region && g.points.length >= 4; })
      .sort(function (a, b) { return b.points.length - a.points.length; });
    var showSub = groups.length <= 4 || state.transform.k >= 1.6;
    groups.forEach(function (group) {
      var region = group.region;
      var terms = (region.terms || []).slice(0, 3).join(' \u00b7 ');
      if (!terms) return;
      var rep = region.representative && model.byId[region.representative];
      var sub = showSub && rep && rep.label ? 'around ' + rep.label : '';
      var cx = d3.mean(group.points, function (p) { return p.x; });
      var top = d3.min(group.points, function (p) { return p.y - p.r; }) - 26;
      ctx.font = '700 11px ' + c.font;
      var text = terms.toUpperCase();
      var width = ctx.measureText(text).width + 10;
      var boxH = sub ? 28 : 16;
      var y = Math.max(14, top);
      // Try the label a few steps lower before giving up on it.
      for (var attempt = 0; attempt < 4; attempt++, y += 16) {
        var box = { x: cx - width / 2, y: y - 8, w: width, h: boxH };
        if (box.x < 2 || box.x + box.w > state.size.w - 2 || box.y + box.h > state.size.h - 2) continue;
        var clash = placed.some(function (o) {
          return box.x < o.x + o.w && o.x < box.x + box.w && box.y < o.y + o.h && o.y < box.y + box.h;
        });
        if (clash) continue;
        placed.push(box);
        ctx.font = '700 11px ' + c.font;
        ctx.lineWidth = 4;
        ctx.lineJoin = 'round';
        ctx.strokeStyle = alpha(c.bg, 0.9);
        ctx.strokeText(text, cx, y);
        ctx.fillStyle = alpha(c.text, 0.62);
        ctx.fillText(text, cx, y);
        if (sub) {
          ctx.font = '500 10px ' + c.font;
          ctx.strokeText(sub, cx, y + 13);
          ctx.fillStyle = alpha(c.muted, 0.95);
          ctx.fillText(sub, cx, y + 13);
        }
        break;
      }
    });
    ctx.restore();
  }

  /* ---------------------------------------------------------------- interaction */

  function ensureZoom() {
    if (zoom || !d3) return;
    zoom = d3
      .zoom()
      .scaleExtent([0.6, 40])
      .filter(function (event) {
        if (event.type === 'dblclick') return false;
        return !event.ctrlKey || event.type === 'wheel';
      })
      .on('zoom', function (event) {
        state.transform = event.transform;
        hideHover();
        draw();
      });
    d3.select(els.canvas).call(zoom).on('dblclick.zoom', null);
  }

  function pointer(event) {
    var rect = els.canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  function hit(p) {
    var best = null;
    var bestD = Infinity;
    for (var i = 0; i < state.screen.length; i++) {
      var s = state.screen[i];
      var d = Math.hypot(s.x - p.x, s.y - p.y);
      var limit = s.r + 4;
      if (d <= limit && d < bestD) {
        best = s;
        bestD = d;
      }
    }
    return best;
  }

  function showHover(s, p) {
    var node = s.node;
    var model = state.model;
    clear(els.hover);
    els.hover.appendChild(h('div', { class: 'map-hover-title', text: node.title || node.label }));
    var meta = [];
    if (node.authors.length) meta.push(node.authors.slice(0, 2).join(', ') + (node.authors.length > 2 ? ' et al.' : ''));
    if (node.year) meta.push(String(node.year));
    if (node.venue) meta.push(node.venue);
    if (meta.length) els.hover.appendChild(h('div', { class: 'map-hover-meta', text: meta.join(' · ') }));
    var cites = node.citation_count === null || node.citation_count === undefined ? 'Citations unknown' : node.citation_count.toLocaleString() + ' citations';
    els.hover.appendChild(h('div', { class: 'map-hover-meta', text: Core.roleLabel(node) + ' · ' + cites }));
    if (node.role === 'suggested' && model.suggestionById[node.id]) {
      els.hover.appendChild(h('div', { class: 'map-hover-why', text: Core.oneLineReason(model, model.suggestionById[node.id]) }));
    }
    els.hover.hidden = false;
    var wrapW = state.size.w;
    var width = els.hover.offsetWidth;
    var heightPx = els.hover.offsetHeight;
    var left = p.x + 14;
    if (left + width > wrapW - 6) left = p.x - width - 14;
    var top = p.y + 14;
    if (top + heightPx > state.size.h - 6) top = p.y - heightPx - 14;
    els.hover.style.left = Math.max(6, left) + 'px';
    els.hover.style.top = Math.max(6, top) + 'px';
  }

  function hideHover() {
    els.hover.hidden = true;
  }

  function onMove(event) {
    if (event.buttons) return;
    var p = pointer(event);
    var s = hit(p);
    var id = s ? s.node.id : null;
    els.canvas.classList.toggle('is-hovering', !!s);
    if (id !== state.hoverId) {
      state.hoverId = id;
      draw();
    }
    if (s) showHover(s, p);
    else hideHover();
  }

  var downAt = null;
  function onDown(event) {
    downAt = pointer(event);
  }

  function onUp(event) {
    if (!downAt) return;
    var p = pointer(event);
    var moved = Math.hypot(p.x - downAt.x, p.y - downAt.y);
    downAt = null;
    if (moved > 4) return;
    var s = hit(p);
    if (s) select(s.node.id, false);
    else if (state.selectedId) select(null, false);
  }

  function onDouble(event) {
    var s = hit(pointer(event));
    if (!s) return;
    event.preventDefault();
    state.selectedId = s.node.id;
    focusOn(s.node.id);
  }

  function select(id, centre) {
    state.selectedId = id;
    if (id && state.model.byId[id] && !state.visible.some(function (n) { return n.id === id; })) {
      state.forced[id] = true;
    }
    syncUrl();
    refreshKeepZoom();
    if (id && centre) panTo(id);
    renderPanel();
    if (state.view === 'list') markListSelection();
  }

  /** Redraw after a selection without throwing away the learner's zoom. */
  function refreshKeepZoom() {
    if (state.view === 'list') {
      refresh(false);
      return;
    }
    var transform = state.transform;
    state.visible = computeVisible();
    state.layout = computeLayout();
    state.transform = transform;
    els.count.textContent = countText();
    draw();
  }

  function panTo(id) {
    if (!state.layout || !state.layout.world[id] || !zoom) return;
    var w = state.layout.world[id];
    var base = [state.layout.sx(w[0]), state.layout.sy(w[1])];
    var t = state.transform;
    var k = Math.max(t.k, 1.4);
    var next = d3.zoomIdentity.translate(state.size.w / 2 - base[0] * k, state.size.h / 2 - base[1] * k).scale(k);
    d3.select(els.canvas).transition().duration(350).call(zoom.transform, next);
  }

  function focusOn(id) {
    state.focusId = id;
    state.forced[id] = true;
    renderPanel();
    refresh(true);
  }

  function exitFocus() {
    state.focusId = null;
    refresh(true);
    renderPanel();
  }

  function zoomBy(factor) {
    if (!zoom) return;
    d3.select(els.canvas).transition().duration(180).call(zoom.scaleBy, factor);
  }

  function syncUrl() {
    var next = Core.buildQuery({ pillar: PILLAR, view: state.view, select: state.selectedId || '' });
    try {
      global.history.replaceState(null, '', global.location.pathname + next);
    } catch (e) {
      /* a URL is a convenience */
    }
  }

  function onKey(event) {
    var tag = event.target && event.target.tagName;
    var typing = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
    if (event.key === '/' && !typing && !event.metaKey && !event.ctrlKey) {
      event.preventDefault();
      els.find.focus();
      els.find.select();
      return;
    }
    if (event.key === 'Escape') {
      if (state.focusId) exitFocus();
      else if (state.find && event.target === els.find) {
        els.find.value = '';
        setFind('');
      } else if (state.selectedId && !typing) select(null, false);
    }
  }

  function onCanvasKey(event) {
    var visible = state.screen.slice().sort(function (a, b) { return a.x - b.x || a.y - b.y; });
    if (!visible.length) return;
    var index = -1;
    visible.forEach(function (s, i) { if (s.node.id === state.selectedId) index = i; });
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') index = (index + 1) % visible.length;
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') index = (index - 1 + visible.length) % visible.length;
    else if (event.key === '+' || event.key === '=') return zoomBy(1.4);
    else if (event.key === '-') return zoomBy(1 / 1.4);
    else if (event.key === 'Enter' && state.selectedId) return focusOn(state.selectedId);
    else return;
    event.preventDefault();
    select(visible[index].node.id, false);
  }

  function setFind(value) {
    state.find = value;
    state.matches = Core.findMatches(state.model.nodes, value);
    if (state.view === 'list') renderList();
    else draw();
  }

  /* ----------------------------------------------------------------- list view */

  var COLUMNS = [
    { key: 'title', label: 'Paper' },
    { key: 'year', label: 'Year', num: true },
    { key: 'citations', label: 'Citations', num: true },
    { key: 'local', label: 'Cited on this map', num: true },
    { key: 'role', label: 'Role' },
    { key: 'topic', label: 'Topic' },
  ];

  function renderList() {
    var model = state.model;
    var rows = Core.sortRows(state.visible, state.sort.key, state.sort.dir);
    var hasFind = state.find.trim() !== '';
    if (hasFind) rows = rows.filter(function (n) { return state.matches[n.id]; });
    clear(els.list);
    var thead = h('tr', {}, COLUMNS.map(function (col) {
      var active = state.sort.key === col.key;
      return h(
        'th',
        {
          scope: 'col',
          class: col.num ? 'is-num' : '',
          'aria-sort': active ? (state.sort.dir === 'asc' ? 'ascending' : 'descending') : 'none',
        },
        [
          h('button', {
            type: 'button',
            class: 'map-sort',
            text: col.label + (active ? (state.sort.dir === 'asc' ? ' ↑' : ' ↓') : ''),
            on: {
              click: function () {
                state.sort = { key: col.key, dir: active && state.sort.dir === 'asc' ? 'desc' : 'asc' };
                renderList();
              },
            },
          }),
        ]
      );
    }));
    var tbody = h('tbody', {}, rows.map(function (node) {
      var topic = node.topic !== null && node.topic !== undefined && model.topics[node.topic] ? (model.topics[node.topic].terms || []).slice(0, 2).join(', ') : '';
      var tr = h(
        'tr',
        { 'data-id': node.id, 'aria-selected': node.id === state.selectedId ? 'true' : 'false' },
        [
          h('td', { class: 'map-list-title' }, [
            h('button', {
              type: 'button',
              class: 'map-link map-list-open',
              text: node.title || node.label,
              on: { click: function () { select(node.id, false); } },
            }),
            h('span', { class: 'map-list-sub', text: node.authors.slice(0, 2).join(', ') + (node.authors.length > 2 ? ' et al.' : '') }),
          ]),
          h('td', { class: 'is-num', text: node.year === null || node.year === undefined ? '—' : String(node.year) }),
          h('td', { class: 'is-num', text: node.citation_count === null || node.citation_count === undefined ? '—' : node.citation_count.toLocaleString() }),
          h('td', { class: 'is-num', text: String(node.local_citations || 0) }),
          h('td', {}, [badge(Core.roleLabel(node), node.role === 'yours' ? 'accent' : node.role === 'suggested' ? 'info' : 'neutral')]),
          h('td', { class: 'map-list-topic', text: topic }),
        ]
      );
      return tr;
    }));
    var table = h('table', { class: 'data-table map-table' }, [
      h('caption', { class: 'visually-hidden', text: 'Papers on the map' }),
      h('thead', {}, [thead]),
      tbody,
    ]);
    els.list.appendChild(h('div', { class: 'table-wrap' }, [table]));
    if (!rows.length) {
      els.list.appendChild(h('p', { class: 'map-list-empty', text: hasFind ? 'No paper on this map matches the search.' : 'No papers to list.' }));
    }
  }

  function markListSelection() {
    Array.prototype.forEach.call(els.list.querySelectorAll('tr[data-id]'), function (tr) {
      tr.setAttribute('aria-selected', tr.getAttribute('data-id') === state.selectedId ? 'true' : 'false');
    });
  }

  /* -------------------------------------------------------------------- legend */

  function svg(width, height, kids) {
    var ns = 'http://www.w3.org/2000/svg';
    var el = document.createElementNS(ns, 'svg');
    el.setAttribute('width', String(width));
    el.setAttribute('height', String(height));
    el.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
    el.setAttribute('aria-hidden', 'true');
    kids.forEach(function (k) {
      var child = document.createElementNS(ns, k[0]);
      Object.keys(k[1]).forEach(function (a) { child.setAttribute(a, k[1][a]); });
      el.appendChild(child);
    });
    return el;
  }

  function renderLegend() {
    var color = pillarColor(PILLAR);
    clear(els.legend);
    function item(icon, text) {
      els.legend.appendChild(h('span', { class: 'map-legend-item' }, [icon, text]));
    }
    if (state.view === 'list') {
      item(null, 'The same papers as the map, as a table. Click a paper for the panel.');
      return;
    }
    item(svg(14, 14, [['circle', { cx: 7, cy: 7, r: 5.5, fill: color }]]), 'Yours');
    item(svg(14, 14, [['circle', { cx: 7, cy: 7, r: 5, fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }]]), 'Suggested (numbered)');
    item(svg(14, 14, [['circle', { cx: 7, cy: 7, r: 5, fill: color, 'fill-opacity': 0.38, stroke: color, 'stroke-width': 1.5 }]]), 'Yours, other pillar');
    item(svg(14, 14, [['circle', { cx: 7, cy: 7, r: 3, fill: 'currentColor', 'fill-opacity': 0.5 }]]), 'Connected');
    item(svg(28, 10, [['path', { d: 'M27 5 C18 5 12 5 3 5', stroke: 'currentColor', 'stroke-width': 1.2, fill: 'none' }], ['path', { d: 'M2 5 l5 -3 v6z', fill: 'currentColor' }]]), 'Cites (arrow at the cited paper)');
    if (state.lineage) item(svg(28, 10, [['path', { d: 'M2 5 H26', stroke: color === '' ? 'currentColor' : 'var(--accent)', 'stroke-width': 3, 'stroke-linecap': 'round' }]]), 'Main line of development (a sketch)');
    if (state.align) item(svg(28, 10, [['path', { d: 'M2 5 H26', stroke: color, 'stroke-width': 2.4, 'stroke-dasharray': '5 4' }]]), 'Aligned (thicker is stronger)');
    var rule = state.view === 'timeline' ? 'Size is citations (log). ' : 'Distance is similarity; only distance means anything. ';
    els.legend.appendChild(h('span', { class: 'map-legend-note', text: rule + 'Double-click a paper to focus on its links.' }));
  }

  /* --------------------------------------------------------------------- panel */

  function setTab(tab) {
    state.tab = tab;
    state.selectedId = null;
    syncUrl();
    refreshKeepZoom();
    renderPanel();
  }

  function renderTabs() {
    Array.prototype.forEach.call(els.tabs.querySelectorAll('[role=tab]'), function (btn) {
      var active = btn.getAttribute('data-tab') === state.tab && !state.selectedId;
      btn.setAttribute('aria-selected', active ? 'true' : 'false');
    });
  }

  function renderPanel() {
    renderTabs();
    clear(els.panel);
    els.panel.scrollTop = 0;
    if (state.selectedId && state.model.byId[state.selectedId]) {
      els.panel.appendChild(detailView(state.model.byId[state.selectedId]));
      return;
    }
    if (state.tab === 'queue') els.panel.appendChild(queueView());
    else if (state.tab === 'yours') els.panel.appendChild(yoursView());
    else if (state.tab === 'lineage') els.panel.appendChild(lineageView());
    else els.panel.appendChild(suggestedView());
  }

  function metaLine(node) {
    var bits = [];
    if (node.authors.length) bits.push(node.authors.slice(0, 3).join(', ') + (node.authors.length > 3 ? ' et al.' : ''));
    if (node.year) bits.push(String(node.year));
    if (node.venue) bits.push(node.venue);
    if (node.citation_count !== null && node.citation_count !== undefined) bits.push(node.citation_count.toLocaleString() + ' citations');
    return bits.join(' · ');
  }

  function nodeLinks(ids) {
    var wrap = document.createDocumentFragment();
    (ids || []).forEach(function (id, i) {
      if (i) wrap.appendChild(document.createTextNode(', '));
      wrap.appendChild(link(Core.labelOf(state.model, id), function () { select(id, true); }));
    });
    return wrap;
  }

  function queueButton(node) {
    var status = state.queued[node.id];
    var label = '+';
    var cls = 'map-add';
    var text = 'Add ' + (node.label || 'paper') + ' to this pillar’s discovery queue';
    if (status && status.kind === 'pending') {
      label = '…';
    } else if (status && status.kind === 'queued') {
      label = 'Queued';
      cls += ' is-done';
    } else if (status && status.kind === 'no_pdf') {
      label = 'No PDF';
      cls += ' is-blocked';
    }
    var button = h('button', {
      type: 'button',
      class: cls,
      'aria-label': text,
      title: status && status.message ? status.message : 'Add to the discovery queue. Nothing is processed or spent.',
      text: label,
      on: {
        click: function (event) {
          event.stopPropagation();
          queueNode(node);
        },
      },
    });
    if (status && (status.kind === 'pending' || status.kind === 'queued')) button.disabled = true;
    return button;
  }

  function queueNode(node) {
    var request = Core.queueRequest(PILLAR, node);
    state.queued[node.id] = { kind: 'pending', message: '' };
    renderPanel();
    fetch(request.url, request.options)
      .then(function (response) {
        return response
          .json()
          .catch(function () { return null; })
          .then(function (body) { return { status: response.status, body: body }; });
      })
      .then(function (result) {
        state.queued[node.id] = Core.interpretQueueResponse(result.status, result.body);
        renderPanel();
      })
      .catch(function (error) {
        state.queued[node.id] = { kind: 'error', message: 'Could not reach the server: ' + (error && error.message ? error.message : error) };
        renderPanel();
      });
  }

  function queueNote(node) {
    var status = state.queued[node.id];
    if (!status || status.kind === 'pending') return null;
    var tone = status.kind === 'queued' ? 'ok' : 'warn';
    var kids = [status.message];
    if (status.kind === 'no_pdf') {
      kids = [status.message + ' ', h('a', { href: '/pillars/' + encodeURIComponent(PILLAR) + '#add', text: 'Upload it from the pillar page' }), '.'];
    }
    return h('p', { class: 'map-note map-note-' + tone, role: 'status' }, kids);
  }

  function suggestedView() {
    var model = state.model;
    var root = h('div', { class: 'map-suggested' });
    root.appendChild(h('p', { class: 'map-panel-intro', text: 'Papers not in your library that link most strongly to this pillar’s papers. “+” adds one to the discovery queue; nothing is processed or spent.' }));
    var chips = h('div', { class: 'filter-chips map-dir-chips', role: 'group', 'aria-label': 'Direction' });
    [['', 'All'], ['foundation', 'Foundations'], ['frontier', 'Frontier']].forEach(function (pair) {
      chips.appendChild(h('button', {
        type: 'button',
        class: 'filter-chip',
        'aria-pressed': state.direction === pair[0] ? 'true' : 'false',
        text: pair[1],
        on: { click: function () { state.direction = pair[0]; renderPanel(); } },
      }));
    });
    root.appendChild(chips);
    var items = model.suggestions.filter(function (item) { return !state.direction || item.direction === state.direction; });
    if (!model.suggestions.length) {
      root.appendChild(h('p', { class: 'map-panel-empty', text: 'The build found no suggestions for this pillar. That can mean the pillar has no citation links yet; it does not mean there is nothing to read.' }));
      return root;
    }
    if (!items.length) {
      root.appendChild(h('p', { class: 'map-panel-empty', text: 'No suggestion in this direction.' }));
      return root;
    }
    var list = h('ol', { class: 'map-srows' });
    items.forEach(function (item) {
      var node = model.byId[item.id];
      if (!node) return;
      var dir = Core.DIRECTIONS[item.direction];
      var row = h('li', { class: 'map-srow' + (node.id === state.hoverId ? ' is-hl' : ''), 'data-id': node.id }, [
        h('span', { class: 'map-rank', text: String(item.position) }),
        h('div', { class: 'map-srow-main' }, [
          h('button', { type: 'button', class: 'map-stitle', text: node.title || node.label, on: { click: function () { select(node.id, true); } } }),
          h('div', { class: 'map-srow-meta', text: metaLine(node) }),
          h('div', { class: 'map-srow-why', text: Core.oneLineReason(model, item) }),
          dir ? badge(dir.label, item.direction === 'foundation' ? 'info' : item.direction === 'frontier' ? 'accent' : 'neutral') : null,
          queueNote(node),
        ]),
        queueButton(node),
      ]);
      list.appendChild(row);
    });
    root.appendChild(list);
    return root;
  }

  function queueView() {
    var model = state.model;
    var view = Core.queueView(model);
    var root = h('div', { class: 'map-queue' });
    root.appendChild(h('p', { class: 'map-panel-intro', text: 'Your discovery queue, sorted by how well each paper fits this pillar. Nothing is hidden or removed; this is a hint about where to spend attention.' }));
    if (!view.available) {
      root.appendChild(h('p', { class: 'map-note map-note-warn', role: 'status', text: 'The live queue could not be read (' + view.reason + '). Verdicts from the build are shown without what changed since.' }));
    }
    if (!view.rows.length) {
      root.appendChild(h('p', { class: 'map-panel-empty', text: view.available ? 'The queue for this pillar is empty.' : 'The build holds no queue verdicts for this pillar.' }));
      return root;
    }
    var list = h('ul', { class: 'map-qrows' });
    view.rows.forEach(function (row) {
      var node = row.node ? model.byId[row.node] : null;
      var title = node ? link(row.title || node.title, function () { select(node.id, true); }, { class: 'map-link map-qtitle' }) : h('span', { class: 'map-qtitle', text: row.title || '(untitled)' });
      var status = [];
      if (row.processed) status.push('processed');
      if (row.gone) status.push('removed since the build');
      list.appendChild(h('li', { class: 'map-qrow' }, [
        h('div', { class: 'map-qhead' }, [badge(row.info.label, row.info.tone), status.length ? h('span', { class: 'map-qstatus', text: status.join(' · ') }) : null]),
        title,
        row.reason ? h('div', { class: 'map-srow-why', text: row.reason }) : null,
      ]));
    });
    root.appendChild(list);
    if (view.removed.length && !view.rows.some(function (r) { return r.gone; })) {
      root.appendChild(h('p', { class: 'map-panel-foot', text: Core.plural(view.removed.length, 'queued paper') + ' removed since the build.' }));
    }
    return root;
  }

  function yoursView() {
    var model = state.model;
    var root = h('div', { class: 'map-yours' });
    root.appendChild(h('p', { class: 'map-panel-intro', text: 'The papers you have in ' + (PILLAR_NAME || 'this pillar') + '. Recall comes from your reviewed quiz cards.' }));
    if (!model.yours.length) {
      root.appendChild(h('p', { class: 'map-panel-empty', text: 'No papers in this pillar yet.' }));
      return root;
    }
    var sorted = Core.sortRows(model.yours, 'year', 'desc');
    var list = h('ul', { class: 'map-yrows' });
    sorted.forEach(function (node) {
      var m = Core.masteryText(model, node);
      var bar = h('div', { class: 'map-bar', role: 'img', 'aria-label': m.text }, [h('i', {})]);
      if (m.known) bar.firstChild.style.width = Math.round(m.value * 100) + '%';
      list.appendChild(h('li', { class: 'map-yrow' }, [
        link(node.title || node.label, function () { select(node.id, true); }, { class: 'map-link map-qtitle' }),
        h('div', { class: 'map-srow-meta', text: metaLine(node) }),
        h('div', { class: 'map-recall' }, [bar, h('span', { class: 'map-recall-text', text: m.text })]),
      ]));
    });
    root.appendChild(list);
    return root;
  }

  function lineageView() {
    var model = state.model;
    var root = h('div', { class: 'map-lineage' });
    var steps = Core.lineageSteps(model);
    root.appendChild(h('p', { class: 'map-panel-intro', text: model.lineage.note || 'A likely line of development through what this map shows.' }));
    if (model.lineage.kind === 'none' || !steps.length) {
      root.appendChild(h('p', { class: 'map-panel-empty', text: 'There are no citation links between dated works on this map, so no line of development can be traced. That reflects the data, not the pillar.' }));
      return root;
    }
    if (model.lineage.kind === 'main_path') {
      root.appendChild(h('p', { class: 'map-note map-note-warn', text: 'None of your papers are on the citation graph, so this is the main path through the neighbourhood, not through your papers.' }));
    }
    steps.forEach(function (path, index) {
      var ol = h('ol', { class: 'map-steps' });
      path.forEach(function (step) {
        ol.appendChild(h('li', { class: step.yours ? 'is-yours' : '' }, [
          link(step.label, function () { select(step.id, true); }),
          h('span', { class: 'map-step-year', text: step.year ? ' ' + step.year : '' }),
          step.yours ? badge('yours', 'accent') : null,
        ]));
      });
      root.appendChild(h('h3', { class: 'map-h3', text: steps.length > 1 ? 'Path ' + (index + 1) : 'Path' }));
      root.appendChild(ol);
    });
    return root;
  }

  function detailView(node) {
    var model = state.model;
    var root = h('div', { class: 'map-detail' });
    root.appendChild(h('button', { type: 'button', class: 'map-back', text: '← Back', on: { click: function () { select(null, false); } } }));
    var head = h('div', { class: 'map-detail-badges' }, [badge(Core.roleLabel(node), node.role === 'yours' ? 'accent' : node.role === 'suggested' ? 'info' : 'neutral')]);
    if (node.pillar && (node.role === 'yours' || node.role === 'other')) {
      var pill = h('span', { class: 'chip chip-pillar pillar-' + (PILLAR_CLASS.test(node.pillar) ? node.pillar : ''), text: node.pillar === PILLAR ? PILLAR_NAME : pillarName(node.pillar) });
      head.appendChild(pill);
    }
    root.appendChild(head);
    root.appendChild(h('h2', { class: 'map-detail-title', text: node.title || node.label }));
    var meta = metaLine(node);
    if (meta) root.appendChild(h('p', { class: 'map-detail-meta', text: meta }));

    var actions = h('div', { class: 'map-actions' });
    if ((node.role === 'yours' || node.role === 'other') && node.paper_ids.length) {
      actions.appendChild(h('a', { class: 'btn btn-primary btn-sm', href: Core.paperHref(node.paper_ids[0]), text: 'Open paper' }));
    }
    if (node.role === 'suggested' || (node.role === 'context' && !node.queue_ids.length)) {
      actions.appendChild(h('span', { class: 'map-actions-add' }, [queueButton(node), h('span', { class: 'map-actions-label', text: 'Add to the discovery queue' })]));
    }
    if (node.title && node.role !== 'yours' && node.role !== 'other') {
      actions.appendChild(h('a', { class: 'btn btn-secondary btn-sm', href: 'https://www.semanticscholar.org/search?q=' + encodeURIComponent(node.title), target: '_blank', rel: 'noopener noreferrer', text: 'Search Semantic Scholar' }));
    }
    actions.appendChild(h('button', { type: 'button', class: 'btn btn-ghost btn-sm', text: 'Focus on map', on: { click: function () { focusOn(node.id); } } }));
    root.appendChild(actions);
    var note = queueNote(node);
    if (note) root.appendChild(note);

    root.appendChild(whyBox(node));

    if (node.role === 'yours') {
      var m = Core.masteryText(model, node);
      var bar = h('div', { class: 'map-bar', role: 'img', 'aria-label': m.text }, [h('i', {})]);
      if (m.known) bar.firstChild.style.width = Math.round(m.value * 100) + '%';
      root.appendChild(h('h3', { class: 'map-h3', text: 'Recall' }));
      root.appendChild(h('div', { class: 'map-recall' }, [bar, h('span', { class: 'map-recall-text', text: m.text })]));
    }

    var community = [];
    var topic = node.topic !== null && node.topic !== undefined ? model.topics[node.topic] : null;
    if (topic && topic.terms && topic.terms.length) community.push('Topic: ' + topic.terms.slice(0, 4).join(', ') + ' (' + Core.plural(topic.size, 'paper') + ')');
    var region = node.region !== null && node.region !== undefined ? model.regions[node.region] : null;
    if (region && region.terms && region.terms.length && model.clustered) community.push('Region: ' + region.terms.slice(0, 3).join(', '));
    if (community.length) {
      root.appendChild(h('h3', { class: 'map-h3', text: 'Where it sits' }));
      community.forEach(function (line) { root.appendChild(h('p', { class: 'map-line', text: line })); });
    }
    return root;
  }

  function whyBox(node) {
    var model = state.model;
    var mine = node.role === 'yours' || node.role === 'other';
    var box = h('section', { class: 'map-why' });
    box.appendChild(h('h3', { class: 'map-h3', text: mine ? 'How it sits in your library' : 'Why it is connected' }));
    var list = h('ul', {});
    var info = Core.connectionsFor(model, node.id);

    var suggestion = model.suggestionById[node.id];
    if (suggestion) {
      var dir = Core.DIRECTIONS[suggestion.direction];
      if (dir) list.appendChild(h('li', { text: dir.label + ': ' + dir.hint + '.' }));
      Core.suggestionReasons(model, suggestion).forEach(function (reason) {
        var li = h('li', {}, [reason.text]);
        if (reason.ids.length) {
          li.appendChild(document.createTextNode(': '));
          li.appendChild(nodeLinks(reason.ids));
        }
        list.appendChild(li);
      });
    } else {
      if (info.citedByYours.length) {
        var a = h('li', {}, ['Cited by ' + Core.plural(info.citedByYours.length, 'paper') + ' of yours: ']);
        a.appendChild(nodeLinks(info.citedByYours.map(function (n) { return n.id; })));
        list.appendChild(a);
      }
      if (info.citesYours.length) {
        var b = h('li', {}, ['Cites ' + Core.plural(info.citesYours.length, 'paper') + ' of yours: ']);
        b.appendChild(nodeLinks(info.citesYours.map(function (n) { return n.id; })));
        list.appendChild(b);
      }
    }
    var position = Core.lineagePosition(model, node.id);
    if (position) {
      list.appendChild(h('li', { text: 'On the main line of development, step ' + position.step + ' of ' + position.of + '. It is a sketch built from the citations on this map.' }));
    }
    if (node.duplicate_of) {
      var dup = h('li', {}, ['Looks like another version of ']);
      dup.appendChild(nodeLinks([node.duplicate_of]));
      dup.appendChild(document.createTextNode('.'));
      list.appendChild(dup);
    }
    if (info.citedByCount || info.citesCount) {
      list.appendChild(h('li', { text: 'In the neighbourhood this map was built from, it is cited by ' + Core.plural(info.citedByCount, 'work') + ' and cites ' + Core.plural(info.citesCount, 'work') + '.' }));
    }
    var queued = node.queue_ids.length ? model.queueRows.filter(function (r) { return r.node === node.id; })[0] : null;
    if (queued) {
      var vi = Core.verdictInfo(queued.verdict);
      list.appendChild(h('li', { text: 'It is in your discovery queue: ' + vi.label.toLowerCase() + '.' }));
    }
    if (list.childNodes.length) box.appendChild(list);

    if (info.evidence.length) {
      box.appendChild(h('h4', { class: 'map-h4', text: 'Strongest links' }));
      var ev = h('ul', { class: 'map-evidence' });
      info.evidence.forEach(function (row) {
        ev.appendChild(h('li', {}, [
          link(Core.labelOf(model, row.other), function () { select(row.other, true); }, { class: 'map-link map-ev-who' }),
          h('span', { class: 'map-ev-text', text: row.sentence }),
        ]));
      });
      box.appendChild(ev);
      if (info.more) box.appendChild(h('p', { class: 'map-panel-foot', text: '+ ' + info.more + ' more links stored for this paper.' }));
    }
    if (!list.childNodes.length && !info.evidence.length) {
      box.appendChild(h('p', { class: 'map-panel-empty', text: 'The snapshot holds no stored link between this paper and the others shown. It is on the map by position only.' }));
    }
    return box;
  }

  /* ---------------------------------------------------------------- wiring up */

  /** Clusters is about communities, so it opens with more of the neighbourhood drawn. */
  function applyDefaultContext() {
    if (state.contextTouched) return;
    state.nContext = state.view === 'clusters' ? Core.CLUSTER_CONTEXT : Core.DEFAULT_CONTEXT;
    els.ncontext.value = String(state.nContext);
  }

  function applyViewButtons() {
    Array.prototype.forEach.call(els.views.querySelectorAll('button'), function (btn) {
      btn.setAttribute('aria-pressed', btn.getAttribute('data-view') === state.view ? 'true' : 'false');
    });
  }

  function wire() {
    Array.prototype.forEach.call(els.views.querySelectorAll('button'), function (btn) {
      btn.addEventListener('click', function () {
        var view = btn.getAttribute('data-view');
        if (view === state.view) return;
        state.view = view;
        applyDefaultContext();
        applyViewButtons();
        syncUrl();
        if (state.model) refresh(true);
      });
    });
    Array.prototype.forEach.call(els.tabs.querySelectorAll('[role=tab]'), function (btn) {
      btn.addEventListener('click', function () { setTab(btn.getAttribute('data-tab')); });
    });
    els.yaxis.addEventListener('change', function () { state.yaxis = els.yaxis.value; refresh(true); });
    els.nsugg.addEventListener('change', function () { state.nSugg = parseInt(els.nsugg.value, 10) || 0; state.forced = {}; refresh(false); });
    els.ncontext.addEventListener('change', function () { state.nContext = parseInt(els.ncontext.value, 10) || 0; state.contextTouched = true; state.forced = {}; refresh(false); });
    els.cites.addEventListener('change', function () { state.cites = els.cites.checked; draw(); });
    els.align.addEventListener('change', function () { state.align = els.align.checked; draw(); });
    els.lineage.addEventListener('change', function () { state.lineage = els.lineage.checked; renderLegend(); refresh(false); });
    els.find.addEventListener('input', function () { setFind(els.find.value); });
    els.find.addEventListener('keydown', function (event) {
      if (event.key !== 'Enter') return;
      var ids = Object.keys(state.matches);
      if (ids.length) select(ids[0], true);
    });
    els.pillExit.addEventListener('click', exitFocus);
    els.zoomIn.addEventListener('click', function () { zoomBy(1.5); });
    els.zoomOut.addEventListener('click', function () { zoomBy(1 / 1.5); });
    els.zoomFit.addEventListener('click', function () {
      if (zoom) d3.select(els.canvas).transition().duration(250).call(zoom.transform, d3.zoomIdentity);
    });
    els.canvas.addEventListener('pointermove', onMove);
    els.canvas.addEventListener('pointerdown', onDown);
    els.canvas.addEventListener('pointerup', onUp);
    els.canvas.addEventListener('pointerleave', function () {
      hideHover();
      if (state.hoverId) {
        state.hoverId = null;
        draw();
      }
    });
    els.canvas.addEventListener('dblclick', onDouble);
    els.canvas.addEventListener('keydown', onCanvasKey);
    document.addEventListener('keydown', onKey);
    if (global.ResizeObserver) {
      new global.ResizeObserver(function () {
        if (state.model && state.view !== 'list') refresh(false);
      }).observe(els.wrap);
    } else {
      global.addEventListener('resize', function () { if (state.model) refresh(false); });
    }
    // Light, dark and the Look setting all repaint the canvas from the tokens.
    var repaint = function () {
      colorsCache = null;
      pillarColors = {};
      if (state.model) {
        renderLegend();
        draw();
      }
    };
    new MutationObserver(repaint).observe(document.documentElement, { attributes: true, attributeFilter: ['data-mode', 'data-theme', 'data-density'] });
    if (global.matchMedia) {
      var mq = global.matchMedia('(prefers-color-scheme: dark)');
      if (mq.addEventListener) mq.addEventListener('change', repaint);
    }
  }

  if (!d3 || !ctx) {
    showState({
      kind: 'error',
      title: 'The map could not start',
      message: 'This page needs d3 and a canvas, and one of them did not load. The library list has the same papers.',
    });
    return;
  }
  wire();
  load();
})(typeof window !== 'undefined' ? window : globalThis);
