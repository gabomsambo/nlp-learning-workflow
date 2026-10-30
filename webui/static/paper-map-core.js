/*
 * Paper map: the pure half. No DOM, no canvas, no fetch.
 *
 * `prepare()` turns the GET /api/map answer into the model the page draws;
 * everything else here is a small rule the page depends on and a test can
 * check: which works are visible, how big a dot is, where the labels fit,
 * what a connection says, what state the page is in, what "+" posts.
 *
 * The snapshot's evidence sentences are written by the build job from stored
 * data. This file adds no claim of its own: it counts, orders and words what
 * the snapshot already holds, and says "unavailable" where it holds nothing.
 */
(function (global) {
  'use strict';

  var VIEWS = ['timeline', 'clusters', 'list'];
  var DEFAULT_VIEW = 'timeline';
  var SUGGESTION_CHOICES = [0, 5, 10, 15, 20];
  var DEFAULT_SUGGESTIONS = 20;
  var CONTEXT_CHOICES = [0, 15, 30, 60];
  var DEFAULT_CONTEXT = 15;
  var CLUSTER_CONTEXT = 60;
  var EVIDENCE_SHOWN = 8;

  var Y_AXES = {
    citations: { label: 'Citations (log)', caption: 'MORE CITATIONS' },
    local: { label: 'Cited within this map', caption: 'CITED MORE OFTEN ON THIS MAP' },
    cohort: { label: 'Percentile among same-year papers', caption: 'HIGHER AMONG PAPERS OF ITS YEAR' },
    topic: { label: 'Community', caption: 'COMMUNITY' },
  };

  /* Best fit first, as the build job sorts them. Nothing is ever hidden. */
  var VERDICTS = {
    close_strong: { label: 'Close to your papers', tone: 'ok', order: 0 },
    close_related: { label: 'Related to your papers', tone: 'info', order: 1 },
    on_topic_new_direction: { label: 'On topic, a new direction', tone: 'accent', order: 2 },
    far_unanchored: { label: 'Far from your papers', tone: 'neutral', order: 3 },
    off_topic: { label: 'Off topic', tone: 'warn', order: 4 },
    duplicate: { label: 'Looks like a paper you have', tone: 'neutral', order: 5 },
    already_yours: { label: 'Already in your library', tone: 'neutral', order: 6 },
    unscored: { label: 'Not scored', tone: 'neutral', order: 7 },
  };

  var DIRECTIONS = {
    foundation: { label: 'Foundation', hint: 'older work your papers build on' },
    frontier: { label: 'Frontier', hint: 'newer work that builds on yours' },
    related: { label: 'Related', hint: 'close to your papers, neither older nor newer' },
  };

  /* ------------------------------------------------------------- small utils */

  function isArray(value) {
    return Array.isArray(value);
  }

  function num(value) {
    return typeof value === 'number' && isFinite(value) ? value : null;
  }

  function plural(count, one, many) {
    return count + ' ' + (count === 1 ? one : many || one + 's');
  }

  function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
  }

  /* A stable 0..1 from a string, so dots in one year do not stack on a line. */
  function hash01(text) {
    var h = 2166136261;
    var s = String(text);
    for (var i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return (h % 10007) / 10007;
  }

  /* --------------------------------------------------------------- the query */

  function parseQuery(search) {
    var out = { pillar: '', view: DEFAULT_VIEW, select: '' };
    var params;
    try {
      params = new URLSearchParams(search || '');
    } catch (e) {
      return out;
    }
    out.pillar = params.get('pillar') || '';
    var view = params.get('view') || '';
    out.view = VIEWS.indexOf(view) >= 0 ? view : DEFAULT_VIEW;
    out.select = params.get('select') || '';
    return out;
  }

  function buildQuery(state) {
    var params = new URLSearchParams();
    if (state.pillar) params.set('pillar', state.pillar);
    if (state.view && state.view !== DEFAULT_VIEW) params.set('view', state.view);
    if (state.select) params.set('select', state.select);
    var text = params.toString();
    return text ? '?' + text : '';
  }

  function paperHref(paperId) {
    if (typeof paperId !== 'string' || !paperId) return null;
    return '/papers/' + encodeURIComponent(paperId);
  }

  function mapHref(pillarId) {
    return '/papers/map' + (pillarId ? '?pillar=' + encodeURIComponent(pillarId) : '');
  }

  /* ------------------------------------------------------------------ states */

  /**
   * What the page is showing, decided before anything is drawn.
   * `kind` is one of ready | not_built | empty | error | migration. An error is
   * never reported as an empty map; that is the one thing this function is for.
   */
  function classify(httpStatus, body, networkError) {
    if (networkError) {
      return {
        kind: 'error',
        title: 'Could not load the map',
        message: 'The request failed: ' + String(networkError) + '. This does not mean the map is empty.',
      };
    }
    if (httpStatus === 404) {
      return {
        kind: 'error',
        title: 'No such pillar',
        message: detail(body) || 'The map is built per pillar, and this pillar does not exist.',
      };
    }
    if (httpStatus >= 400) {
      var text = detail(body) || 'The server answered ' + httpStatus + '.';
      var migration = /docs\/migrations\/\d+_[\w.]+\.sql/.exec(text);
      return {
        kind: migration ? 'migration' : 'error',
        title: migration ? 'The map tables are not set up' : 'Could not load the map',
        message: text + (migration ? '' : ' This does not mean the map is empty.'),
        migration: migration ? migration[0] : null,
      };
    }
    if (!body || typeof body !== 'object' || typeof body.status !== 'string') {
      return {
        kind: 'error',
        title: 'Could not read the map',
        message: 'The server sent something that is not a map. This does not mean the map is empty.',
      };
    }
    if (body.status === 'not_built') {
      return {
        kind: 'not_built',
        title: 'This map has not been built yet',
        message: body.message || 'Nothing has been computed for this pillar yet.',
        command: buildCommand(body.scope),
      };
    }
    if (body.status === 'empty') {
      return {
        kind: 'empty',
        title: 'This map holds no papers yet',
        message: body.message || 'The map was built and holds no works.',
      };
    }
    if (body.status === 'ready' && body.snapshot && isArray(body.snapshot.nodes)) {
      return { kind: 'ready' };
    }
    return {
      kind: 'error',
      title: 'Could not read the map',
      message: 'The map answered "' + body.status + '" without a snapshot. This does not mean it is empty.',
    };
  }

  function detail(body) {
    if (!body) return '';
    if (typeof body.detail === 'string') return body.detail;
    if (isArray(body.detail)) {
      return body.detail
        .map(function (item) {
          return item && item.msg ? item.msg : '';
        })
        .filter(Boolean)
        .join('; ');
    }
    return '';
  }

  function buildCommand(scope) {
    return 'python -m nlp_pillars.cli map build --pillar ' + (scope || '<pillar>');
  }

  /* ------------------------------------------------------------------- model */

  /**
   * The snapshot, indexed. `pillarId` is the scope the page asked for.
   * Tolerates every optional part being absent: an older snapshot, an overlay
   * that could not be read, a payload from before a field existed.
   */
  function prepare(response, pillarId) {
    var snap = (response && response.snapshot) || {};
    var overlays = (response && response.overlays) || {};
    var rawNodes = isArray(snap.nodes) ? snap.nodes : [];
    var suggestionItems = snap.suggestions && isArray(snap.suggestions.pillar) ? snap.suggestions.pillar : [];

    var suggestions = suggestionItems.map(function (item, index) {
      return Object.assign({}, item, { position: index + 1 });
    });
    var suggestionById = {};
    suggestions.forEach(function (item) {
      suggestionById[item.id] = item;
    });

    var nodes = rawNodes.map(function (raw) {
      var node = Object.assign({}, raw);
      node.authors = isArray(raw.authors) ? raw.authors : [];
      node.paper_ids = isArray(raw.paper_ids) ? raw.paper_ids : [];
      node.pillars = isArray(raw.pillars) ? raw.pillars : [];
      node.queue_ids = isArray(raw.queue_ids) ? raw.queue_ids : [];
      node.role = roleOf(node, pillarId, suggestionById);
      node.position = suggestionById[node.id] ? suggestionById[node.id].position : null;
      node.pillar = node.pillars.indexOf(pillarId) >= 0 ? pillarId : node.pillars[0] || '';
      return node;
    });
    var byId = {};
    nodes.forEach(function (node) {
      byId[node.id] = node;
    });

    var citations = (isArray(snap.citations) ? snap.citations : []).filter(function (c) {
      return c && byId[c.citing] && byId[c.cited];
    });
    var cites = {};
    var citedBy = {};
    citations.forEach(function (c) {
      (cites[c.citing] = cites[c.citing] || []).push(c.cited);
      (citedBy[c.cited] = citedBy[c.cited] || []).push(c.citing);
    });

    var evidence = {};
    (isArray(snap.evidence) ? snap.evidence : []).forEach(function (row) {
      if (!row || !byId[row.a] || !byId[row.b]) return;
      push(evidence, row.a, { other: row.b, weight: num(row.weight) || 0, sentence: row.sentence || '', evidence: row.evidence || {} });
      push(evidence, row.b, { other: row.a, weight: num(row.weight) || 0, sentence: row.sentence || '', evidence: row.evidence || {} });
    });
    Object.keys(evidence).forEach(function (id) {
      evidence[id].sort(function (a, b) {
        return b.weight - a.weight || (a.other < b.other ? -1 : 1);
      });
    });

    var communities = snap.communities || {};
    var topics = indexById(communities.topics);
    var regions = indexById(communities.regions);

    var lineagePaths = snap.lineage && isArray(snap.lineage.paths) ? snap.lineage.paths : [];

    var queueOverlay = overlays.queue || { available: false, reason: 'The queue was not read.' };
    var masteryOverlay = overlays.mastery || { available: false, reason: 'Mastery was not read.' };

    var yours = nodes.filter(function (n) {
      return n.role === 'yours';
    });

    return {
      scope: (response && response.scope) || pillarId,
      pillarId: pillarId,
      pillarName: (snap.pillar && snap.pillar.name) || pillarId,
      builtAt: (response && response.built_at) || null,
      clustered: snap.clustered !== false,
      stats: snap.stats || {},
      warnings: isArray(snap.warnings) ? snap.warnings : [],
      nodes: nodes,
      byId: byId,
      yours: yours,
      citations: citations,
      cites: cites,
      citedBy: citedBy,
      evidence: evidence,
      topics: topics,
      regions: regions,
      lineage: {
        kind: (snap.lineage && snap.lineage.kind) || 'none',
        note: (snap.lineage && snap.lineage.note) || '',
        paths: lineagePaths,
      },
      suggestions: suggestions,
      suggestionById: suggestionById,
      queueRows: isArray(snap.queue) ? snap.queue : [],
      queue: queueOverlay,
      mastery: masteryOverlay,
      insights: isArray(snap.insights) ? snap.insights : [],
    };
  }

  function push(map, key, value) {
    (map[key] = map[key] || []).push(value);
  }

  function indexById(rows) {
    var out = {};
    (isArray(rows) ? rows : []).forEach(function (row) {
      if (row && row.id !== undefined && row.id !== null) out[row.id] = row;
    });
    return out;
  }

  /**
   * yours | other | suggested | context.
   * A library work from another pillar is "other": it is on this map because it
   * links in, and it is still the learner's. An external work is "suggested"
   * only when the build ranked it for this pillar.
   */
  function roleOf(node, pillarId, suggestionById) {
    if (node.kind === 'library' || node.kind === 'document') {
      var pillars = isArray(node.pillars) ? node.pillars : [];
      return !pillars.length || pillars.indexOf(pillarId) >= 0 ? 'yours' : 'other';
    }
    return suggestionById && suggestionById[node.id] ? 'suggested' : 'context';
  }

  /* --------------------------------------------------------------- visibility */

  function citationsOf(node) {
    return num(node.citation_count);
  }

  /** The works a view draws. Yours always; the rest are capped and ranked. */
  function visibleNodes(model, opts) {
    var options = opts || {};
    var nSuggestions = options.suggestions === undefined ? DEFAULT_SUGGESTIONS : options.suggestions;
    var nContext = options.context === undefined ? DEFAULT_CONTEXT : options.context;
    var force = options.force || {};
    var direction = options.direction || '';
    var out = [];
    var taken = {};
    function add(node) {
      if (node && !taken[node.id]) {
        taken[node.id] = true;
        out.push(node);
      }
    }
    model.nodes.forEach(function (node) {
      if (node.role === 'yours') add(node);
    });
    var linked = model.nodes
      .filter(function (node) {
        return node.role === 'other';
      })
      .sort(rankLocal);
    linked.slice(0, Math.max(nContext, 5)).forEach(add);
    model.suggestions.slice(0, nSuggestions).forEach(function (item) {
      if (direction && item.direction !== direction) return;
      add(model.byId[item.id]);
    });
    model.nodes
      .filter(function (node) {
        return node.role === 'context';
      })
      .sort(rankLocal)
      .slice(0, nContext)
      .forEach(add);
    if (options.lineage) {
      model.lineage.paths.forEach(function (path) {
        (path.condensed || path.nodes || []).forEach(function (id) {
          add(model.byId[id]);
        });
      });
    }
    Object.keys(force).forEach(function (id) {
      if (force[id]) add(model.byId[id]);
    });
    return out;
  }

  function rankLocal(a, b) {
    return (
      (b.local_citations || 0) - (a.local_citations || 0) ||
      (citationsOf(b) || 0) - (citationsOf(a) || 0) ||
      (a.id < b.id ? -1 : 1)
    );
  }

  /* ----------------------------------------------------------------- geometry */

  /** Dot radius in px. Log, so one classic does not eat the page. */
  function radiusFor(citations, role) {
    var base = 4.5 + 3.1 * Math.log10(1 + Math.max(0, num(citations) || 0));
    var r = clamp(base, 4.5, 15);
    if (role === 'context') r = Math.max(3, r - 1.5);
    if (role === 'suggested') r = Math.max(r, 8);
    return r;
  }

  /** Year plus a stable spread inside the year, so same-year dots do not stack. */
  function timelineX(node) {
    var year = num(node.year);
    if (year === null) return null;
    return year + 0.12 + 0.76 * hash01(node.id);
  }

  function yValue(node, mode, model) {
    if (mode === 'local') return Math.log10(1 + (node.local_citations || 0));
    if (mode === 'cohort') {
      var p = num(node.cohort_percentile);
      return p === null ? null : p;
    }
    if (mode === 'topic') {
      var t = num(node.topic);
      if (t === null) return null;
      return t + (hash01('y' + node.id) - 0.5) * 0.7;
    }
    var c = citationsOf(node);
    return c === null ? null : Math.log10(1 + Math.max(0, c));
  }

  /** True when at least one drawn work has a citation count to place it by. */
  function hasCitationCounts(nodes) {
    return nodes.some(function (node) {
      return citationsOf(node) !== null && citationsOf(node) > 0;
    });
  }

  /** Min and max of values, padded; a single value gets a usable span. */
  function extent(values, padFraction) {
    var xs = values.filter(function (v) {
      return v !== null && v !== undefined && isFinite(v);
    });
    if (!xs.length) return [0, 1];
    var low = Math.min.apply(null, xs);
    var high = Math.max.apply(null, xs);
    if (low === high) {
      low -= 0.5;
      high += 0.5;
    }
    var pad = (high - low) * (padFraction === undefined ? 0.06 : padFraction);
    return [low - pad, high + pad];
  }

  /** "Nice" integer years for the time axis. */
  function yearTicks(low, high, maxTicks) {
    var span = Math.max(1, high - low);
    var steps = [1, 2, 5, 10, 20, 25, 50, 100];
    var step = steps[steps.length - 1];
    for (var i = 0; i < steps.length; i++) {
      if (span / steps[i] <= (maxTicks || 8)) {
        step = steps[i];
        break;
      }
    }
    var out = [];
    for (var y = Math.ceil(low / step) * step; y <= high; y += step) out.push(y);
    return out;
  }

  /** Citation tick values on the log axis, as counts. */
  function citationTicks(maxLog) {
    var out = [];
    for (var power = 0; power <= Math.ceil(maxLog) + 1; power++) {
      var count = power === 0 ? 0 : Math.pow(10, power);
      out.push({ count: count, value: Math.log10(1 + count) });
    }
    return out.filter(function (t) {
      return t.value <= maxLog + 0.4;
    });
  }

  function compactCount(count) {
    if (count >= 1e6) return trimZero((count / 1e6).toFixed(1)) + 'M';
    if (count >= 1e3) return trimZero((count / 1e3).toFixed(1)) + 'k';
    return String(count);
  }

  function trimZero(text) {
    return text.replace(/\.0$/, '');
  }

  /* -------------------------------------------------------------------- labels */

  /**
   * Greedy label placement. `items` come in priority order with a measured
   * `w`/`h` and an anchor (x, y, r); the first label that does not touch an
   * earlier one or a dot is kept. Pure, so the collision rule is testable.
   */
  function placeLabels(items, bounds) {
    var placed = [];
    var out = [];
    var dots = items.map(function (item) {
      return { x: item.x, y: item.y, r: item.r };
    });
    items.forEach(function (item, index) {
      var candidates = [
        { x: item.x - item.w / 2, y: item.y + item.r + 2 },
        { x: item.x - item.w / 2, y: item.y - item.r - 2 - item.h },
        { x: item.x + item.r + 4, y: item.y - item.h / 2 },
        { x: item.x - item.r - 4 - item.w, y: item.y - item.h / 2 },
      ];
      for (var c = 0; c < candidates.length; c++) {
        var box = { x: candidates[c].x, y: candidates[c].y, w: item.w, h: item.h };
        if (bounds && !inside(box, bounds)) continue;
        if (placed.some(function (other) { return overlaps(box, other); })) continue;
        if (dots.some(function (dot, di) { return di !== index && hitsDot(box, dot); })) continue;
        placed.push(box);
        out.push({ id: item.id, x: box.x, y: box.y, w: box.w, h: box.h });
        return;
      }
    });
    return out;
  }

  function overlaps(a, b) {
    return a.x < b.x + b.w + 2 && b.x < a.x + a.w + 2 && a.y < b.y + b.h + 1 && b.y < a.y + a.h + 1;
  }

  function hitsDot(box, dot) {
    var nx = clamp(dot.x, box.x, box.x + box.w);
    var ny = clamp(dot.y, box.y, box.y + box.h);
    var dx = dot.x - nx;
    var dy = dot.y - ny;
    return dx * dx + dy * dy < dot.r * dot.r;
  }

  function inside(box, bounds) {
    return box.x >= bounds.x && box.y >= bounds.y && box.x + box.w <= bounds.x + bounds.w && box.y + box.h <= bounds.y + bounds.h;
  }

  /** Label order: what the learner asked about first, then what matters most. */
  function labelPriority(node, state) {
    var score = 0;
    if (state.selectedId === node.id) score += 1000;
    if (state.hoverId === node.id) score += 900;
    if (state.matches && state.matches[node.id]) score += 500;
    if (node.role === 'yours') score += 300;
    if (node.role === 'suggested') score += 200 - (node.position || 0);
    if (node.on_lineage && state.lineage) score += 120;
    if (state.neighbours && state.neighbours[node.id]) score += 150;
    score += Math.min(60, Math.log10(1 + (node.local_citations || 0)) * 20);
    return score;
  }

  function nodeCaption(node, expanded) {
    var label = node.label || shortTitle(node.title, 40) || node.id;
    if (!expanded) return label;
    var title = node.title && node.title !== label ? ' – ' + shortTitle(node.title, 56) : '';
    return label + title;
  }

  function shortTitle(title, limit) {
    var text = String(title || '').trim();
    if (text.length <= limit) return text;
    return text.slice(0, limit - 1).trimEnd() + '…';
  }

  /* -------------------------------------------------------------- explanations */

  function labelOf(model, id) {
    var node = model.byId[id];
    if (!node) return id;
    return node.label || shortTitle(node.title, 40) || id;
  }

  function percentTop(percentile) {
    var top = Math.max(100 - percentile, 0.1);
    return top >= 1 ? Math.round(top) + '%' : top.toFixed(1) + '%';
  }

  /**
   * Why a suggestion is on the list, as short clauses, the counts first
   * (Local Citation Network style). Each clause comes from a stored number.
   */
  function suggestionReasons(model, item) {
    var out = [];
    var cited = item.cited_by_yours || [];
    var cites = item.cites_yours || [];
    if (cited.length) out.push({ text: 'Cited by ' + plural(cited.length, 'paper') + ' of yours', ids: cited });
    if (cites.length) out.push({ text: 'Cites ' + plural(cites.length, 'paper') + ' of yours', ids: cites });
    if (item.shared_references && item.shared_with) {
      out.push({
        text: 'Shares ' + plural(item.shared_references, 'reference') + ' with ' + labelOf(model, item.shared_with),
        ids: [item.shared_with],
      });
    }
    var percentile = num(item.text_percentile);
    if (percentile !== null && percentile >= 80) {
      out.push({ text: 'Abstract in the top ' + percentTop(percentile) + ' most similar pairs on this map', ids: [] });
    }
    var topic = item.topic !== undefined && item.topic !== null ? model.topics[item.topic] : null;
    if (topic && topic.terms && topic.terms.length) {
      out.push({ text: 'In the topic “' + topic.terms.slice(0, 3).join(', ') + '”', ids: [] });
    }
    if (!out.length) out.push({ text: 'Ranked by link structure and abstract similarity together', ids: [] });
    return out;
  }

  function oneLineReason(model, item) {
    var reasons = suggestionReasons(model, item);
    return reasons
      .slice(0, 2)
      .map(function (r) {
        return r.text;
      })
      .join(' · ');
  }

  /** What the panel says about one work, in order. */
  function connectionsFor(model, id) {
    var rows = model.evidence[id] || [];
    var shown = rows.slice(0, EVIDENCE_SHOWN);
    var node = model.byId[id];
    var yoursOnly = shown.filter(function (row) {
      var other = model.byId[row.other];
      return other && other.role !== 'context' && other.role !== 'suggested';
    });
    var cites = (model.cites[id] || []).map(function (other) {
      return model.byId[other];
    });
    var citedBy = (model.citedBy[id] || []).map(function (other) {
      return model.byId[other];
    });
    var mineCites = cites.filter(isYours);
    var mineCitedBy = citedBy.filter(isYours);
    return {
      node: node,
      evidence: shown,
      withYours: yoursOnly,
      more: Math.max(0, rows.length - shown.length),
      citesYours: mineCites,
      citedByYours: mineCitedBy,
      citesCount: cites.length,
      citedByCount: citedBy.length,
    };
  }

  function isYours(node) {
    return !!node && (node.role === 'yours' || node.role === 'other');
  }

  /** Where a work sits on the main line, or null. */
  function lineagePosition(model, id) {
    for (var p = 0; p < model.lineage.paths.length; p++) {
      var path = model.lineage.paths[p];
      var steps = path.condensed && path.condensed.length ? path.condensed : path.nodes || [];
      var at = steps.indexOf(id);
      if (at >= 0) return { path: p, step: at + 1, of: steps.length };
    }
    return null;
  }

  /** The condensed paths as ordered works, for the overlay and its panel. */
  function lineageSteps(model) {
    return model.lineage.paths.map(function (path) {
      var ids = path.condensed && path.condensed.length ? path.condensed : path.nodes || [];
      var libraryIds = {};
      (path.library || []).forEach(function (id) {
        libraryIds[id] = true;
      });
      return ids
        .filter(function (id) {
          return !!model.byId[id];
        })
        .map(function (id) {
          return { id: id, label: labelOf(model, id), year: model.byId[id].year, yours: !!libraryIds[id] };
        });
    });
  }

  /** Arcs the overlay draws: consecutive works on a path, and whether the arc is a direct citation. */
  function lineageArcs(model) {
    var out = [];
    var cited = {};
    model.citations.forEach(function (c) {
      cited[c.cited + '>' + c.citing] = true;
    });
    lineageSteps(model).forEach(function (steps, pathIndex) {
      for (var i = 0; i + 1 < steps.length; i++) {
        var from = steps[i].id;
        var to = steps[i + 1].id;
        out.push({ from: from, to: to, direct: !!cited[from + '>' + to], path: pathIndex });
      }
    });
    return out;
  }

  /* ------------------------------------------------------------------ insights */

  var INSIGHT_ORDER = ['most_aligned', 'furthest', 'bridge', 'cross_pillar', 'gap', 'sparse'];
  var INSIGHT_TITLE = {
    most_aligned: 'Most aligned pair',
    furthest: 'Furthest from the rest',
    bridge: 'Links communities',
    cross_pillar: 'Crosses pillars',
    gap: 'A gap on the map',
    sparse: 'A sparse map',
  };

  /** Up to `limit` cards: the build's insights, then the top suggestion and the queue. */
  function insightCards(model, limit) {
    var cards = [];
    var seen = {};
    INSIGHT_ORDER.forEach(function (kind) {
      model.insights.forEach(function (insight) {
        if (insight.kind !== kind || seen[kind]) return;
        seen[kind] = true;
        cards.push({
          kind: kind,
          title: INSIGHT_TITLE[kind] || kind,
          text: insight.text || '',
          ids: isArray(insight.nodes) ? insight.nodes : [],
        });
      });
    });
    var top = model.suggestions[0];
    if (top && model.byId[top.id]) {
      cards.push({
        kind: 'top_suggestion',
        title: 'Top suggestion',
        text: (model.byId[top.id].title || labelOf(model, top.id)) + '. ' + oneLineReason(model, top),
        ids: [top.id],
      });
    }
    var counts = queueCounts(model);
    if (counts.total) {
      cards.push({
        kind: 'queue_check',
        title: 'Discovery queue check',
        text:
          plural(counts.total, 'paper') +
          ' queued: ' +
          counts.close +
          ' close to yours, ' +
          counts.offTopic +
          ' off topic.',
        ids: [],
        tab: 'queue',
      });
    }
    return cards.slice(0, limit || 4);
  }

  /* --------------------------------------------------------------------- queue */

  function verdictInfo(verdict) {
    return VERDICTS[verdict] || { label: verdict || 'Not scored', tone: 'neutral', order: 99 };
  }

  function queueCounts(model) {
    var rows = model.queueRows;
    var counts = { total: rows.length, close: 0, offTopic: 0 };
    rows.forEach(function (row) {
      if (row.verdict === 'close_strong' || row.verdict === 'close_related') counts.close += 1;
      if (row.verdict === 'off_topic') counts.offTopic += 1;
    });
    return counts;
  }

  /**
   * The queue as the learner has it now. The snapshot supplies each row's
   * verdict; the live overlay supplies what changed since the build. A paper
   * queued after the build has no verdict, and says so rather than guessing.
   */
  function queueView(model) {
    var overlay = model.queue || {};
    var snapshotRows = model.queueRows.slice();
    snapshotRows.sort(function (a, b) {
      return verdictInfo(a.verdict).order - verdictInfo(b.verdict).order || (a.rank || 1e9) - (b.rank || 1e9);
    });
    var liveById = {};
    if (overlay.available && isArray(overlay.items)) {
      overlay.items.forEach(function (item) {
        liveById[item.queue_id] = item;
      });
    }
    var rows = snapshotRows.map(function (row) {
      var live = liveById[row.queue_id];
      return {
        queue_id: row.queue_id,
        node: row.node,
        title: row.title || (live && live.title) || '',
        verdict: row.verdict,
        info: verdictInfo(row.verdict),
        processed: live ? !!live.processed : !!row.processed,
        gone: overlay.available && !live,
        scored: true,
        reason: queueReason(row),
      };
    });
    var fresh = [];
    if (overlay.available && isArray(overlay.items)) {
      overlay.items.forEach(function (item) {
        if (!item.in_snapshot) {
          fresh.push({
            queue_id: item.queue_id,
            node: null,
            title: item.title || '',
            verdict: null,
            info: { label: 'Queued since the build', tone: 'neutral', order: 99 },
            processed: !!item.processed,
            gone: false,
            scored: false,
            reason: 'Not scored yet. Build the map again to score it.',
          });
        }
      });
    }
    return {
      available: overlay.available !== false,
      reason: overlay.reason || '',
      rows: rows.concat(fresh),
      removed: overlay.available && isArray(overlay.removed_since_build) ? overlay.removed_since_build : [],
    };
  }

  function queueReason(row) {
    var bits = [];
    if (row.duplicate_of) bits.push('Looks like a paper you already have');
    var cited = row.cited_by_yours || [];
    var cites = row.cites_yours || [];
    if (cited.length) bits.push('cited by ' + plural(cited.length, 'paper') + ' of yours');
    if (cites.length) bits.push('cites ' + plural(cites.length, 'paper') + ' of yours');
    if (row.shared_references) bits.push('shares ' + plural(row.shared_references, 'reference') + ' with yours');
    if (!bits.length && typeof row.rank_fraction === 'number') {
      bits.push('ranked in the top ' + Math.max(1, Math.round(row.rank_fraction * 100)) + '% of the pool');
    }
    if (!bits.length && row.verdict === 'off_topic') bits.push('No link to your papers and far from the pillar’s goal');
    if (!bits.length && row.verdict === 'on_topic_new_direction') bits.push('Near the pillar’s goal, not near your papers yet');
    var text = bits.join(', ');
    return text ? text.charAt(0).toUpperCase() + text.slice(1) + '.' : '';
  }

  /* ------------------------------------------------------------------- "+" */

  /**
   * The request behind "+": the M6 route. Only the work id is sent; the server
   * resolves it to a paper with a PDF, and refuses (422) when it cannot.
   */
  function queueRequest(pillarId, node) {
    return {
      url: '/api/pillars/' + encodeURIComponent(pillarId) + '/queue/external',
      options: {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ work_id: node.id, title: node.title || '' }),
      },
    };
  }

  function interpretQueueResponse(status, body) {
    var text = detail(body);
    if (status >= 200 && status < 300) {
      return { kind: 'queued', message: 'Queued for Add papers. Nothing has been processed or spent.' };
    }
    if (status === 409) {
      return { kind: 'queued', message: text || 'Already in this pillar’s queue.' };
    }
    if (status === 422) {
      return {
        kind: 'no_pdf',
        message: text || 'No open PDF is known for this paper, so it cannot be queued. Upload it from the pillar page instead.',
      };
    }
    if (status === 404 || status === 405) {
      return { kind: 'unavailable', message: 'Queueing from the map is not available on this server yet.' };
    }
    return { kind: 'error', message: text || 'Could not queue the paper (HTTP ' + status + ').' };
  }

  /* ---------------------------------------------------------------- list sort */

  var LIST_COLUMNS = {
    title: function (n) { return (n.title || n.label || '').toLowerCase(); },
    year: function (n) { return num(n.year); },
    citations: function (n) { return citationsOf(n); },
    local: function (n) { return n.local_citations || 0; },
    role: function (n) { return roleSortKey(n); },
    topic: function (n) { return num(n.topic); },
  };

  function roleSortKey(node) {
    if (node.role === 'yours') return 0;
    if (node.role === 'other') return 1;
    if (node.role === 'suggested') return 2 + (node.position || 0) / 100;
    return 3;
  }

  /** Stable sort; a missing value always sorts last, in either direction. */
  function sortRows(nodes, key, direction) {
    var pick = LIST_COLUMNS[key] || LIST_COLUMNS.role;
    var sign = direction === 'desc' ? -1 : 1;
    return nodes
      .map(function (node, index) {
        return { node: node, index: index, value: pick(node) };
      })
      .sort(function (a, b) {
        var av = a.value;
        var bv = b.value;
        if (av === null && bv === null) return a.index - b.index;
        if (av === null) return 1;
        if (bv === null) return -1;
        if (av < bv) return -sign;
        if (av > bv) return sign;
        return a.index - b.index;
      })
      .map(function (row) {
        return row.node;
      });
  }

  function roleLabel(node) {
    if (node.role === 'yours') return node.kind === 'document' ? 'Your document' : 'In your library';
    if (node.role === 'other') return 'Yours, another pillar';
    if (node.role === 'suggested') return 'Suggested #' + node.position;
    return 'Connected paper';
  }

  /* -------------------------------------------------------------------- search */

  function findMatches(nodes, query) {
    var terms = String(query || '')
      .toLowerCase()
      .split(/\s+/)
      .filter(Boolean);
    var out = {};
    if (!terms.length) return out;
    nodes.forEach(function (node) {
      var hay = [node.title, node.label, node.authors.join(' '), node.year].join(' ').toLowerCase();
      if (
        terms.every(function (t) {
          return hay.indexOf(t) >= 0;
        })
      ) {
        out[node.id] = true;
      }
    });
    return out;
  }

  /* ----------------------------------------------------------------- citations */

  /**
   * Whether the snapshot can say anything about citations. Zero lines on a map
   * is a statement about data, not about the papers, and the page must say so.
   */
  function citationAvailability(model) {
    var lines = model.citations.length;
    var counts = hasCitationCounts(model.nodes);
    var unknownIds = model.stats.unknown_ids || 0;
    return {
      lines: lines,
      counts: counts,
      unavailable: lines === 0,
      message:
        lines === 0
          ? 'Citation data is unavailable for this pillar, so the map has no citation lines. It is not a map of papers that cite nothing. Positions come from abstract similarity alone.'
          : '',
      countsMessage: counts
        ? ''
        : 'No work on this map has a citation count, so the vertical axis cannot show citations.',
      unknownIds: unknownIds,
    };
  }

  function formatBuiltAt(value) {
    if (!value) return '';
    var date = new Date(value);
    if (isNaN(date.getTime())) return '';
    return date.toISOString().slice(0, 10);
  }

  function masteryText(model, node) {
    var overlay = model.mastery;
    if (!overlay || overlay.available === false) {
      return { known: false, text: 'Recall is unavailable' + (overlay && overlay.reason ? ': ' + overlay.reason : '.') };
    }
    var ids = node.paper_ids || [];
    var values = [];
    ids.forEach(function (id) {
      var row = overlay.papers && overlay.papers[id];
      if (row && typeof row.retrievability === 'number') values.push(row.retrievability);
    });
    if (!values.length) return { known: false, text: 'No reviewed cards yet' };
    var mean = values.reduce(function (a, b) { return a + b; }, 0) / values.length;
    return { known: true, value: mean, text: Math.round(mean * 100) + '% recall' };
  }

  var api = {
    VIEWS: VIEWS,
    DEFAULT_VIEW: DEFAULT_VIEW,
    SUGGESTION_CHOICES: SUGGESTION_CHOICES,
    DEFAULT_SUGGESTIONS: DEFAULT_SUGGESTIONS,
    CONTEXT_CHOICES: CONTEXT_CHOICES,
    DEFAULT_CONTEXT: DEFAULT_CONTEXT,
    CLUSTER_CONTEXT: CLUSTER_CONTEXT,
    Y_AXES: Y_AXES,
    VERDICTS: VERDICTS,
    DIRECTIONS: DIRECTIONS,
    parseQuery: parseQuery,
    buildQuery: buildQuery,
    paperHref: paperHref,
    mapHref: mapHref,
    classify: classify,
    buildCommand: buildCommand,
    prepare: prepare,
    roleOf: roleOf,
    visibleNodes: visibleNodes,
    radiusFor: radiusFor,
    timelineX: timelineX,
    yValue: yValue,
    hasCitationCounts: hasCitationCounts,
    extent: extent,
    yearTicks: yearTicks,
    citationTicks: citationTicks,
    compactCount: compactCount,
    placeLabels: placeLabels,
    labelPriority: labelPriority,
    nodeCaption: nodeCaption,
    shortTitle: shortTitle,
    labelOf: labelOf,
    suggestionReasons: suggestionReasons,
    oneLineReason: oneLineReason,
    connectionsFor: connectionsFor,
    lineagePosition: lineagePosition,
    lineageSteps: lineageSteps,
    lineageArcs: lineageArcs,
    insightCards: insightCards,
    verdictInfo: verdictInfo,
    queueCounts: queueCounts,
    queueView: queueView,
    queueRequest: queueRequest,
    interpretQueueResponse: interpretQueueResponse,
    sortRows: sortRows,
    roleLabel: roleLabel,
    findMatches: findMatches,
    citationAvailability: citationAvailability,
    formatBuiltAt: formatBuiltAt,
    masteryText: masteryText,
    plural: plural,
    percentTop: percentTop,
    hash01: hash01,
  };
  global.PaperMapCore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
