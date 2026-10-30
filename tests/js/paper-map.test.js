/*
 * Pure helpers behind the paper map page. The canvas, the zoom and the panel
 * need a DOM and are checked in a browser; these lock the rules they depend on:
 * what state the page is in, which works are drawn, how a connection is worded
 * and what "+" sends.
 */
const { test } = require('node:test');
const assert = require('node:assert');

const core = require('../../webui/static/paper-map-core.js');

const PILLAR = 'model-interpretability';

function node(id, extra) {
  return Object.assign(
    {
      id,
      kind: 'external',
      label: id.toUpperCase() + ' 2020',
      title: 'Title of ' + id,
      year: 2020,
      authors: [],
      paper_ids: [],
      pillars: [],
      citation_count: 10,
      local_citations: 0,
      topic: 0,
      region: 0,
      x: 0,
      y: 0,
      on_lineage: false,
      queue_ids: [],
    },
    extra || {}
  );
}

function response(overrides) {
  const snapshot = {
    pillar: { id: PILLAR, name: 'Model interpretability' },
    clustered: true,
    stats: { citations: 3 },
    nodes: [
      node('mine1', { kind: 'library', pillars: [PILLAR], paper_ids: ['arxiv:1'], year: 2020, citation_count: 100, local_citations: 2 }),
      node('mine2', { kind: 'library', pillars: [PILLAR], paper_ids: ['arxiv:2'], year: 2021, citation_count: 40, local_citations: 1 }),
      node('doc', { kind: 'document', pillars: [PILLAR], paper_ids: ['file:abc'], year: null, citation_count: null }),
      node('away', { kind: 'library', pillars: ['ml-systems-production'], paper_ids: ['arxiv:9'] }),
      node('s1', { year: 2015, citation_count: 5000, local_citations: 3 }),
      node('s2', { year: 2023, citation_count: 12, local_citations: 1 }),
      node('c1', { year: 2018, citation_count: 900, local_citations: 5 }),
      node('c2', { year: 2019, citation_count: 50, local_citations: 0 }),
      node('c3', { year: 2017, citation_count: 70, local_citations: 2 }),
    ],
    citations: [
      { citing: 'mine1', cited: 's1', spc: 0.5, influential: true, intents: ['methodology'] },
      { citing: 's2', cited: 'mine1', spc: 0.2, influential: false, intents: [] },
      { citing: 'mine2', cited: 'mine1', spc: 0.4, influential: false, intents: [] },
      { citing: 'ghost', cited: 'mine1', spc: 0, influential: false, intents: [] },
    ],
    communities: {
      topics: [{ id: 0, size: 9, terms: ['probing', 'bert', 'layers'], representative: 'mine1' }],
      regions: [{ id: 0, size: 9, terms: ['probing'], representative: 'mine1' }],
    },
    lineage: {
      kind: 'key_node',
      note: 'A likely line of development.',
      paths: [{ nodes: ['s1', 'c3', 'mine1', 's2'], library: ['mine1'], condensed: ['s1', 'mine1', 's2'] }],
    },
    suggestions: {
      pillar: [
        { id: 's1', rank: 1, direction: 'foundation', cited_by_yours: ['mine1'], cites_yours: [], shared_references: 0, shared_with: null, text_percentile: 97.5, topic: 0 },
        { id: 's2', rank: 2, direction: 'frontier', cited_by_yours: [], cites_yours: ['mine1', 'mine2'], shared_references: 4, shared_with: 'mine2', text_percentile: 60, topic: 0 },
      ],
      communities: [],
    },
    queue: [
      { queue_id: 'q-off', node: 'c2', title: 'Dolphin clicks', verdict: 'off_topic', processed: false, rank: 30, rank_fraction: 0.9 },
      { queue_id: 'q-strong', node: 's1', title: 'A close one', verdict: 'close_strong', processed: false, rank: 1, rank_fraction: 0.01, cited_by_yours: ['mine1'] },
      { queue_id: 'q-gone', node: 'c3', title: 'Removed later', verdict: 'on_topic_new_direction', processed: false },
    ],
    insights: [
      { kind: 'gap', text: '12 works gather around X; none of them are yours.', nodes: ['c1'] },
      { kind: 'most_aligned', text: 'Your most closely connected pair: A and B.', nodes: ['mine1', 'mine2'] },
    ],
    evidence: [
      { a: 'mine1', b: 'mine2', weight: 0.9, sentence: 'They share 5 references.', evidence: {} },
      { a: 'mine1', b: 's1', weight: 1.4, sentence: 'Mine1 cites S1 (builds on).', evidence: {} },
      { a: 'nowhere', b: 's1', weight: 9, sentence: 'dropped: not on the map', evidence: {} },
    ],
    warnings: [],
  };
  return Object.assign(
    {
      scope: PILLAR,
      status: 'ready',
      built_at: '2026-10-30T10:00:00+00:00',
      snapshot,
      overlays: {
        queue: {
          available: true,
          items: [
            { queue_id: 'q-off', title: 'Dolphin clicks', processed: false, in_snapshot: true },
            { queue_id: 'q-strong', title: 'A close one', processed: true, in_snapshot: true },
            { queue_id: 'q-new', title: 'Queued last night', processed: false, in_snapshot: false },
          ],
          removed_since_build: [{ queue_id: 'q-gone', title: 'Removed later' }],
        },
        mastery: {
          available: true,
          papers: { 'arxiv:1': { retrievability: 0.8 }, 'arxiv:2': { retrievability: null } },
        },
      },
    },
    overrides || {}
  );
}

test('classify keeps a failed load apart from an empty map', () => {
  assert.equal(core.classify(200, response()).kind, 'ready');
  assert.equal(core.classify(200, { scope: PILLAR, status: 'not_built', message: 'm' }).kind, 'not_built');
  assert.equal(core.classify(200, { scope: PILLAR, status: 'empty' }).kind, 'empty');

  const failed = core.classify(503, { detail: 'Could not read the map: boom' });
  assert.equal(failed.kind, 'error');
  assert.match(failed.message, /does not mean the map is empty/);

  const network = core.classify(0, null, new Error('Failed to fetch'));
  assert.equal(network.kind, 'error');
  assert.match(network.message, /Failed to fetch/);
  assert.match(network.message, /does not mean the map is empty/);

  // ready without a snapshot is a broken answer, never an empty map
  assert.equal(core.classify(200, { status: 'ready', snapshot: null }).kind, 'error');
  assert.equal(core.classify(200, null).kind, 'error');
  assert.equal(core.classify(404, { detail: "No pillar with id 'x'" }).kind, 'error');
});

test('classify names the migration when the tables are missing', () => {
  const state = core.classify(503, { detail: 'Apply docs/migrations/034_map_snapshots.sql by hand.' });
  assert.equal(state.kind, 'migration');
  assert.equal(state.migration, 'docs/migrations/034_map_snapshots.sql');
});

test('not built carries the command that builds it', () => {
  const state = core.classify(200, { scope: PILLAR, status: 'not_built', message: 'm' });
  assert.equal(state.command, 'python -m nlp_pillars.cli map build --pillar ' + PILLAR);
});

test('prepare assigns roles, suggestion numbers and drops dangling links', () => {
  const model = core.prepare(response(), PILLAR);
  const role = (id) => model.byId[id].role;
  assert.equal(role('mine1'), 'yours');
  assert.equal(role('doc'), 'yours');
  assert.equal(role('away'), 'other');
  assert.equal(role('s1'), 'suggested');
  assert.equal(role('c1'), 'context');
  assert.equal(model.byId.s1.position, 1);
  assert.equal(model.byId.s2.position, 2);
  assert.equal(model.citations.length, 3, 'the citation from an unknown work is dropped');
  assert.equal(model.evidence.nowhere, undefined);
  assert.equal(model.yours.length, 3);
});

test('prepare tolerates a payload with every optional part missing', () => {
  const model = core.prepare({ status: 'ready', snapshot: { nodes: [node('a')] } }, PILLAR);
  assert.equal(model.nodes.length, 1);
  assert.equal(model.suggestions.length, 0);
  assert.equal(model.queue.available, false);
  assert.equal(model.mastery.available, false);
  assert.equal(model.clustered, true);
});

test('visibleNodes always keeps yours and caps the rest', () => {
  const model = core.prepare(response(), PILLAR);
  const ids = (list) => list.map((n) => n.id).sort();

  const none = core.visibleNodes(model, { suggestions: 0, context: 0 });
  assert.deepEqual(ids(none), ['away', 'doc', 'mine1', 'mine2']);

  const one = core.visibleNodes(model, { suggestions: 1, context: 0 });
  assert.ok(ids(one).includes('s1'));
  assert.ok(!ids(one).includes('s2'));

  const withContext = core.visibleNodes(model, { suggestions: 0, context: 2 });
  assert.deepEqual(ids(withContext).filter((id) => id.startsWith('c')), ['c1', 'c3'], 'the most-cited-on-this-map come first');

  const forced = core.visibleNodes(model, { suggestions: 0, context: 0, force: { c2: true, nope: true } });
  assert.ok(ids(forced).includes('c2'));

  const foundations = core.visibleNodes(model, { suggestions: 20, context: 0, direction: 'foundation' });
  assert.ok(ids(foundations).includes('s1'));
  assert.ok(!ids(foundations).includes('s2'));
});

test('the lineage overlay keeps its works on the map even when capped out', () => {
  const model = core.prepare(response(), PILLAR);
  const without = core.visibleNodes(model, { suggestions: 0, context: 0 });
  assert.ok(!without.some((n) => n.id === 's1'));
  const withLine = core.visibleNodes(model, { suggestions: 0, context: 0, lineage: true });
  assert.ok(withLine.some((n) => n.id === 's1'));
  assert.ok(withLine.some((n) => n.id === 's2'));
});

test('radiusFor grows with the log of citations and stays bounded', () => {
  assert.equal(core.radiusFor(null, 'yours'), 4.5);
  assert.ok(core.radiusFor(1000, 'yours') > core.radiusFor(10, 'yours'));
  assert.equal(core.radiusFor(1e12, 'yours'), 15);
  assert.ok(core.radiusFor(0, 'suggested') >= 8, 'a rank number has to fit inside');
  assert.ok(core.radiusFor(0, 'context') < 4.5);
});

test('timelineX puts a work inside its year, and undated works nowhere', () => {
  const x = core.timelineX(node('a', { year: 2019 }));
  assert.ok(x > 2019 && x < 2020);
  assert.equal(x, core.timelineX(node('a', { year: 2019 })), 'stable');
  assert.notEqual(core.timelineX(node('b', { year: 2019 })), x, 'same-year dots spread');
  assert.equal(core.timelineX(node('c', { year: null })), null);
});

test('yValue: unknown stays unknown instead of becoming zero', () => {
  const model = core.prepare(response(), PILLAR);
  assert.equal(core.yValue(model.byId.doc, 'citations', model), null);
  assert.equal(core.yValue(model.byId.mine1, 'citations', model), Math.log10(101));
  assert.equal(core.yValue(model.byId.mine1, 'local', model), Math.log10(3));
  assert.equal(core.yValue(node('p', { cohort_percentile: null }), 'cohort', model), null);
  assert.equal(core.yValue(node('p', { cohort_percentile: 80 }), 'cohort', model), 80);
});

test('hasCitationCounts is false when every count is missing or zero', () => {
  assert.equal(core.hasCitationCounts([node('a', { citation_count: null }), node('b', { citation_count: 0 })]), false);
  assert.equal(core.hasCitationCounts([node('a', { citation_count: 3 })]), true);
});

test('extent pads, and gives a single value a usable span', () => {
  assert.deepEqual(core.extent([]), [0, 1]);
  const [low, high] = core.extent([5, 5]);
  assert.ok(low < 5 && high > 5);
  const [a, b] = core.extent([0, 10], 0.1);
  assert.equal(a, -1);
  assert.equal(b, 11);
  assert.deepEqual(core.extent([null, 4, undefined, 6], 0), [4, 6]);
});

test('yearTicks stay few and land on round years', () => {
  const ticks = core.yearTicks(1998.4, 2026.2, 8);
  assert.ok(ticks.length <= 9 && ticks.length >= 3);
  ticks.forEach((t) => assert.equal(t % 5, 0));
  assert.deepEqual(core.yearTicks(2020.2, 2023.9, 8), [2021, 2022, 2023]);
});

test('compactCount reads like an axis label', () => {
  assert.equal(core.compactCount(0), '0');
  assert.equal(core.compactCount(950), '950');
  assert.equal(core.compactCount(1000), '1k');
  assert.equal(core.compactCount(12500), '12.5k');
  assert.equal(core.compactCount(2000000), '2M');
});

test('placeLabels skips a label that would sit on another or on a dot', () => {
  const items = [
    { id: 'a', x: 100, y: 100, r: 6, w: 60, h: 12 },
    { id: 'b', x: 110, y: 104, r: 6, w: 60, h: 12 }, // sits on a's label
    { id: 'c', x: 300, y: 100, r: 6, w: 60, h: 12 },
  ];
  const placed = core.placeLabels(items, { x: 0, y: 0, w: 400, h: 300 });
  const ids = placed.map((p) => p.id);
  assert.ok(ids.includes('a'));
  assert.ok(ids.includes('c'));
  const a = placed.find((p) => p.id === 'a');
  const b = placed.find((p) => p.id === 'b');
  if (b) {
    const apart = a.x + a.w + 2 <= b.x || b.x + b.w + 2 <= a.x || a.y + a.h + 1 <= b.y || b.y + b.h + 1 <= a.y;
    assert.ok(apart, 'placed labels never overlap');
  }
});

test('placeLabels keeps labels inside the canvas', () => {
  const placed = core.placeLabels([{ id: 'edge', x: 4, y: 4, r: 5, w: 80, h: 12 }], { x: 0, y: 0, w: 400, h: 300 });
  placed.forEach((p) => {
    assert.ok(p.x >= 0 && p.y >= 0 && p.x + p.w <= 400 && p.y + p.h <= 300);
  });
});

test('labelPriority puts the selection first, then yours, then suggestions by rank', () => {
  const model = core.prepare(response(), PILLAR);
  const state = { selectedId: 'c1', lineage: false };
  const p = (id) => core.labelPriority(model.byId[id], state);
  assert.ok(p('c1') > p('mine1'));
  assert.ok(p('mine1') > p('s1'));
  assert.ok(p('s1') > p('s2'));
  assert.ok(p('s2') > p('c2'));
});

test('suggestionReasons counts first and never invents a reason', () => {
  const model = core.prepare(response(), PILLAR);
  const first = core.suggestionReasons(model, model.suggestionById.s1);
  assert.equal(first[0].text, 'Cited by 1 paper of yours');
  assert.deepEqual(first[0].ids, ['mine1']);
  assert.ok(first.some((r) => /top 3% most similar/.test(r.text) || /top 2\.5%/.test(r.text) || /top 3%/.test(r.text)));

  const second = core.suggestionReasons(model, model.suggestionById.s2);
  assert.equal(second[0].text, 'Cites 2 papers of yours');
  assert.ok(second.some((r) => r.text === 'Shares 4 references with MINE2 2020'));
  assert.ok(!second.some((r) => /most similar/.test(r.text)), 'a 60th percentile is not worth a sentence');

  const bare = core.suggestionReasons(model, { id: 'x' });
  assert.equal(bare.length, 1);
  assert.match(bare[0].text, /link structure and abstract similarity/);
});

test('connectionsFor lists the strongest evidence first and counts the rest', () => {
  const model = core.prepare(response(), PILLAR);
  const c = core.connectionsFor(model, 'mine1');
  assert.deepEqual(c.evidence.map((e) => e.other), ['s1', 'mine2']);
  assert.deepEqual(c.withYours.map((e) => e.other), ['mine2']);
  assert.equal(c.more, 0);
  assert.equal(c.citedByCount, 2);
  assert.deepEqual(c.citedByYours.map((n) => n.id), ['mine2']);
  assert.equal(c.citesCount, 1);
  assert.deepEqual(c.citesYours, [], 's1 is a suggestion, not yours');
});

test('lineage helpers follow the condensed path and mark which hops are citations', () => {
  const model = core.prepare(response(), PILLAR);
  assert.deepEqual(core.lineagePosition(model, 'mine1'), { path: 0, step: 2, of: 3 });
  assert.equal(core.lineagePosition(model, 'c3'), null, 'condensed away');
  assert.equal(core.lineagePosition(model, 'c1'), null);
  const steps = core.lineageSteps(model);
  assert.deepEqual(steps[0].map((s) => s.id), ['s1', 'mine1', 's2']);
  assert.deepEqual(steps[0].map((s) => s.yours), [false, true, false]);
  const arcs = core.lineageArcs(model);
  // s1 <- mine1 is a stored citation (mine1 cites s1); mine1 <- s2 is one too
  assert.deepEqual(arcs.map((a) => a.direct), [true, true]);
});

test('insightCards: the build first, then the top suggestion and the queue', () => {
  const model = core.prepare(response(), PILLAR);
  const cards = core.insightCards(model, 6);
  assert.deepEqual(cards.map((c) => c.kind), ['most_aligned', 'gap', 'top_suggestion', 'queue_check']);
  assert.match(cards[2].text, /Title of s1/);
  assert.match(cards[3].text, /3 papers queued: 1 close to yours, 1 off topic/);
  assert.equal(core.insightCards(model, 2).length, 2);
});

test('queueView orders by fit, keeps everything, and flags what changed since the build', () => {
  const model = core.prepare(response(), PILLAR);
  const view = core.queueView(model);
  assert.equal(view.available, true);
  assert.deepEqual(
    view.rows.map((r) => r.queue_id),
    ['q-strong', 'q-gone', 'q-off', 'q-new']
  );
  const strong = view.rows[0];
  assert.equal(strong.processed, true, 'processed since the build, from the live overlay');
  assert.match(strong.reason, /Cited by 1 paper of yours/);
  const fresh = view.rows[3];
  assert.equal(fresh.scored, false);
  assert.equal(fresh.verdict, null);
  assert.match(fresh.reason, /Not scored yet/);
  assert.equal(view.rows[1].gone, true, 'removed since the build');
  assert.deepEqual(view.removed.map((r) => r.queue_id), ['q-gone']);
});

test('queueView says the queue is unavailable instead of showing an empty one', () => {
  const unavailable = response();
  unavailable.overlays.queue = { available: false, reason: 'queue down' };
  const view = core.queueView(core.prepare(unavailable, PILLAR));
  assert.equal(view.available, false);
  assert.equal(view.reason, 'queue down');
  assert.equal(view.rows.length, 3, 'the snapshot verdicts are still shown');
});

test('verdictInfo covers every verdict the build emits', () => {
  [
    'close_strong',
    'close_related',
    'on_topic_new_direction',
    'far_unanchored',
    'off_topic',
    'duplicate',
    'already_yours',
    'unscored',
  ].forEach((v) => assert.ok(core.VERDICTS[v], v));
  assert.equal(core.verdictInfo('something_new').label, 'something_new');
});

test('queueRequest targets the M6 route and sends only what identifies the work', () => {
  const req = core.queueRequest('model-interpretability', { id: 'w-1', title: 'A title' });
  assert.equal(req.url, '/api/pillars/model-interpretability/queue/external');
  assert.equal(req.options.method, 'POST');
  assert.deepEqual(JSON.parse(req.options.body), { work_id: 'w-1', title: 'A title' });
  assert.equal(core.queueRequest('a/b', { id: 'w' }).url, '/api/pillars/a%2Fb/queue/external');
});

test('interpretQueueResponse: a 422 is "no open PDF", never a crash', () => {
  assert.equal(core.interpretQueueResponse(200, {}).kind, 'queued');
  assert.equal(core.interpretQueueResponse(201, {}).kind, 'queued');
  assert.equal(core.interpretQueueResponse(409, { detail: 'Already queued.' }).message, 'Already queued.');
  const noPdf = core.interpretQueueResponse(422, { detail: 'No open PDF is known for this paper.' });
  assert.equal(noPdf.kind, 'no_pdf');
  assert.equal(noPdf.message, 'No open PDF is known for this paper.');
  assert.equal(core.interpretQueueResponse(422, null).kind, 'no_pdf');
  assert.equal(core.interpretQueueResponse(404, null).kind, 'unavailable');
  assert.equal(core.interpretQueueResponse(503, { detail: 'db down' }).kind, 'error');
  assert.equal(core.interpretQueueResponse(500, null).message, 'Could not queue the paper (HTTP 500).');
});

test('sortRows is stable and always puts unknown values last', () => {
  const rows = [
    node('a', { citation_count: 5 }),
    node('b', { citation_count: null }),
    node('c', { citation_count: 50 }),
    node('d', { citation_count: 5 }),
  ];
  assert.deepEqual(core.sortRows(rows, 'citations', 'desc').map((n) => n.id), ['c', 'a', 'd', 'b']);
  assert.deepEqual(core.sortRows(rows, 'citations', 'asc').map((n) => n.id), ['a', 'd', 'c', 'b']);
  assert.deepEqual(core.sortRows(rows, 'nonsense', 'asc').length, 4);
});

test('findMatches needs every term, in title, label, authors or year', () => {
  const nodes = [
    node('a', { title: 'Control tasks for probes', authors: ['John Hewitt'], year: 2019 }),
    node('b', { title: 'Attention is not explanation', authors: ['Sarthak Jain'], year: 2019 }),
  ];
  assert.deepEqual(Object.keys(core.findMatches(nodes, 'hewitt 2019')), ['a']);
  assert.deepEqual(Object.keys(core.findMatches(nodes, '2019')).sort(), ['a', 'b']);
  assert.deepEqual(core.findMatches(nodes, ''), {});
  assert.deepEqual(core.findMatches(nodes, 'zzz'), {});
});

test('citationAvailability says "unavailable" for a map with no citation lines', () => {
  const none = response();
  none.snapshot.citations = [];
  const bad = core.citationAvailability(core.prepare(none, PILLAR));
  assert.equal(bad.unavailable, true);
  assert.match(bad.message, /not a map of papers that cite nothing/);

  const good = core.citationAvailability(core.prepare(response(), PILLAR));
  assert.equal(good.unavailable, false);
  assert.equal(good.message, '');
  assert.equal(good.counts, true);
});

test('masteryText separates "no reviews yet" from "unavailable"', () => {
  const model = core.prepare(response(), PILLAR);
  assert.equal(core.masteryText(model, model.byId.mine1).text, '80% recall');
  assert.equal(core.masteryText(model, model.byId.mine2).known, false);
  assert.match(core.masteryText(model, model.byId.mine2).text, /No reviewed cards yet/);
  const down = response();
  down.overlays.mastery = { available: false, reason: 'migration 024' };
  const m = core.prepare(down, PILLAR);
  assert.match(core.masteryText(m, m.byId.mine1).text, /unavailable: migration 024/);
});

test('parseQuery and buildQuery round-trip, and an unknown view falls back to Timeline', () => {
  assert.deepEqual(core.parseQuery('?pillar=a&view=list&select=w1'), { pillar: 'a', view: 'list', select: 'w1' });
  assert.equal(core.parseQuery('?view=globe').view, 'timeline');
  assert.equal(core.parseQuery('').view, 'timeline');
  assert.equal(core.buildQuery({ pillar: 'a', view: 'timeline', select: '' }), '?pillar=a');
  assert.equal(core.buildQuery({ pillar: 'a', view: 'clusters', select: 'w 1' }), '?pillar=a&view=clusters&select=w+1');
  assert.equal(core.buildQuery({ pillar: '', view: 'timeline' }), '');
});

test('links are encoded: a paper id carries a colon and a pillar can carry anything', () => {
  assert.equal(core.paperHref('arxiv:2402.08593'), '/papers/arxiv%3A2402.08593');
  assert.equal(core.paperHref(''), null);
  assert.equal(core.mapHref('model-interpretability'), '/papers/map?pillar=model-interpretability');
  assert.equal(core.mapHref('a b'), '/papers/map?pillar=a%20b');
});

test('roleLabel words each role the way the legend does', () => {
  const model = core.prepare(response(), PILLAR);
  assert.equal(core.roleLabel(model.byId.mine1), 'In your library');
  assert.equal(core.roleLabel(model.byId.doc), 'Your document');
  assert.equal(core.roleLabel(model.byId.away), 'Yours, another pillar');
  assert.equal(core.roleLabel(model.byId.s2), 'Suggested #2');
  assert.equal(core.roleLabel(model.byId.c1), 'Connected paper');
});

test('nodeCaption expands the selected label to author, year and title', () => {
  const n = node('a', { label: 'Hewitt 2019', title: 'Designing and Interpreting Probes with Control Tasks and Selectivity Measures' });
  assert.equal(core.nodeCaption(n, false), 'Hewitt 2019');
  assert.match(core.nodeCaption(n, true), /^Hewitt 2019 – Designing and Interpreting Probes/);
  assert.ok(core.nodeCaption(n, true).endsWith('…'));
});
