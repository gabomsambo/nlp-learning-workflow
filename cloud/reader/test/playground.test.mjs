// The Playground tab and the code drawer under the player, built from the
// export's own golden files (tests/fixtures/cloud/golden) so the bundle and
// the page are held to the same shape. The app's playground.js and
// follow-along.js are loaded as the page loads them, as globals.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { test } from 'node:test';

import { installFakeDom } from './fake-dom.mjs';

installFakeDom();
const require = createRequire(import.meta.url);
const RA = require('../../../webui/static/readalong.js');
const Core = require('../../../webui/static/listen-core.js');
require('../../../webui/static/playground-core.js');
require('../../../webui/static/playground.js');
const FA = require('../../../webui/static/follow-along.js');

const { playgroundPanel } = await import('../public/js/playground-tab.js');
const { codeDrawer } = await import('../public/js/follow.js');
const {
  codeDrawerState, drawerStep, droppedNote, droppedSummary, outputForRender, parseRoute, playgroundCellHref, requestedCell,
} = await import('../public/js/lib.js');

const golden = (path) => JSON.parse(readFileSync(new URL(`../../../tests/fixtures/cloud/golden/${path}`, import.meta.url), 'utf8'));
const PG_ID = '99999999-9999-4999-8999-999999999991';
const PLAYGROUND = golden(`playgrounds/${PG_ID}.json`);
const EPISODE = golden('episodes/11111111-1111-4111-8111-111111111111.json');
const STALE_EPISODE = golden('episodes/22222222-2222-4222-8222-222222222222.json');
const sha = (text) => createHash('sha256').update(text).digest('hex');
const allText = (node) => node.textContent;

// ---- pure helpers -----------------------------------------------------------

test('the Playground tab is a route, and a cell link carries the cell', () => {
  assert.deepEqual(parseRoute('#/paper/arxiv%3A1706.03762/playground'), { name: 'paper', id: 'arxiv:1706.03762', tab: 'playground' });
  assert.equal(playgroundCellHref('arxiv:1706.03762', 3), '#/paper/arxiv%3A1706.03762/playground?cell=3');
  assert.equal(requestedCell(new URLSearchParams('cell=3')), 3);
  for (const junk of ['cell=', 'cell=-1', 'cell=3x', 'cell=123456', '']) assert.equal(requestedCell(new URLSearchParams(junk)), null, junk);
});

test('published output text is handed to the renderer as text, and a dropped one is not rendered', () => {
  assert.deepEqual(outputForRender({ kind: 'stream', name: 'stdout', body: '3\n' }), { kind: 'stream', name: 'stdout', text: '3\n' });
  assert.deepEqual(outputForRender({ kind: 'png', data: 'abc', alt: 'x' }), { kind: 'png', data: 'abc', alt: 'x' });
  assert.equal(outputForRender({ kind: 'dropped', what: 'figure', bytes: 2_400_000 }), null);
  assert.match(droppedNote({ kind: 'dropped', what: 'figure', bytes: 2_400_000 }), /^This figure \(2\.4 MB\) was left out of the phone copy/);
  assert.match(droppedNote({ kind: 'dropped', what: 'output', bytes: 900 }), /^This output \(1 KB\)/);
  assert.equal(droppedSummary({ figures: 0, outputs: 0 }), '');
  assert.equal(droppedSummary({ figures: 2, outputs: 1 }),
    'To keep the phone copy small, 2 figures and 1 output were left out. They are in the notebook on your computer.');
});

function fixtureModel(episode) {
  const scriptSha = sha(episode.script);
  const timeline = { ...episode.timeline, script_sha256: scriptSha };
  return { scriptSha, timeline, model: RA.buildModel(timeline, Core.scriptLines(episode.script), scriptSha) };
}

test('the drawer is offered only when the code can follow the audio', () => {
  const { scriptSha, timeline } = fixtureModel(EPISODE);
  const links = EPISODE.follow_along;
  assert.equal(codeDrawerState(FA, { hasAudio: true, timeline, scriptSha, links: [] }).mode, 'none');
  assert.equal(codeDrawerState(FA, { hasAudio: true, timeline, scriptSha, links }).mode, 'ready');
  assert.match(codeDrawerState(FA, { hasAudio: false, timeline, scriptSha, links }).reason, /has none yet/);
  assert.match(codeDrawerState(FA, { hasAudio: true, timeline: null, scriptSha, links }).reason, /no read-along timing/);
  // A mapping made for another version of the script: the computer has to match it again.
  const stale = codeDrawerState(FA, { hasAudio: true, timeline, scriptSha, links: STALE_EPISODE.follow_along });
  assert.equal(stale.mode, 'off');
  assert.match(stale.reason, /Match it again on the computer\.$/);
});

test('time -> card follows the desktop lookup: nothing before the span, the card inside it, seeking agrees', () => {
  const { model } = fixtureModel(EPISODE);
  const spans = FA.normalizeSpans(EPISODE.follow_along[0].mapping, PLAYGROUND.steps);
  assert.deepEqual(spans, [{ line_start: 2, line_end: 2, cell_index: 1, step_id: 'step-1' }]);
  assert.equal(drawerStep(FA, model, spans, 1000, null).kind, 'wait');
  const inside = drawerStep(FA, model, spans, 6000, null);
  assert.equal(inside.kind, 'card');
  assert.equal(inside.span.cell_index, 1);
  assert.equal(drawerStep(FA, model, spans, 7000, inside.at).kind, 'same');
  // Seeking straight there lands on the same card as playing there.
  assert.deepEqual(drawerStep(FA, model, spans, 7000, null).at, drawerStep(FA, model, spans, 6000, null).at);
  // A span pointing at a cell the study guide no longer has is dropped, not shown as other code.
  assert.deepEqual(FA.normalizeSpans([{ line_start: 2, line_end: 2, cell_index: 5, step_id: 'step-1' }], PLAYGROUND.steps), []);
});

// ---- the tab ----------------------------------------------------------------

test('the tab shows each step and card, says once that code runs on the computer, and nothing runs', () => {
  localStorage.clear();
  const root = playgroundPanel(PLAYGROUND);
  const text = allText(root);
  assert.equal(text.split('Run code on your computer.').length - 1, 1);
  assert.deepEqual(root.querySelectorAll('.pg-step-title').map(allText), ['Load the points', 'Look at them', 'Rest of the notebook']);
  assert.deepEqual(root.querySelectorAll('.reader-step-link').map(allText), ['Load the points', 'Look at them']);
  assert.deepEqual(root.querySelectorAll('.pg-card').map((c) => c.id), ['pg-cell-1', 'pg-cell-2', 'pg-cell-3', 'pg-cell-4', 'pg-cell-5']);
  assert.equal(root.querySelectorAll('.pg-md').length, 1);
  assert.equal(root.querySelectorAll('textarea').length + root.querySelectorAll('.pg-run').length, 0);
  assert.ok(!/\bRun\b(?! code on your computer)/.test(text.replace('Run code on your computer.', '')));
  // The step list scrolls within the page; it never touches the route.
  root.querySelectorAll('.reader-step-link')[1].click();
  assert.ok(root.querySelector('#pg-step-2').scrolledIntoView);
});

test('explanations hide for a self-quiz, one card at a time can be revealed, and the choice is kept', () => {
  localStorage.clear();
  const root = playgroundPanel(PLAYGROUND);
  const explains = () => root.querySelectorAll('.pg-explain');
  assert.equal(explains().length, 4);
  assert.ok(explains().every((e) => !e.hidden));
  const hide = root.querySelector('.pg-actions').querySelector('button');
  hide.click();
  assert.ok(explains().every((e) => e.hidden));
  assert.equal(hide.getAttribute('aria-pressed'), 'true');
  assert.equal(localStorage.getItem('nlp:playgroundHideExplanations'), 'true');
  const firstReveal = root.querySelector('#pg-cell-1').querySelector('.pg-reveal');
  firstReveal.click();
  assert.deepEqual(explains().map((e) => e.hidden), [false, true, true, true]);
  assert.ok(playgroundPanel(PLAYGROUND).querySelectorAll('.pg-explain').every((e) => e.hidden));
  localStorage.clear();
});

test('a card has its code with tap-a-line notes, the saved output, the try as a diff and the quiz', () => {
  const card = playgroundPanel(PLAYGROUND).querySelector('#pg-cell-1');
  const noted = card.querySelectorAll('.has-note');
  assert.equal(noted.length, 1);
  const note = card.querySelector('.pg-line-note');
  assert.equal(note.hidden, true);
  noted[0].click();
  assert.equal(note.hidden, false);
  assert.equal(allText(note), 'read_csv turns the file into a table.');
  // The sanitized table, built element by element, with its numeric colspan only.
  const td = card.querySelector('td');
  assert.equal(td.getAttribute('colspan'), '2');
  assert.deepEqual([...td.attributes.keys()], ['colspan']);
  const attempt = card.querySelector('.pg-try');
  assert.equal(allText(attempt.querySelector('summary')), 'Try: Show one row');
  assert.equal(allText(attempt.querySelector('.pg-expect')), 'Only the first row is shown.');
  assert.deepEqual(attempt.querySelectorAll('.pg-diff-line').map((l) => l.className.replace('pg-diff-line ', '')),
    ['is-same', 'is-same', 'is-del', 'is-add']);
  assert.match(allText(card.querySelector('.pg-quiz')), /^Can you answer: What does head\(\) show by default\?$/);
});

test('saved outputs: stream text, a figure as a data image, an error, and a capped one says where it went', () => {
  const pg = structuredClone(PLAYGROUND);
  pg.cells[2].outputs.push({ kind: 'dropped', what: 'output', bytes: 5000 });
  pg.dropped = { figures: 0, outputs: 1 };
  const root = playgroundPanel(pg);
  assert.equal(allText(root.querySelector('#pg-cell-2').querySelector('.pg-out')), '3\n');
  assert.match(root.querySelector('#pg-cell-3').querySelector('img').getAttribute('src'), /^data:image\/png;base64,iVBOR/);
  assert.ok(root.querySelector('#pg-cell-4').querySelector('.pg-out').classList.contains('is-error'));
  assert.match(allText(root.querySelector('#pg-cell-2')), /This output \(5 KB\) was left out of the phone copy/);
  assert.match(allText(root.querySelector('.pg-head')), /1 output was left out/);
});

test('a link to one cell marks it and scrolls to it once the page is on screen', () => {
  const root = playgroundPanel(PLAYGROUND, { cell: 3 });
  const card = root.querySelector('#pg-cell-3');
  assert.ok(!card.classList.contains('is-target'));
  root.focusCell();
  assert.ok(card.classList.contains('is-target') && card.scrolledIntoView);
});

// ---- the drawer -------------------------------------------------------------

async function drawerFor(episode, clock) {
  const { scriptSha, timeline, model } = fixtureModel(episode);
  const asked = [];
  const store = { doc: async (key) => { asked.push(key); return structuredClone(PLAYGROUND); } };
  const drawer = codeDrawer({ store, episode, model, scriptSha, timeline, hasAudio: true, now: () => clock });
  await new Promise((resolve) => setTimeout(resolve, 0));
  return { drawer, asked };
}

test('the drawer follows playback, dims between spans, pins, and its card opens the cell', async () => {
  localStorage.clear();
  const clock = { ms: 1000, playing: true };
  const { drawer, asked } = await drawerFor(EPISODE, clock);
  assert.deepEqual(asked, [`playgrounds/${PG_ID}.json`]);
  const slot = drawer.root.querySelector('.follow-card-slot');
  assert.equal(allText(slot), 'The code shows up here when the narrator gets to it.');
  assert.equal(drawer.root.querySelector('details').open, true);

  clock.ms = 6000;
  drawer.update(false);
  const card = slot.querySelector('.follow-card');
  assert.equal(card.getAttribute('href'), '#/paper/arxiv%3A1706.03762/playground?cell=1');
  assert.equal(allText(card.querySelector('.follow-step')), 'Load the points');
  assert.match(allText(card.querySelector('.follow-kicker')), /^Step 1 · In \[1\]$/);
  assert.ok(!card.classList.contains('is-dimmed'));

  const pin = drawer.root.querySelector('.follow-pin');
  assert.equal(pin.hidden, false);
  pin.click();
  assert.equal(pin.getAttribute('aria-pressed'), 'true');
  clock.ms = 500;
  drawer.update(false);
  assert.equal(slot.querySelector('.follow-card'), card);
  pin.click(); // unpinned: back to where the audio is
  assert.equal(allText(slot), 'The code shows up here when the narrator gets to it.');
  drawer.dispose();
});

test('between spans the last card stays, dimmed, and comes back when its span does', async () => {
  const clock = { ms: 1000, playing: true };
  const mapping = [{ line_start: 0, line_end: 0, cell_index: 2, step_id: 'step-1' }];
  const episode = { ...EPISODE, follow_along: [{ ...EPISODE.follow_along[0], mapping }] };
  const { drawer } = await drawerFor(episode, clock);
  const slot = drawer.root.querySelector('.follow-card-slot');
  const card = slot.querySelector('.follow-card');
  assert.equal(card.getAttribute('href'), '#/paper/arxiv%3A1706.03762/playground?cell=2');
  assert.ok(!card.classList.contains('is-dimmed'));
  clock.ms = 6000; // line 2: no code of its own
  drawer.update(false);
  assert.equal(slot.querySelector('.follow-card'), card);
  assert.ok(card.classList.contains('is-dimmed'));
  clock.ms = 2000;
  drawer.update(false);
  assert.ok(!slot.querySelector('.follow-card').classList.contains('is-dimmed'));
});

test('the drawer says why it cannot follow a script that changed, and loads nothing', async () => {
  // The audio and timing are fine; the stored mapping indexes another version of the script.
  const episode = { ...EPISODE, follow_along: STALE_EPISODE.follow_along };
  const { drawer, asked } = await drawerFor(episode, { ms: 0, playing: false });
  assert.deepEqual(asked, []);
  assert.match(allText(drawer.root.querySelector('.follow-status')), /Match it again on the computer\./);
  assert.equal(drawer.root.querySelector('.follow-pin').hidden, true);
});

test('an episode that follows no notebook has no drawer', () => {
  const { scriptSha, timeline, model } = fixtureModel(EPISODE);
  const episode = { ...EPISODE, follow_along: [] };
  assert.equal(codeDrawer({ store: {}, episode, model, scriptSha, timeline, hasAudio: true, now: () => ({ ms: 0 }) }), null);
});
