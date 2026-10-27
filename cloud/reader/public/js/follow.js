/*
 * The code drawer under the episode player (docs/PLAYGROUND.md, "Follow-along").
 *
 * While the episode plays, the card for the code the narrator is talking
 * about. Time -> line -> span -> card is the desktop panel's own chain, from
 * follow-along.js (loaded as a classic script, index.html): normalizeSpans
 * checks the stored mapping against the playground, followAt maps a time to a
 * span and cardFor a span to its step and cell. Between spans the last card
 * stays, dimmed; Pin keeps it while the episode plays on. Tapping the card
 * opens that cell on the paper's Playground tab.
 */
import { h } from './dom.js';
import { codeDrawerState, drawerStep, playgroundCellHref } from './lib.js';
import { renderSavedOutput } from './playground-tab.js';

const FA = window.FollowAlong;
const PG = window.Playground;
const OPEN_KEY = 'nlp:followDrawerOpen';
const WAIT_PLAYING = 'The code shows up here when the narrator gets to it.';
const WAIT_PAUSED = 'Press play. The code the narrator is talking about shows up here.';

function storedOpen() {
  try { return localStorage.getItem(OPEN_KEY) !== 'false'; } catch (error) { return true; }
}

/**
 * opts: { store, episode, model, scriptSha, hasAudio, timeline, now: () => {ms, playing} }.
 * Returns null when the episode follows no notebook, else { root, update(force), dispose }.
 */
export function codeDrawer({ store, episode, model, scriptSha, hasAudio, timeline, now }) {
  const decided = codeDrawerState(FA, { hasAudio, timeline, scriptSha, links: episode.follow_along });
  if (decided.mode === 'none') return null;

  const status = h('p', { class: 'follow-status', role: 'status', 'aria-live': 'polite' });
  const source = h('span', { class: 'follow-source' });
  const pin = h('button', { type: 'button', class: 'btn btn-secondary btn-sm follow-pin', 'aria-pressed': 'false', hidden: true, text: 'Pin' });
  const slot = h('div', { class: 'follow-card-slot', 'aria-live': 'off' });
  const drawer = h('details', { class: 'follow-drawer' },
    h('summary', { class: 'follow-summary' }, h('h2', { class: 'section-title', text: 'Code' })),
    h('div', { class: 'follow-body' },
      h('div', { class: 'follow-toolbar' }, source, h('span', { class: 'follow-tools' }, pin)),
      status, slot));
  drawer.open = storedOpen();
  drawer.addEventListener('toggle', () => {
    try { localStorage.setItem(OPEN_KEY, drawer.open ? 'true' : 'false'); } catch (error) { /* this view only */ }
  });
  const root = h('aside', { class: 'follow-panel is-drawer reader-follow', 'aria-label': 'Code for this episode' }, drawer);

  let playground = null;
  let spans = [];
  let shown = null;
  let pinned = false;
  let gone = false;

  function placeholder(text) {
    if (slot.textContent !== text || !slot.firstChild || slot.firstChild.className !== 'follow-empty') {
      slot.replaceChildren(h('p', { class: 'follow-empty', text }));
    }
  }

  function renderCard(span, inside) {
    const card = FA.cardFor(span, playground);
    if (!card) {
      placeholder('That cell is no longer in the study guide.');
      return;
    }
    const count = card.cell.execution_count != null ? `In [${card.cell.execution_count}]` : `Cell ${card.cell.index}`;
    const outputs = card.cell.outputs || [];
    slot.replaceChildren(h('a', {
      class: `follow-card${inside ? '' : ' is-dimmed'}`,
      href: playgroundCellHref(playground.paper_id, card.cell.index),
      'aria-label': `Open cell ${card.cell.index} on the Playground tab: ${card.step.title || ''}`,
    },
    h('p', { class: 'follow-kicker', text: `Step ${card.number} · ${count}` }),
    h('h3', { class: 'follow-step', text: card.step.title || `Step ${card.number}` }),
    card.guide && card.guide.explanation ? h('p', { class: 'follow-explain', text: card.guide.explanation }) : null,
    // No line notes here: a note is a button, and this card is one link.
    PG.renderCode(card.cell.source, []),
    outputs.length ? h('div', { class: 'pg-outputs' }, h('p', { class: 'pg-label', text: 'Saved output' }), outputs.map(renderSavedOutput)) : null,
    h('span', { class: 'follow-open', text: 'Open in Playground →' })));
  }

  /** Show the card for wherever the audio is. Cheap when nothing changed. */
  function update(force) {
    if (!playground || !spans.length || (pinned && !force)) return;
    const { ms, playing } = now();
    const step = drawerStep(FA, model, spans, ms, force ? null : shown);
    if (step.kind === 'wait') placeholder(playing ? WAIT_PLAYING : WAIT_PAUSED);
    else if (step.kind === 'dim' && slot.firstChild && slot.firstChild.classList) slot.firstChild.classList.toggle('is-dimmed', !step.at.inside);
    else if (step.kind === 'card' || step.kind === 'dim') renderCard(spans[step.at.index], step.at.inside);
    shown = step.at;
  }

  function paintPin() {
    pin.textContent = pinned ? 'Pinned' : 'Pin';
    pin.setAttribute('aria-pressed', pinned ? 'true' : 'false');
    pin.title = pinned ? 'The card stays put. Press again to follow the audio.' : 'Keep this card while the episode plays on';
  }
  pin.addEventListener('click', () => {
    pinned = !pinned;
    paintPin();
    if (!pinned) update(true); // snap back to where the audio is
  });
  paintPin();

  if (decided.mode !== 'ready') {
    status.textContent = decided.reason;
  } else {
    const link = decided.link;
    source.textContent = `Following ${link.title || 'a notebook'}`;
    status.textContent = 'Loading the notebook…';
    store.doc(link.doc).then((pg) => {
      if (gone) return;
      playground = pg;
      spans = FA.normalizeSpans(link.mapping, pg.steps);
      status.textContent = spans.length ? '' : 'This episode does not walk through any of the notebook’s cells.';
      pin.hidden = !spans.length;
      update(true);
    }).catch((error) => {
      if (!gone) status.textContent = `The notebook could not be loaded. ${error && error.message ? error.message : ''}`.trim();
    });
  }

  return {
    root,
    update,
    dispose() { gone = true; },
  };
}
