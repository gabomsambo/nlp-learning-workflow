/*
 * The Review page: one card, reveal, four ratings, the same shape as the app's Review page
 * (it uses the same `.review-*` styles). The queue is the published snapshot with this device's
 * reviews folded on top, scheduled by fsrs.js, so it works with no connection and gives the
 * computer the same schedule once the reviews reach it.
 */
import { emptyState, h } from './dom.js';
import { formatInterval, newCard, previewIntervals } from './fsrs.js';
import { buildQueue, nextDueAt, summarize } from './schedule.js';
import { paperHref, plural } from './lib.js';
import { statusLine } from './reviews.js';

const RATINGS = [
  { rating: 1, label: 'Again' },
  { rating: 2, label: 'Hard' },
  { rating: 3, label: 'Good' },
  { rating: 4, label: 'Easy' },
];
// Learning steps are minutes. When the next card is that close, wait for it rather than end.
const WAIT_FOR_MS = 15 * 60 * 1000;

/** Card text from the paper's document, or null when it is not there. */
async function contentFor(store, library, state) {
  const entry = library.papers.find((p) => p.id === state.paper_id);
  if (!entry) return null;
  const paper = await store.doc(entry.doc);
  const card = (paper.quiz_cards || []).find((c) => c.id === state.card_id);
  if (!card) return null;
  return { card, paper: entry, pillar: library.pillars.find((p) => p.id === entry.pillar_id) };
}

export async function reviewView(store, reviews, { now = Date.now, newId = () => globalThis.crypto.randomUUID() } = {}) {
  const library = await store.library();
  const session = { id: newId(), introduced: new Set(), skipped: new Set(), reviewed: 0, tally: [0, 0, 0, 0] };

  const heading = h('header', { class: 'page-header' }, h('div', null,
    h('p', { class: 'page-eyebrow', text: 'Spaced repetition' }),
    h('h1', { class: 'page-title', text: 'Review' })));
  const progress = h('p', { class: 'review-progress' });
  const sync = h('p', { class: 'review-help', role: 'status', 'aria-live': 'polite' });
  const body = h('article', { class: 'review-card' });
  const root = h('section', { class: 'review', 'aria-label': 'Review session' }, progress, sync, body);
  const page = h('div', null, heading, root);

  let current = null; // {state, content, shownAt, revealed}
  let timer = null;
  let token = 0;

  function paintSync() {
    sync.textContent = statusLine(reviews.status());
    sync.hidden = sync.textContent === '';
  }
  const stopListening = reviews.onChange(() => {
    if (!page.isConnected) { stopListening(); clearTimeout(timer); return; }
    paintSync();
  });

  async function show() {
    clearTimeout(timer);
    const mine = ++token;
    const states = await reviews.states();
    if (mine !== token) return;
    const at = now();
    const queue = buildQueue(states, at, { introduced: session.introduced })
      .filter((s) => !session.skipped.has(s.card_id));
    const counts = summarize(states, at);
    progress.textContent = `${plural(counts.due, 'card')} due · ${counts.fresh} new`;

    for (const state of queue) {
      let content;
      try {
        content = await contentFor(store, library, state);
      } catch (error) {
        content = null;
      }
      if (mine !== token) return;
      if (content) {
        current = { state, content, shownAt: now(), revealed: false };
        paintCard();
        return;
      }
      // The text is not here (a card that was regenerated, or a document this device cannot reach yet).
      session.skipped.add(state.card_id);
    }
    current = null;
    paintEnd(states, at);
  }

  function paintCard() {
    const { state, content } = current;
    const previews = previewIntervals(state.fresh ? newCard() : state.card, now());
    const meta = h('div', { class: 'review-meta' },
      content.pillar ? h('span', { class: `chip chip-pillar pillar-${content.pillar.id}`, text: content.pillar.name }) : null,
      state.fresh ? h('span', { class: 'chip', text: 'New' }) : null,
      h('a', { class: 'review-paper', href: paperHref(content.paper.id), text: content.paper.title }));
    const question = h('h2', { class: 'review-question', id: 'review-question', text: content.card.question });
    const reveal = h('button', { type: 'button', class: 'btn btn-primary', 'data-reveal': 'true', text: 'Show answer' });
    const answer = h('div', { class: 'review-answer', hidden: true },
      h('p', { class: 'review-answer-label', text: 'Answer' }),
      h('p', { class: 'review-answer-text', text: content.card.answer }));
    const ratings = h('div', { class: 'review-ratings', hidden: true }, RATINGS.map(({ rating, label }) => {
      const interval = formatInterval(previews[rating]);
      return h('button', { type: 'button', class: `review-rate review-rate-${rating}`, 'data-rating': rating, 'aria-label': `${label}, next review ${interval}` },
        h('span', { class: 'review-rate-key', text: String(rating) }),
        h('span', { class: 'review-rate-label', text: label }),
        h('span', { class: 'review-rate-interval', text: interval }));
    }));
    body.setAttribute('aria-labelledby', 'review-question');
    body.replaceChildren(meta, question, reveal, answer, ratings);
    reveal.focus({ preventScroll: true });
  }

  function reveal() {
    if (!current || current.revealed) return;
    current.revealed = true;
    body.querySelector('[data-reveal]').hidden = true;
    body.querySelector('.review-answer').hidden = false;
    body.querySelector('.review-ratings').hidden = false;
    body.querySelector('[data-rating="3"]').focus({ preventScroll: true });
  }

  function rate(rating) {
    if (!current || !current.revealed) return;
    const { state, shownAt } = current;
    current = null;
    if (state.fresh) session.introduced.add(state.card_id);
    session.reviewed += 1;
    session.tally[rating - 1] += 1;
    reviews.rate(state.card_id, rating, { responseMs: Math.max(0, now() - shownAt), sessionId: session.id });
    paintSync();
    show();
  }

  function paintEnd(states, at) {
    const upcoming = nextDueAt(states, at);
    const [again, hard, good, easy] = session.tally;
    const soon = upcoming !== null && upcoming - at <= WAIT_FOR_MS;
    body.removeAttribute('aria-labelledby');
    body.replaceChildren(
      h('h2', { class: 'review-end-title', text: session.reviewed ? 'Session complete' : 'Nothing to review right now' }),
      session.reviewed ? h('p', { class: 'review-end-count', text: `You reviewed ${plural(session.reviewed, 'card')}.` }) : null,
      session.reviewed ? h('p', { class: 'review-tally', text: `Again ${again} · Hard ${hard} · Good ${good} · Easy ${easy}` }) : null,
      upcoming !== null
        ? h('p', { class: 'review-next', text: soon ? 'The next card is due in a few minutes. It will appear here.' : `Next card due in ${formatInterval(upcoming - at)}.` })
        : h('p', { class: 'review-next', text: 'No other cards are scheduled.' }),
      h('div', { class: 'review-end-actions' },
        h('a', { class: 'btn btn-primary', href: '#/', text: 'Back to Today' })));
    if (soon) timer = setTimeout(() => { if (page.isConnected) show(); }, upcoming - at + 500);
  }

  root.addEventListener('click', (event) => {
    const button = event.target.closest('button');
    if (!button) return;
    if (button.dataset.reveal) reveal();
    else if (button.dataset.rating) rate(Number(button.dataset.rating));
  });
  const onKey = (event) => {
    if (!page.isConnected) { document.removeEventListener('keydown', onKey); return; }
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const tag = (event.target && event.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (event.key === ' ' && tag !== 'BUTTON' && tag !== 'A') { event.preventDefault(); reveal(); }
    else if (['1', '2', '3', '4'].includes(event.key)) rate(Number(event.key));
  };
  document.addEventListener('keydown', onKey);

  paintSync();
  try {
    await show();
  } catch (error) {
    return h('div', null, heading, emptyState('Cards could not be loaded', 'Try again in a moment.'));
  }
  // Fetch a fresh snapshot and other devices' reviews behind the page; the queue is recomputed
  // from whatever is held the next time a card is shown.
  reviews.sync().then(() => { paintSync(); });
  return page;
}
