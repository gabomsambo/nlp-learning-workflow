import { ReaderError, createStore } from './data.js';
import { emptyState, h, icon } from './dom.js';
import { episodeView, listenView } from './episodes.js';
import { formatDate, parseRoute } from './lib.js';
import { createOutbox, localStorageStorage, memoryStorage } from './outbox.js';
import { reviewView } from './review.js';
import { createReviews } from './reviews.js';
import { createSync } from './sync.js';
import { lessonsView, libraryView, paperView, todayView } from './views.js';

const store = createStore();
// Reviews are kept in localStorage for now; the offline step moves them to IndexedDB behind the same interface.
let outboxStorage;
try {
  outboxStorage = localStorageStorage();
} catch (error) {
  outboxStorage = memoryStorage();
}
const outbox = createOutbox(outboxStorage);
const reviews = createReviews({ store, outbox, sync: createSync({ outbox }) });
const main = document.getElementById('main');

const NAV = [
  { name: 'today', href: '#/', label: 'Today', icon: 'today' },
  { name: 'library', href: '#/library', label: 'Library', icon: 'library' },
  { name: 'listen', href: '#/listen', label: 'Listen', icon: 'listen' },
  { name: 'lessons', href: '#/lessons', label: 'Lessons', icon: 'lessons' },
  { name: 'review', href: '#/review', label: 'Review', icon: 'review' },
];

function paintNav(active) {
  const current = active === 'paper' ? 'library' : active === 'episode' ? 'listen' : active;
  for (const nav of document.querySelectorAll('[data-nav]')) {
    const tabs = nav.dataset.nav === 'tabs';
    nav.replaceChildren(...NAV.map((item) =>
      h('a', { class: tabs ? null : 'nav-link', href: item.href, 'aria-current': item.name === current ? 'page' : null },
        icon(item.icon), h('span', { text: item.label }))));
  }
}

async function paintFreshness() {
  let text = '';
  try {
    const manifest = await store.manifest();
    text = `Published ${formatDate(manifest.generated_at)}`;
  } catch (error) {
    text = '';
  }
  const waiting = outbox.waiting();
  if (waiting > 0) text = [text, `${waiting} review${waiting === 1 ? '' : 's'} waiting to sync`].filter(Boolean).join(' · ');
  for (const el of document.querySelectorAll('[data-freshness]')) el.textContent = text;
}

function failure(error) {
  const retry = h('button', { type: 'button', class: 'btn btn-primary btn-md reader-retry', text: 'Try again' });
  retry.addEventListener('click', () => render());
  if (error instanceof ReaderError && error.kind === 'auth') {
    const signIn = h('a', { class: 'btn btn-primary btn-md', href: '/', text: 'Sign in again' });
    return emptyState('Your session has ended', 'Sign in again to keep reading. Nothing has been lost.', signIn);
  }
  if (error instanceof ReaderError && error.kind === 'empty') {
    return emptyState('Nothing published yet', 'The library appears here after the first publish from the computer.');
  }
  const message = error instanceof ReaderError ? error.message : 'Something went wrong showing this page.';
  return emptyState('Could not load this page', message, retry);
}

let token = 0;
let firstRender = true;
let mounted = null;

async function render() {
  const mine = ++token;
  const [path, query] = location.hash.split('?');
  const route = parseRoute(path);
  paintNav(route.name);
  let view;
  try {
    if (route.name === 'today') view = await todayView(store, reviews);
    else if (route.name === 'library') view = await libraryView(store, new URLSearchParams(query || ''));
    else if (route.name === 'lessons') view = await lessonsView(store);
    else if (route.name === 'review') view = await reviewView(store, reviews);
    else if (route.name === 'listen') view = await listenView(store);
    else if (route.name === 'episode') view = await episodeView(store, route.id);
    else if (route.name === 'paper') view = await paperView(store, route.id, route.tab);
    else view = emptyState('Page not found', 'There is nothing at this address.', h('a', { class: 'btn btn-primary btn-md', href: '#/', text: 'Go to Today' }));
    document.title = { paper: 'Paper · NLP Reader', listen: 'Listen · NLP Reader', episode: 'Episode · NLP Reader' }[route.name] || 'NLP Reader';
  } catch (error) {
    view = failure(error);
  }
  if (mine !== token) {
    if (view.dispose) view.dispose(); // built, never shown: its audio must not start
    return; // a newer navigation won
  }
  // Leaving an episode stops it; the audio element goes with the page.
  if (mounted && mounted.dispose) mounted.dispose();
  mounted = view;
  main.replaceChildren(view);
  if (!firstRender) main.focus({ preventScroll: true });
  firstRender = false;
  paintFreshness();
}

// Send what is waiting whenever there is a reason to think the connection is back.
window.addEventListener('online', () => { reviews.flush(); });
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') reviews.flush(); });
reviews.onChange(() => { paintFreshness(); });
reviews.flush();

window.addEventListener('hashchange', () => { window.scrollTo(0, 0); render(); });
render();
