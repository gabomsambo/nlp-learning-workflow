import { ReaderError, createStore } from './data.js';
import { emptyState, h, icon } from './dom.js';
import { episodeView, listenView } from './episodes.js';
import { formatDate, parseRoute } from './lib.js';
import { openKv } from './kv.js';
import { createOffline, registerWorker } from './offline.js';
import { headerStatus, outboxPanel } from './offline-core.js';
import { SIGN_IN_HREF, createBanner, offlineView } from './offline-view.js';
import { createOutbox, localStorageStorage, memoryStorage } from './outbox.js';
import { reviewView } from './review.js';
import { createReviews } from './reviews.js';
import { createSync } from './sync.js';
import { lessonsView, libraryView, paperView, todayView } from './views.js';

// Documents and saved episodes live on the device (kv.js: IndexedDB), so the reader opens with no connection.
const kv = openKv();
const store = createStore({ kv });
// Reviews stay in localStorage on purpose: a write there is synchronous, so a review is on disk before
// the tap handler returns, which an IndexedDB write (finished later) cannot promise if the page is killed.
let outboxStorage;
try {
  outboxStorage = localStorageStorage();
} catch (error) {
  outboxStorage = memoryStorage();
}
const outbox = createOutbox(outboxStorage);
const reviews = createReviews({ store, outbox, sync: createSync({ outbox }) });
// Seventy-odd megabytes should not arrive unasked on a connection the person has told the browser to spare.
const offline = createOffline({ store, kv, autoAllowed: () => !(navigator.connection && navigator.connection.saveData) });
const main = document.getElementById('main');

const online = () => navigator.onLine !== false;
const worker = registerWorker({ onChange: () => paintStatus() });

// Chrome offers installation through an event; Safari has only Share > Add to Home Screen.
let installEvent = null;
window.addEventListener('beforeinstallprompt', (event) => { event.preventDefault(); installEvent = event; });
window.addEventListener('appinstalled', () => { installEvent = null; });
const installer = {
  hint: () => ({
    standalone: window.matchMedia('(display-mode: standalone)').matches || navigator.standalone === true,
    canPrompt: !!installEvent,
    ios: /iphone|ipad|ipod/i.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1),
  }),
  async prompt() {
    if (!installEvent) return;
    const event = installEvent;
    installEvent = null;
    await event.prompt();
  },
};

const NAV = [
  { name: 'today', href: '#/', label: 'Today', icon: 'today' },
  { name: 'library', href: '#/library', label: 'Library', icon: 'library' },
  { name: 'listen', href: '#/listen', label: 'Listen', icon: 'listen' },
  { name: 'lessons', href: '#/lessons', label: 'Lessons', icon: 'lessons' },
  { name: 'review', href: '#/review', label: 'Review', icon: 'review' },
];

const banner = createBanner({ store, retry: () => { reviews.sync().then(() => render()); } });
document.querySelector('.phone-bar').after(banner.el);

function paintNav(active) {
  const current = active === 'offline' ? null : active === 'paper' ? 'library' : active === 'episode' ? 'listen' : active;
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

/** The one link in the header that says what needs attention, or points at the Offline page. */
function paintStatus() {
  const panel = outboxPanel({ ...reviews.status(), online: online() });
  const status = headerStatus({ mode: store.status().mode, outbox: panel, updateReady: worker.state().updateReady });
  for (const el of document.querySelectorAll('[data-status]')) {
    el.textContent = status.text;
    el.className = `reader-status is-${status.tone}`;
  }
}

function failure(error) {
  const retry = h('button', { type: 'button', class: 'btn btn-primary btn-md reader-retry', text: 'Try again' });
  retry.addEventListener('click', () => render());
  if (error instanceof ReaderError && error.kind === 'auth') {
    const signIn = h('a', { class: 'btn btn-primary btn-md', href: SIGN_IN_HREF, text: 'Sign in again' });
    return emptyState('Your session has ended', 'Sign in again to keep reading. Nothing has been lost.', signIn);
  }
  if (error instanceof ReaderError && error.kind === 'empty') {
    return emptyState('Nothing published yet', 'The library appears here after the first publish from the computer.');
  }
  if (error instanceof ReaderError && error.kind === 'unavailable') {
    return emptyState('Not saved on this device', 'You are offline and this page was not saved. It opens once you are back online.',
      h('a', { class: 'btn btn-secondary btn-md', href: '#/offline', text: 'See what is saved' }));
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
    else if (route.name === 'listen') view = await listenView(store, offline);
    else if (route.name === 'episode') view = await episodeView(store, route.id, offline);
    else if (route.name === 'offline') view = await offlineView({ store, reviews, offline, worker, installer, online });
    else if (route.name === 'paper') view = await paperView(store, route.id, route.tab, new URLSearchParams(query || ''));
    else view = emptyState('Page not found', 'There is nothing at this address.', h('a', { class: 'btn btn-primary btn-md', href: '#/', text: 'Go to Today' }));
    document.title = { paper: route.tab === 'playground' ? 'Playground · NLP Reader' : 'Paper · NLP Reader', listen: 'Listen · NLP Reader', episode: 'Episode · NLP Reader', offline: 'Offline & sync · NLP Reader' }[route.name] || 'NLP Reader';
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
  if (view.mounted) view.mounted(); // e.g. scroll to the cell a link asked for
  paintFreshness();
  paintStatus();
}

// Send what is waiting whenever there is a reason to think the connection is back, and
// pick up what was published meanwhile.
// The Listen list marks what plays offline, so it is redrawn once a sync has changed that.
function syncOffline() {
  return offline.sync().then(() => { if (parseRoute(location.hash.split('?')[0]).name === 'listen') render(); });
}

async function comeBack() {
  await reviews.sync();
  syncOffline();
  paintStatus();
  paintFreshness();
}
window.addEventListener('online', comeBack);
window.addEventListener('offline', () => { paintStatus(); });
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible') return;
  worker.check();
  reviews.flush();
});
reviews.onChange(() => { paintFreshness(); paintStatus(); });
store.onStatus(() => { paintStatus(); paintFreshness(); });
reviews.flush();

// Back from the Access login: drop the marker so a reload is an ordinary one.
if (new URLSearchParams(location.search).has('signin')) history.replaceState(null, '', `${location.pathname}${location.hash}`);

window.addEventListener('hashchange', () => { window.scrollTo(0, 0); render(); });
// Once the first page is up: hold the newest episodes, and ask the browser to keep what is held.
render().then(() => { if (online()) syncOffline(); });
