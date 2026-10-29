/*
 * The Offline & sync page, the banner above every page, and the save button on an episode.
 * All of it is createElement + textContent. What each one says is decided in offline-core.js.
 */
import { emptyState, h } from './dom.js';
import { formatDate, formatDuration } from './lib.js';
import {
  connectionNotice, downloadProblem, formatBytes, installHint, outboxPanel, savedLabel,
} from './offline-core.js';

/** Goes through the Access login: the worker leaves `?signin` alone, so a cached page cannot answer it. */
export const SIGN_IN_HREF = '/?signin=1';

const button = (label, onClick, { primary = false, disabled = false } = {}) => {
  const el = h('button', { type: 'button', class: `btn ${primary ? 'btn-primary' : 'btn-secondary'} btn-md`, text: label, disabled });
  el.addEventListener('click', onClick);
  return el;
};

/** Repaint at most every `ms`: a download reports progress far more often than anyone can read it. */
function throttled(paint, ms = 250) {
  let timer = null;
  return () => {
    if (timer) return;
    timer = setTimeout(() => { timer = null; paint(); }, ms);
  };
}

/** The strip under the header when the page is showing saved content. Returns the element. */
export function createBanner({ store, retry }) {
  const el = h('div', { class: 'reader-banner', role: 'status', hidden: true });
  const paint = () => {
    const status = store.status();
    const notice = connectionNotice({ mode: status.mode, publishedAt: status.publishedAt ? formatDate(status.publishedAt) : '' });
    el.hidden = !notice;
    el.className = notice ? `reader-banner is-${notice.tone}` : 'reader-banner';
    if (!notice) {
      el.replaceChildren();
      return;
    }
    const action = notice.action === 'signin'
      ? h('a', { class: 'btn btn-primary btn-sm', href: SIGN_IN_HREF, text: 'Sign in again' })
      : button('Try again', retry, {});
    el.replaceChildren(h('span', { text: notice.text }), action);
  };
  paint();
  store.onStatus(paint);
  return { el, paint };
}

/** One episode's save button, repainting as the download moves. `row` is offline.episodes()'s item. */
export function saveControl(offline, episodeId) {
  const root = h('div', { class: 'offline-save' });
  let alive = true;
  async function paint() {
    let row;
    try {
      row = (await offline.episodes()).find((e) => e.id === episodeId);
    } catch (error) {
      row = null;
    }
    if (!alive) return;
    if (!row) {
      root.replaceChildren();
      return;
    }
    const label = savedLabel({ status: row.status, pinned: row.pinned, progress: row.progress });
    const detail = row.status === 'saved' && row.bytes ? ` · ${formatBytes(row.bytes)}` : '';
    const note = row.status === 'error' ? downloadProblem(row.problem) : `${label}${detail}`;
    let action = null;
    if (row.status === 'saving') action = null;
    else if (row.status === 'saved' && row.pinned) action = button('Remove from this device', () => offline.remove(episodeId));
    else if (row.status === 'saved') action = button('Keep on this device', () => offline.save(episodeId));
    else action = button(row.status === 'error' ? 'Try again' : 'Save for offline', () => offline.save(episodeId));
    root.replaceChildren(...[action, note ? h('span', { class: 'offline-note', text: note }) : null].filter(Boolean));
  }
  const unsubscribe = offline.onChange(throttled(paint, 150));
  paint();
  root.dispose = () => { alive = false; unsubscribe(); };
  return root;
}

/** A chip for the Listen list: whether this episode plays with no connection. */
export function savedChip(rows, episodeId) {
  const row = rows.get(episodeId);
  if (!row || row.status !== 'saved') return null;
  return h('span', { class: 'chip chip-accent', text: 'Offline' });
}

export async function offlineView({ store, reviews, offline, worker, installer, online }) {
  const root = h('div', { class: 'offline-page' });
  let alive = true;
  let note = '';

  async function paint() {
    if (!alive) return;
    const status = store.status();
    const outbox = outboxPanel({ ...reviews.status(), online: online() });
    let rows = [];
    try {
      rows = await offline.episodes();
    } catch (error) {
      rows = null;
    }
    const auto = await offline.autoKeep();
    const usage = await offline.usage();
    const persisted = await offline.persisted();
    if (!alive) return;

    const signIn = () => h('a', { class: 'btn btn-primary btn-md', href: SIGN_IN_HREF, text: 'Sign in again' });
    const syncNow = button('Sync now', async () => {
      note = 'Syncing';
      paint();
      await reviews.sync();
      await offline.sync();
      note = '';
      paint();
    }, { primary: outbox.action === 'sync' });

    const connection = h('section', { class: 'card offline-card' },
      h('h2', { class: 'section-title', text: 'Connection' }),
      h('p', { class: 'offline-lead', text: {
        live: online() ? 'Online. The reader is up to date.' : 'The browser says you are offline.',
        offline: 'Offline. You are reading what was saved on this device.',
        'signed-out': 'Your session has ended. You are reading what was saved on this device.',
      }[status.mode] }),
      status.publishedAt ? h('p', { class: 'offline-note', text: `Library published ${formatDate(status.publishedAt)}` }) : null,
      h('div', { class: 'offline-actions' }, status.mode === 'signed-out' ? signIn() : null, syncNow),
      note ? h('p', { class: 'offline-note', role: 'status', text: note }) : null);

    const reviewCard = h('section', { class: `card offline-card is-${outbox.tone}` },
      h('h2', { class: 'section-title', text: 'Reviews' }),
      h('p', { class: 'offline-lead', text: outbox.headline }),
      outbox.detail ? h('p', { class: 'offline-note', text: outbox.detail }) : null,
      outbox.action === 'signin' ? h('div', { class: 'offline-actions' }, signIn()) : null,
      reviews.status().rejected > 0
        ? h('div', { class: 'offline-actions' }, button('Dismiss', () => { reviews.clearRejected(); paint(); }))
        : null);

    const toggle = h('input', { type: 'checkbox', id: 'offline-auto', checked: auto });
    toggle.addEventListener('change', async () => { await offline.setAutoKeep(toggle.checked); paint(); });
    const episodeList = rows === null
      ? emptyState('Episodes are not available', 'They load once the library has been fetched.')
      : rows.length === 0
        ? h('p', { class: 'offline-note', text: 'No episode has audio yet.' })
        : h('ul', { class: 'offline-episodes' }, rows.map((row) => h('li', { class: 'offline-episode' },
          h('div', { class: 'offline-episode-copy' },
            h('a', { href: `#/listen/${encodeURIComponent(row.id)}`, text: row.title }),
            h('span', { class: 'offline-note', text: [
              formatDuration(row.duration_seconds),
              row.status === 'error' ? downloadProblem(row.problem) : savedLabel({ status: row.status, pinned: row.pinned, progress: row.progress }),
              row.status === 'saved' && row.bytes ? formatBytes(row.bytes) : '',
            ].filter(Boolean).join(' · ') })),
          row.status === 'saving' ? null : (row.status === 'saved' && row.pinned
            ? button('Remove', () => offline.remove(row.id).then(paint))
            : button(row.status === 'saved' ? 'Keep' : row.status === 'error' ? 'Try again' : 'Save', () => offline.save(row.id).then(paint))))));

    const episodes = h('section', { class: 'card offline-card' },
      h('h2', { class: 'section-title', text: 'Episodes on this device' }),
      h('label', { class: 'offline-toggle', for: 'offline-auto' }, toggle, h('span', { text: 'Keep the newest 3 automatically' })),
      episodeList);

    const hint = installHint(installer.hint());
    const state = worker.state();
    const device = h('section', { class: 'card offline-card' },
      h('h2', { class: 'section-title', text: 'This device' }),
      h('p', { class: 'offline-lead', text: hint.text }),
      hint.prompt ? h('div', { class: 'offline-actions' }, button('Install', () => installer.prompt().then(paint), { primary: true })) : null,
      h('p', { class: 'offline-note', text: !state.supported
        ? 'This browser cannot keep the app for offline use.'
        : state.error ? `Offline use is not available: ${state.error}`
          : state.registered ? 'The app opens with no connection.' : 'Setting up offline use' }),
      state.updateReady
        ? h('div', { class: 'offline-actions' }, h('span', { class: 'offline-note', text: 'A new version is ready.' }), button('Reload to update', () => worker.applyUpdate(), { primary: true }))
        : null,
      h('p', { class: 'offline-note', text: [
        usage ? `Using ${formatBytes(usage.usage)} of ${formatBytes(usage.quota)}.` : '',
        persisted === true ? 'The browser will not clear this data on its own.'
          : persisted === false ? 'The browser may clear this data if the device runs low on space.' : '',
      ].filter(Boolean).join(' ') }));

    root.replaceChildren(
      h('header', { class: 'page-header' },
        h('div', null, h('p', { class: 'page-eyebrow', text: 'This device' }), h('h1', { class: 'page-title', text: 'Offline & sync' }))),
      h('div', { class: 'offline-grid' }, connection, reviewCard, episodes, device));
  }

  const repaint = throttled(paint);
  const unsubscribes = [offline.onChange(repaint), reviews.onChange(repaint), store.onStatus(repaint)];
  root.dispose = () => { alive = false; for (const stop of unsubscribes) stop(); };
  await paint();
  return root;
}
