/*
 * The decisions behind "works without a connection", with nothing to touch: which episodes the
 * phone keeps, which documents belong to one, what to tell the person about the connection and
 * the reviews waiting. No DOM, no network, no storage: node --test imports this file.
 */
import { audioSrc, listenRows, plural } from './lib.js';

/** Newest episodes that have audio, in the order the Listen page shows them. */
export function playableEpisodes(library) {
  return listenRows(library).filter((episode) => !!audioSrc(episode.audio_key));
}

/**
 * What to download and what to delete, to leave the phone holding the newest `keep` episodes
 * plus every one the person saved by hand.
 *
 *   episodes   playableEpisodes(library)
 *   saved      {episodeId: {audio_key}}  the audio the phone holds now
 *   pinned     episode ids the person asked to keep (never evicted by newer episodes)
 *   keep       how many of the newest to hold without being asked (0 = only the pinned)
 *
 * A saved copy goes when it is not wanted (older than the newest `keep` and not pinned), when
 * its episode left the library, or when the episode now has different audio (the bytes behind
 * the old name are not the episode any more). The audio is what costs space; documents are
 * small and are pruned against the manifest instead.
 */
export function planEpisodes({ episodes, saved, pinned = [], keep }) {
  const byId = new Map(episodes.map((e) => [e.id, e]));
  const wanted = new Set(episodes.slice(0, Math.max(0, keep)).map((e) => e.id));
  for (const id of pinned) if (byId.has(id)) wanted.add(id);
  const download = episodes
    .filter((e) => wanted.has(e.id) && !(saved[e.id] && saved[e.id].audio_key === e.audio_key))
    .map((e) => e.id);
  const remove = Object.keys(saved).filter((id) => {
    const episode = byId.get(id);
    return !episode || episode.audio_key !== saved[id].audio_key || !wanted.has(id);
  });
  return { download, remove };
}

/**
 * The documents an episode needs to be read and listened to on its own: the episode, the paper
 * it is about, and the playground cards its code drawer follows. `episodeDoc` is the published
 * episode document, `paperDoc` the paper's (either may be null while it is not known yet).
 */
export function episodeDocKeys(entry, episodeDoc, paperDoc, library) {
  const keys = [entry.doc];
  const paper = (library.papers || []).find((p) => p.id === entry.paper_id);
  if (paper && paper.doc) keys.push(paper.doc);
  if (episodeDoc && Array.isArray(episodeDoc.follow_along)) {
    for (const link of episodeDoc.follow_along) if (link && typeof link.doc === 'string') keys.push(link.doc);
  }
  if (paperDoc && paperDoc.playground && typeof paperDoc.playground.doc === 'string') keys.push(paperDoc.playground.doc);
  return [...new Set(keys)];
}

/** The two documents every page starts from. They are kept whatever else is. */
export const BASE_DOCS = ['library.json', 'schedule.json'];

export function formatBytes(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n < 0) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(n < 10 * 1024 * 1024 ? 1 : 0)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

/** Why a download did not finish, in words the person can act on. */
export function downloadProblem(kind) {
  return ({
    auth: 'Your session has ended. Sign in again, then save it.',
    network: 'The connection dropped. It will try again when you are online.',
    corrupt: 'The download did not match the published file. Try again.',
    full: 'There is not enough space on this device.',
  })[kind] || 'It could not be saved.';
}

/** What an episode's row says about its offline copy. */
export function savedLabel({ status, pinned, progress }) {
  if (status === 'saving') return progress != null ? `Saving ${Math.round(progress * 100)}%` : 'Saving';
  if (status === 'saved') return pinned ? 'Saved on this device' : 'Kept offline';
  if (status === 'error') return 'Not saved';
  return '';
}

/**
 * How the page is getting its content, for the banner. `mode` is the store's:
 * live / offline (the network failed) / signed-out (Access wants a login). Returns null when
 * there is nothing to say.
 */
export function connectionNotice({ mode, publishedAt }) {
  if (mode === 'signed-out') {
    return {
      tone: 'warn',
      text: 'Your session has ended. You are reading what is saved on this device.',
      action: 'signin',
    };
  }
  if (mode === 'offline') {
    return {
      tone: 'info',
      text: publishedAt ? `Offline. Showing what was saved on this device (published ${publishedAt}).` : 'Offline. Showing what was saved on this device.',
      action: 'retry',
    };
  }
  return null;
}

/**
 * The outbox as the Offline page and the header show it. Reviews are never lost whatever this
 * says: they wait on the device until the cloud has them. `online` is the browser's idea, `last`
 * the outcome of the last send (nothing / ok / offline / auth / error).
 */
export function outboxPanel({ waiting, rejected = 0, durable = true, last = 'nothing', online = true }) {
  const out = { tone: 'ok', headline: 'All reviews are sent', detail: '', action: null, short: '' };
  if (waiting > 0) {
    const reviews = plural(waiting, 'review');
    out.headline = `${reviews} waiting to sync`;
    out.short = `${waiting} waiting`;
    if (last === 'auth') {
      Object.assign(out, { tone: 'warn', detail: 'Your session has ended. Sign in again to send them. Nothing is lost.', action: 'signin' });
    } else if (!online || last === 'offline') {
      Object.assign(out, { tone: 'info', detail: 'They will send when you are back online. Nothing is lost.', action: 'sync' });
    } else if (last === 'error') {
      Object.assign(out, { tone: 'warn', detail: 'The cloud refused the last send. They stay here and will be tried again.', action: 'sync' });
    } else {
      Object.assign(out, { tone: 'info', detail: 'Sending now.', action: 'sync' });
    }
  }
  const notes = [];
  if (rejected > 0) notes.push(`${plural(rejected, 'review')} could not be accepted (made too long ago, or with the wrong clock).`);
  if (!durable) notes.push('This browser is not keeping reviews between visits. Reviews made now are lost if the page closes before they sync.');
  if (notes.length) {
    out.detail = [out.detail, ...notes].filter(Boolean).join(' ');
    if (out.tone === 'ok') out.tone = 'warn';
  }
  if (waiting === 0 && !notes.length && last === 'auth') {
    Object.assign(out, { tone: 'warn', detail: 'Your session has ended.', action: 'signin' });
  }
  return out;
}

/**
 * The short thing for the header link: the one most pressing fact, or "Offline & sync" when
 * all is well. `updateReady` is a new version of the app waiting for a reload.
 */
export function headerStatus({ mode, outbox, updateReady }) {
  if (mode === 'signed-out' || (outbox && outbox.action === 'signin')) return { text: 'Sign in again', tone: 'warn' };
  if (updateReady) return { text: 'Update ready', tone: 'info' };
  if (mode === 'offline') return { text: outbox && outbox.short ? `Offline · ${outbox.short}` : 'Offline', tone: 'info' };
  if (outbox && outbox.short) return { text: outbox.short, tone: 'info' };
  return { text: 'Offline & sync', tone: 'quiet' };
}

/** How to install, for the page that explains it. */
export function installHint({ standalone, canPrompt, ios }) {
  if (standalone) return { installed: true, text: 'Installed. It opens from your home screen.' };
  if (canPrompt) return { installed: false, text: 'Install it to open from your home screen, full screen.', prompt: true };
  if (ios) return { installed: false, text: 'In Safari, tap Share, then Add to Home Screen.' };
  return { installed: false, text: 'In your browser menu, choose Install app or Add to Home Screen.' };
}
