/* Pure helpers for the reader. No DOM, no network: node --test imports this file. */

export const ROUTES = ['today', 'library', 'lessons', 'listen', 'review'];

/** `#/paper/<id>/<tab>` and friends. Ids carry `:` and `/`, so they are percent-encoded. */
export function parseRoute(hash) {
  const raw = String(hash || '').replace(/^#\/?/, '');
  const parts = raw.split('/').filter(Boolean);
  if (parts.length === 0) return { name: 'today' };
  if (parts[0] === 'library' && parts.length === 1) return { name: 'library' };
  if (parts[0] === 'lessons' && parts.length === 1) return { name: 'lessons' };
  if (parts[0] === 'review' && parts.length === 1) return { name: 'review' };
  if (parts[0] === 'listen' && parts.length === 1) return { name: 'listen' };
  if (parts[0] === 'listen' && parts.length === 2) {
    try {
      return { name: 'episode', id: decodeURIComponent(parts[1]) };
    } catch (error) {
      return { name: 'notfound' };
    }
  }
  if (parts[0] === 'paper' && parts.length >= 2 && parts.length <= 3) {
    try {
      const id = decodeURIComponent(parts[1]);
      const tab = parts[2] && PAPER_TABS.includes(parts[2]) ? parts[2] : 'summary';
      if (parts[2] && tab !== parts[2]) return { name: 'notfound' };
      return { name: 'paper', id, tab };
    } catch (error) {
      return { name: 'notfound' };
    }
  }
  return { name: 'notfound' };
}

export const PAPER_TABS = ['summary', 'notes', 'lesson', 'cards', 'episode', 'playground'];

export function paperHref(id, tab) {
  const base = `#/paper/${encodeURIComponent(id)}`;
  return tab && tab !== 'summary' ? `${base}/${tab}` : base;
}

export function episodeHref(id) {
  return `#/listen/${encodeURIComponent(id)}`;
}

/** `audio/<sha256>.mp3` as the Worker serves it, or null for any other shape. */
export function audioSrc(key) {
  const match = /^audio\/([0-9a-f]{64}\.mp3)$/.exec(typeof key === 'string' ? key : '');
  return match ? `/api/audio/${match[1]}` : null;
}

export async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

const toMs = (iso) => {
  const ms = Date.parse(iso);
  return Number.isNaN(ms) ? null : ms;
};

/** What the schedule snapshot says about the cards, at `nowMs`. */
export function dueSummary(cards, nowMs) {
  const out = { total: 0, due: 0, fresh: 0, upcoming: 0 };
  for (const card of cards || []) {
    out.total += 1;
    if (card.state === 'new') {
      out.fresh += 1;
      continue;
    }
    const due = toMs(card.due);
    if (due !== null && due <= nowMs) out.due += 1;
    else out.upcoming += 1;
  }
  return out;
}

/** Per-pillar counts for the glance tiles. */
export function pillarGlance(library, schedule, nowMs) {
  const pillarOf = new Map((library.papers || []).map((p) => [p.id, p.pillar_id]));
  const dueBy = new Map();
  for (const card of (schedule && schedule.cards) || []) {
    if (card.state === 'new') continue;
    const due = toMs(card.due);
    if (due !== null && due <= nowMs) {
      const pillar = pillarOf.get(card.paper_id);
      dueBy.set(pillar, (dueBy.get(pillar) || 0) + 1);
    }
  }
  return (library.pillars || []).map((pillar) => {
    const papers = (library.papers || []).filter((p) => p.pillar_id === pillar.id);
    return {
      id: pillar.id,
      name: pillar.name,
      papers: papers.length,
      lessons: papers.filter((p) => p.has_lesson).length,
      cards: papers.reduce((sum, p) => sum + (p.card_count || 0), 0),
      due: dueBy.get(pillar.id) || 0,
    };
  });
}

export function newestFirst(items, field) {
  return [...items].sort((a, b) => String(b[field] || '').localeCompare(String(a[field] || '')));
}

/** Library filter. `status` is "", "lesson", "cards" or "episode". */
export function filterPapers(papers, { query = '', pillar = '', status = '' } = {}) {
  const needle = query.trim().toLowerCase();
  return papers.filter((paper) => {
    if (pillar && paper.pillar_id !== pillar) return false;
    if (status === 'lesson' && !paper.has_lesson) return false;
    if (status === 'cards' && !(paper.card_count > 0)) return false;
    if (status === 'episode' && !(paper.episode_ids && paper.episode_ids.length > 0)) return false;
    if (!needle) return true;
    const haystack = [paper.title, paper.id, paper.venue, (paper.authors || []).join(' ')].join(' ').toLowerCase();
    return haystack.includes(needle);
  });
}

export function greeting(hour) {
  if (hour < 12) return 'Good morning.';
  if (hour < 17) return 'Good afternoon.';
  return 'Good evening.';
}

export function plural(n, one, many = `${one}s`) {
  return `${n} ${n === 1 ? one : many}`;
}

export function formatDate(iso) {
  const ms = toMs(iso);
  if (ms === null) return '';
  return new Date(ms).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
}

export function relativeTime(iso, nowMs) {
  const ms = toMs(iso);
  if (ms === null) return '';
  const minutes = Math.round((nowMs - ms) / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return plural(hours, 'hour') + ' ago';
  const days = Math.round(hours / 24);
  if (days < 30) return plural(days, 'day') + ' ago';
  return formatDate(iso);
}

export function formatDuration(seconds) {
  if (typeof seconds !== 'number' || !(seconds > 0)) return '';
  const minutes = Math.round(seconds / 60);
  return minutes < 1 ? 'under a minute' : `${minutes} min`;
}

export function authorLine(authors, limit = 4) {
  if (!authors || authors.length === 0) return '';
  if (authors.length <= limit) return authors.join(', ');
  return `${authors.slice(0, limit).join(', ')} and ${authors.length - limit} more`;
}

/** An https link only. A stored `file://` upload path is not a link a phone can open. */
export function safePdfHref(url) {
  if (typeof url !== 'string') return null;
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'https:' || parsed.protocol === 'http:' ? parsed.href : null;
  } catch (error) {
    return null;
  }
}

/** Script cues like [MUSIC] and the [HOST]: tag, shown as plain reading text. */
export function readableScript(script) {
  return String(script || '')
    .split('\n')
    .map((line) => line.replace(/^\s*\[(?:HOST|GUEST|[A-Z ]+)\]:\s*/, ''))
    .filter((line) => !/^\s*\[(?:MUSIC|SFX|PAUSE|TRANSITION)[^\]]*\]\s*$/.test(line))
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/** The episodes that can be listened to come first, newest first within each group. */
export function listenRows(library) {
  const rows = newestFirst(library.episodes || [], 'created_at');
  const withAudio = rows.filter((e) => audioSrc(e.audio_key));
  return [...withAudio, ...rows.filter((e) => !audioSrc(e.audio_key))];
}

/** What a published fact check lets the page say. A stale check is never painted onto a changed script. */
export function factcheckState(episode) {
  const check = episode && episode.fact_check;
  if (!check) return { kind: 'none', claims: [] };
  const claims = Array.isArray(check.claims) ? check.claims : [];
  return { kind: check.stale ? 'stale' : 'done', claims, check };
}

/** Where to start: a saved position, unless it is at the very start or the very end. */
export function resumeSeconds(saved, duration) {
  const seconds = Number(saved);
  if (!Number.isFinite(seconds) || seconds < 5) return 0;
  if (Number.isFinite(duration) && duration > 0 && seconds > duration - 10) return 0;
  return seconds;
}

export const PLAYBACK_RATES = [0.75, 1, 1.25, 1.5, 1.75, 2];

export function rateOrDefault(value) {
  const rate = Number(value);
  return PLAYBACK_RATES.includes(rate) ? rate : 1;
}

/** The notice above the text, or '' when the highlight is working as it should. */
export function readAlongNote({ hasAudio, timelineState, wordLevel }) {
  if (!hasAudio) return 'This episode has no audio yet. The script is below.';
  if (timelineState === 'missing') return 'No read-along timing for this episode yet. It still plays, but the text will not follow the audio.';
  if (timelineState === 'stale') return 'The script was edited after this audio was recorded, so the highlight is off.';
  if (!wordLevel) return 'Word timing was weak for this episode, so only the line being spoken is highlighted.';
  return '';
}

export function episodeMeta(episode, pillarName) {
  return [pillarName, formatDuration(episode.duration_seconds), formatDate(episode.created_at)].filter(Boolean).join(' · ');
}

/** The letter on an episode's cover. Episodes are all titled "Deep Dive: <paper>", so skip that. */
export function episodeInitial(title) {
  const match = /[0-9A-Za-z]/.exec(String(title || '').replace(/^\s*deep dive\s*:\s*/i, ''));
  return match ? match[0].toUpperCase() : 'E';
}

// ---- Playground (read-only; docs/PLAYGROUND.md) ------------------------------

/** One cell on the Playground tab, the target of the code drawer's card. */
export function playgroundCellHref(paperId, cellIndex) {
  return `${paperHref(paperId, 'playground')}?cell=${Number(cellIndex)}`;
}

/** The cell a link asked for (`?cell=12`), or null. */
export function requestedCell(params) {
  const raw = params && typeof params.get === 'function' ? params.get('cell') : null;
  if (raw == null || !/^[0-9]{1,5}$/.test(raw)) return null;
  return Number(raw);
}

/**
 * A published output in the shape playground.js's renderOutput reads. The
 * export calls output text `body` (`text` is a key no document may carry).
 * null for one the size caps left out.
 */
export function outputForRender(out) {
  if (!out || typeof out !== 'object' || out.kind === 'dropped') return null;
  if (!('body' in out)) return out;
  const { body, ...rest } = out;
  return { ...rest, text: body };
}

function sizeLabel(bytes) {
  if (!(bytes > 0)) return '';
  return bytes >= 1e6 ? ` (${(bytes / 1e6).toFixed(1)} MB)` : ` (${Math.max(1, Math.round(bytes / 1000))} KB)`;
}

/** Where a capped output was. */
export function droppedNote(out) {
  const what = out && out.what === 'figure' ? 'This figure' : 'This output';
  return `${what}${sizeLabel(out && out.bytes)} was left out of the phone copy to keep it small. It is in the notebook on your computer.`;
}

/** One line for the top of the tab when the caps left anything out, else ''. */
export function droppedSummary(dropped) {
  const figures = Number(dropped && dropped.figures) || 0;
  const outputs = Number(dropped && dropped.outputs) || 0;
  if (!figures && !outputs) return '';
  const parts = [figures ? plural(figures, 'figure') : '', outputs ? plural(outputs, 'output') : ''].filter(Boolean);
  return `To keep the phone copy small, ${parts.join(' and ')} ${figures + outputs === 1 ? 'was' : 'were'} left out. They are in the notebook on your computer.`;
}

/**
 * What the code drawer under the player can do for an episode, decided by the
 * desktop panel's own `FollowAlong.availability` (passed in, as the page loads
 * it as a classic script). `none` when the episode follows no notebook.
 */
export function codeDrawerState(FA, { hasAudio, timeline, scriptSha, links }) {
  const list = Array.isArray(links) ? links : [];
  if (!list.length || !FA) return { mode: 'none' };
  const state = FA.availability({ hasAudio, timeline, scriptSha, links: list });
  // The desktop offers its picker here. Matching again is the computer's job.
  if (state.mode === 'pick') {
    return { mode: 'off', link: state.link, reason: 'The script changed after it was matched to this notebook. Match it again on the computer.' };
  }
  return state;
}

/**
 * What the drawer shows at `ms`, given what it shows now (`shown`, from the
 * previous answer's `at`). Pure in `ms`, like `FollowAlong.followAt`, so
 * seeking lands where playing would.
 *
 *   wait   before the first span: no card yet
 *   same   nothing changed
 *   dim    the same card, now inside or outside its span
 *   card   a different card: `span` is the one to show
 */
export function drawerStep(FA, model, spans, ms, shown) {
  const at = FA.followAt(model, spans, ms);
  if (at.index === -1) return { kind: 'wait', at };
  if (shown && shown.index === at.index) return { kind: shown.inside === at.inside ? 'same' : 'dim', at };
  return { kind: 'card', at, span: spans[at.index] };
}
