/* Pure helpers for the reader. No DOM, no network: node --test imports this file. */

export const ROUTES = ['today', 'library', 'lessons', 'review'];

/** `#/paper/<id>/<tab>` and friends. Ids carry `:` and `/`, so they are percent-encoded. */
export function parseRoute(hash) {
  const raw = String(hash || '').replace(/^#\/?/, '');
  const parts = raw.split('/').filter(Boolean);
  if (parts.length === 0) return { name: 'today' };
  if (parts[0] === 'library' && parts.length === 1) return { name: 'library' };
  if (parts[0] === 'lessons' && parts.length === 1) return { name: 'lessons' };
  if (parts[0] === 'review' && parts.length === 1) return { name: 'review' };
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

export const PAPER_TABS = ['summary', 'notes', 'lesson', 'cards', 'episode'];

export function paperHref(id, tab) {
  const base = `#/paper/${encodeURIComponent(id)}`;
  return tab && tab !== 'summary' ? `${base}/${tab}` : base;
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
