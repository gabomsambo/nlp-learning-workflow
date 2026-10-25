/* The four pages. Each takes the store and returns an element to mount. */
import { bullets, callout, chip, emptyState, h } from './dom.js';
import { asSnapshotCards } from './schedule.js';
import {
  PAPER_TABS, authorLine, dueSummary, filterPapers, formatDate, formatDuration, greeting,
  newestFirst, paperHref, pillarGlance, plural, readableScript, relativeTime, safePdfHref,
} from './lib.js';

const pillarClass = (id) => (id ? `pillar-${id}` : '');

function pageHeader(title, eyebrow) {
  return h('header', { class: 'page-header' },
    h('div', null,
      eyebrow ? h('p', { class: 'page-eyebrow', text: eyebrow }) : null,
      h('h1', { class: 'page-title', text: title })));
}

function paperRow(paper, pillars) {
  const pillar = pillars.get(paper.pillar_id);
  const meta = [paper.year, authorLine(paper.authors, 2)].filter(Boolean).join(' · ');
  return h('li', { class: 'library-row' },
    h('a', { class: 'library-row-open', href: paperHref(paper.id) },
      h('time', { class: 'list-row-date', text: formatDate(paper.added_at) }),
      h('span', { class: 'library-row-main' },
        h('span', { class: 'list-row-title', text: paper.title }),
        h('span', { class: 'list-row-meta' },
          meta,
          meta && pillar ? ' · ' : '',
          pillar ? h('span', { class: `pillar-inline ${pillarClass(pillar.id)}` },
            h('span', { class: 'pillar-dot', 'aria-hidden': 'true' }), pillar.name) : null))),
    h('div', { class: 'library-row-status' },
      paper.has_lesson ? chip('Lesson', 'ok') : null,
      paper.card_count > 0 ? chip('Cards', 'accent') : null,
      paper.episode_ids && paper.episode_ids.length ? chip('Episode') : null));
}

export async function todayView(store, reviews, now = Date.now()) {
  const [library, published] = await Promise.all([store.library(), store.schedule()]);
  const pillars = new Map(library.pillars.map((p) => [p.id, p]));
  // The cards as this device holds them: the published schedule with the reviews made since on top.
  const schedule = { ...published, cards: asSnapshotCards(await reviews.states()) };
  const cards = dueSummary(schedule.cards, now);
  const hour = new Date(now).getHours();

  const summary = cards.total === 0
    ? 'No cards yet.'
    : `${plural(cards.due, 'card')} due, ${cards.fresh} new, ${cards.upcoming} scheduled.`;

  const recentPapers = newestFirst(library.papers, 'added_at').slice(0, 5);
  const episodes = newestFirst(library.episodes, 'created_at').slice(0, 3);
  const glance = pillarGlance(library, schedule, now);

  return h('div', { class: 'today-page' },
    h('header', { class: 'today-hello' },
      h('p', { class: 'today-date', text: new Date(now).toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long' }) }),
      h('h1', { class: 'today-greet', text: greeting(hour) }),
      h('p', { class: 'today-summary', text: summary })),

    h('section', { class: 'today-section', 'aria-labelledby': 'review-heading' },
      h('div', { class: 'section-head' }, h('h2', { id: 'review-heading', class: 'section-title', text: 'Review' })),
      h('p', { class: 'section-lede', text: cards.due + cards.fresh === 0
        ? 'Nothing is due. Cards you review here reach the computer when you are online.'
        : 'Review here, even without a connection. Your answers reach the computer when you are online.' }),
      h('a', { class: 'btn btn-primary btn-md', href: '#/review', text: cards.due + cards.fresh === 0 ? 'Open review' : `Review ${cards.due + cards.fresh > 99 ? 'cards' : plural(cards.due + cards.fresh, 'card')}` })),

    h('section', { class: 'today-section', 'aria-labelledby': 'recent-heading' },
      h('div', { class: 'section-head' },
        h('h2', { id: 'recent-heading', class: 'section-title', text: 'Recently added' }),
        h('a', { href: '#/library', text: 'Library' })),
      h('div', { class: 'activity-list' }, recentPapers.map((paper) =>
        h('a', { class: 'activity-item activity-item-link', href: paperHref(paper.id) },
          h('span', { class: 'activity-main' },
            h('span', { class: 'activity-title', text: paper.title }),
            h('span', { class: 'activity-sub', text: [pillars.get(paper.pillar_id)?.name, authorLine(paper.authors, 2)].filter(Boolean).join(' · ') })),
          h('span', { class: 'activity-meta', title: formatDate(paper.added_at), text: relativeTime(paper.added_at, now) }))))),

    h('section', { class: 'today-section', 'aria-labelledby': 'episodes-heading' },
      h('div', { class: 'section-head' },
        h('h2', { id: 'episodes-heading', class: 'section-title', text: 'New episodes' })),
      episodes.length === 0
        ? h('p', { class: 'section-lede', text: 'No episodes yet.' })
        : h('ul', { class: 'episode-list' }, episodes.map((episode) =>
          h('li', { class: 'episode-row' },
            h('a', { class: 'episode-copy', href: paperHref(episode.paper_id, 'episode') },
              h('span', { class: 'episode-title', text: episode.title }),
              h('span', { class: 'episode-meta', text: [pillars.get(episode.pillar_id)?.name, formatDuration(episode.duration_seconds), relativeTime(episode.created_at, now)].filter(Boolean).join(' · ') })))))),

    h('section', { class: 'today-section', 'aria-labelledby': 'glance-heading' },
      h('div', { class: 'section-head' },
        h('h2', { id: 'glance-heading', class: 'section-title', text: 'Pillars' })),
      h('div', { class: 'glance-grid' }, glance.map((tile) =>
        h('a', { class: `pillar-tile ${pillarClass(tile.id)}${tile.papers === 0 ? ' is-empty' : ''}`, href: `#/library?pillar=${encodeURIComponent(tile.id)}` },
          h('span', { class: 'pillar-tile-name', text: tile.name }),
          h('span', { class: 'pillar-tile-stats' },
            ...[['paper', tile.papers], ['lesson', tile.lessons], ['card', tile.cards]].map(([label, n]) =>
              h('span', null, h('span', { class: 'pillar-stat-n', text: String(n) }), ` ${n === 1 ? label : `${label}s`}`)),
            h('span', null, h('span', { class: 'pillar-stat-n', text: String(tile.due) }), ' due'))))))
  );
}

export async function libraryView(store, params = new URLSearchParams()) {
  const library = await store.library();
  const pillars = new Map(library.pillars.map((p) => [p.id, p]));
  const state = { query: '', pillar: params.get('pillar') || '', status: '' };

  const list = h('ul', { class: 'library-list' });
  const none = h('div', { class: 'empty', hidden: true },
    h('p', { class: 'empty-title', text: 'No papers match' }),
    h('p', { class: 'empty-message', text: 'Clear the filters to see the rest of the library.' }));
  const count = h('span', { id: 'library-count' });

  function paint() {
    const rows = filterPapers(newestFirst(library.papers, 'added_at'), state);
    list.replaceChildren(...rows.map((paper) => paperRow(paper, pillars)));
    count.textContent = plural(rows.length, 'paper');
    none.hidden = rows.length !== 0;
    for (const button of root.querySelectorAll('[data-pillar]')) button.setAttribute('aria-pressed', String(button.dataset.pillar === state.pillar));
    for (const button of root.querySelectorAll('[data-status]')) button.setAttribute('aria-pressed', String(button.dataset.status === state.status));
  }

  const pillarChips = [{ id: '', name: 'All pillars' }, ...library.pillars].map((p) =>
    h('button', { type: 'button', class: `filter-chip ${pillarClass(p.id)}`, 'data-pillar': p.id, 'aria-pressed': 'false' },
      p.id ? h('span', { class: 'pillar-dot', 'aria-hidden': 'true' }) : null, p.name));
  const statusChips = [['', 'All'], ['lesson', 'Has lesson'], ['cards', 'Has cards'], ['episode', 'Has episode']].map(([value, label]) =>
    h('button', { type: 'button', class: 'filter-chip', 'data-status': value, 'aria-pressed': 'false', text: label }));

  const search = h('input', { type: 'search', id: 'library-search', placeholder: 'Title, author, or id', autocomplete: 'off' });
  search.addEventListener('input', () => { state.query = search.value; paint(); });

  const root = h('div', { class: 'library-page' },
    pageHeader('Library', 'Papers'),
    h('div', { class: 'library-toolbar' },
      h('div', { class: 'field library-search' }, h('label', { for: 'library-search', text: 'Search' }), search),
      h('div', { class: 'filter-row' },
        h('div', { class: 'filter-chips', role: 'group', 'aria-label': 'Pillar' }, pillarChips),
        h('div', { class: 'filter-chips', role: 'group', 'aria-label': 'Status' }, statusChips)),
      h('p', { class: 'library-hint' }, count)),
    list, none);
  root.addEventListener('click', (event) => {
    const button = event.target.closest('button');
    if (!button) return;
    if ('pillar' in button.dataset) state.pillar = button.dataset.pillar;
    else if ('status' in button.dataset) state.status = button.dataset.status;
    else return;
    paint();
  });
  paint();
  return root;
}

function lessonBlocks(lesson) {
  return h('article', { class: 'lesson-block' },
    lesson.estimated_time ? h('p', { class: 'lesson-meta' }, h('span', { class: 'paper-muted', text: `${lesson.estimated_time} min read` })) : null,
    lesson.tl_dr ? callout('tldr', 'In one paragraph', lesson.tl_dr) : null,
    lesson.takeaways?.length ? [h('h3', { text: 'Takeaways' }), bullets(lesson.takeaways)] : null,
    lesson.practice_ideas?.length ? [h('h3', { text: 'Try it yourself' }), bullets(lesson.practice_ideas)] : null,
    lesson.connections?.length ? [h('h3', { text: 'Connections' }), bullets(lesson.connections)] : null);
}

function noteBlocks(note) {
  const section = (title, body) => (body && body.length ? [h('h3', { text: title }), Array.isArray(body) ? bullets(body) : h('p', { text: body })] : null);
  return h('article', { class: 'card note-card' },
    section('Problem', note.problem), section('Method', note.method),
    section('Key findings', note.findings), section('Limitations', note.limitations),
    section('Future work', note.future_work),
    note.key_terms?.length ? [h('h3', { text: 'Key terms' }), h('div', { class: 'term-row' }, note.key_terms.map((t) => chip(t)))] : null,
    note.confidence_score ? h('p', { class: 'paper-muted', text: `Confidence ${note.confidence_score}` }) : null);
}

export async function paperView(store, id, tab) {
  const library = await store.library();
  const entry = library.papers.find((p) => p.id === id);
  if (!entry) {
    return h('div', { class: 'paper-page' }, pageHeader('Paper not found', 'Library'),
      emptyState('This paper is not in the library.', id, h('a', { class: 'btn btn-primary btn-md', href: '#/library', text: 'Back to the library' })));
  }
  const paper = await store.doc(entry.doc);
  const pillar = library.pillars.find((p) => p.id === paper.pillar_id);
  const episodes = await Promise.all((entry.episode_ids || []).map((eid) => {
    const e = library.episodes.find((x) => x.id === eid);
    return e ? store.doc(e.doc) : null;
  }));
  const related = (paper.related || [])
    .map((r) => ({ ...r, entry: library.papers.find((p) => p.id === r.paper_id) }))
    .filter((r) => r.entry);
  const concepts = (paper.concepts || [])
    .map((c) => library.concepts.find((x) => x.id === c.id))
    .filter(Boolean);
  const pdf = safePdfHref(paper.url_pdf);
  const labels = {
    summary: 'Summary',
    notes: 'Notes',
    lesson: 'Lesson',
    cards: paper.quiz_cards?.length ? `Cards · ${paper.quiz_cards.length}` : 'Cards',
    episode: episodes.length ? `Episode · ${episodes.length}` : 'Episode',
  };

  const panels = {
    summary: h('section', { class: 'paper-panel paper-prose' },
      paper.lesson?.tl_dr ? callout('tldr', 'In one paragraph', paper.lesson.tl_dr) : null,
      paper.abstract ? [h('h3', { text: 'Abstract' }), h('p', { text: paper.abstract })] : null,
      paper.note?.problem ? [h('h3', { text: 'The problem' }), h('p', { text: paper.note.problem })] : null,
      paper.note?.method ? [h('h3', { text: 'What they did' }), h('p', { text: paper.note.method })] : null,
      paper.note?.findings?.length ? [h('h3', { text: 'Key findings' }), h('ol', null, paper.note.findings.map((f) => h('li', { text: f })))] : null,
      concepts.length ? [h('h3', { text: 'Concepts' }), h('div', { class: 'term-row' }, concepts.map((c) => chip(c.name)))] : null,
      related.length ? [h('h3', { text: 'Related papers' }),
        h('ul', null, related.map((r) => h('li', null, h('a', { href: paperHref(r.paper_id), text: r.entry.title }))))] : null,
      !paper.abstract && !paper.note && !paper.lesson ? h('p', { class: 'paper-muted', text: 'No summary has been written for this paper yet.' }) : null),
    notes: h('section', { class: 'paper-panel paper-prose' }, h('h2', { text: 'Notes' }),
      paper.note ? noteBlocks(paper.note) : h('p', { class: 'paper-muted', text: 'No research notes for this paper yet.' })),
    lesson: h('section', { class: 'paper-panel paper-prose' }, h('h2', { text: 'Lesson' }),
      paper.lesson ? lessonBlocks(paper.lesson) : h('p', { class: 'paper-muted', text: 'No lesson for this paper yet.' })),
    cards: h('section', { class: 'paper-panel' }, h('h2', { text: 'Cards' }),
      paper.quiz_cards?.length
        ? h('div', { class: 'card-list' }, paper.quiz_cards.map((card) =>
          h('article', { class: 'card quiz-read' },
            h('p', { class: 'quiz-question', text: card.question }),
            h('p', { class: 'quiz-answer', text: card.answer }))))
        : h('p', { class: 'paper-muted', text: 'No cards for this paper yet.' })),
    episode: h('section', { class: 'paper-panel' }, h('h2', { text: 'Episode' }),
      episodes.filter(Boolean).length
        ? episodes.filter(Boolean).map((episode) => h('article', { class: 'card' },
          h('h3', { class: 'card-title', text: episode.title }),
          h('p', { class: 'list-row-meta', text: [`${episode.word_count || 0} words`, formatDate(episode.created_at)].filter(Boolean).join(' · ') }),
          h('p', { class: 'paper-muted', text: 'Audio is not available in the reader yet. The script is below.' }),
          episode.key_points?.length ? [h('h3', { text: 'Key points' }), bullets(episode.key_points)] : null,
          h('details', null, h('summary', { text: 'Script' }), h('div', { class: 'reader-script', text: readableScript(episode.script) }))))
        : h('p', { class: 'paper-muted', text: 'No episode for this paper yet.' })),
  };
  panels[tab].classList.add('is-active');

  return h('div', { class: 'paper-page', 'data-paper-id': paper.id },
    h('nav', { class: 'crumb', 'aria-label': 'Breadcrumb' },
      h('a', { href: '#/library', text: 'Library' }),
      pillar ? [h('span', { 'aria-hidden': 'true', text: '/' }),
        h('a', { class: `crumb-pillar ${pillarClass(pillar.id)}`, href: `#/library?pillar=${encodeURIComponent(pillar.id)}` },
          h('span', { class: 'pillar-dot', 'aria-hidden': 'true' }), pillar.name)] : null),
    h('header', { class: 'paper-head' },
      h('p', { class: 'paper-kicker' },
        h('span', { class: 'paper-id', text: paper.id }),
        paper.year ? [h('span', { 'aria-hidden': 'true', text: '·' }), h('span', { text: String(paper.year) })] : null,
        paper.venue ? [h('span', { 'aria-hidden': 'true', text: '·' }), h('span', { text: paper.venue })] : null),
      h('h1', { class: 'paper-title', text: paper.title }),
      paper.authors?.length ? h('p', { class: 'paper-authors', text: authorLine(paper.authors, 8) }) : null,
      pdf ? h('div', { class: 'page-actions' }, h('a', { class: 'btn btn-secondary btn-md', href: pdf, target: '_blank', rel: 'noopener noreferrer', text: 'Open PDF' })) : null),
    h('nav', { class: 'tabs', 'aria-label': 'Paper' }, PAPER_TABS.map((name) =>
      h('a', { href: paperHref(paper.id, name), 'aria-current': name === tab ? 'page' : null, text: labels[name] }))),
    h('div', { class: 'paper-panels' }, Object.values(panels)));
}

export async function lessonsView(store) {
  const library = await store.library();
  const pillars = new Map(library.pillars.map((p) => [p.id, p]));
  const withLesson = newestFirst(library.papers.filter((p) => p.has_lesson), 'added_at');
  const docs = await Promise.all(withLesson.map((p) => store.doc(p.doc)));

  return h('div', { class: 'lessons-page' },
    pageHeader('Lessons', 'Library'),
    h('p', { class: 'library-hint' }, h('span', { text: plural(withLesson.length, 'lesson') }), h('span', { text: 'Newest first' })),
    h('div', { class: 'lessons-column' },
      withLesson.length === 0
        ? emptyState('No lessons yet', 'They show up after a paper is processed.')
        : h('ul', { class: 'lesson-list' }, withLesson.map((paper, i) => {
          const lesson = docs[i].lesson || {};
          const pillar = pillars.get(paper.pillar_id);
          const preview = (lesson.takeaways || []).slice(0, 2);
          const more = (lesson.takeaways || []).length - preview.length;
          return h('li', { class: 'lesson-row' },
            h('a', { class: 'lesson-open', href: paperHref(paper.id, 'lesson') },
              h('time', { class: 'lesson-date', text: formatDate(lesson.created_at || paper.added_at) }),
              h('span', { class: 'lesson-main' },
                h('h2', { class: 'lesson-title', text: paper.title }),
                pillar ? h('p', { class: `lesson-pillar ${pillarClass(pillar.id)}` },
                  h('span', { class: 'pillar-dot', 'aria-hidden': 'true' }), pillar.name) : null,
                preview.length ? h('ul', { class: 'lesson-preview' }, preview.map((t) => h('li', { text: t }))) : null,
                more > 0 ? h('span', { class: 'lesson-more', text: `+${more} more` }) : null,
                lesson.estimated_time ? h('p', { class: 'lesson-facts' }, h('span', { text: `${lesson.estimated_time} min read` })) : null)));
        }))));
}
