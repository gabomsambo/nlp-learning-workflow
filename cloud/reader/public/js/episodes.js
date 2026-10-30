/*
 * Listen: the episode list and one episode with read-along.
 *
 * The pure parts are shared with the desktop Listen page and are loaded as
 * classic scripts before this module (index.html): ReadAlong (timeline to
 * line and word), ListenCore (script lines, claim segments, token groups) and
 * Factcheck (verdict wording). What is here is the DOM and the audio element.
 *
 * Everything is createElement + textContent. The audio element is the one
 * source of time; the highlight follows it and never drives it, and so does
 * the code drawer under the player (follow.js).
 */
import { callout, emptyState, h } from './dom.js';
import { codeDrawer } from './follow.js';
import { saveControl, savedChip } from './offline-view.js';
import {
  PLAYBACK_RATES, audioSrc, episodeHref, episodeInitial, episodeMeta, factcheckState, formatDate, listenRows,
  paperHref, rateOrDefault, readAlongNote, resumeSeconds, sha256Hex,
} from './lib.js';

const RA = window.ReadAlong;
const Core = window.ListenCore;
const Factcheck = window.Factcheck;

// Where in the reading surface the spoken line settles: a little above the
// middle, the way a transcript keeps what is coming in view.
const READ_ANCHOR = 0.38;
const POSITION_KEY = (id) => `nlp:reader:pos:${id}`;
const RATE_KEY = 'nlp:reader:rate';

const pillarClass = (id) => (id ? `pillar-${id}` : '');

function stored(key) {
  try { return localStorage.getItem(key); } catch (error) { return null; }
}

function remember(key, value) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, String(value));
  } catch (error) { /* private window: the position is a convenience */ }
}


export async function listenView(store, offline) {
  const library = await store.library();
  const pillars = new Map(library.pillars.map((p) => [p.id, p]));
  const rows = listenRows(library);
  const papers = new Map(library.papers.map((p) => [p.id, p]));
  // Which episodes play with no connection, for the chip. Without the offline layer, none is marked.
  let saved = new Map();
  if (offline) {
    try { saved = new Map((await offline.episodes()).map((row) => [row.id, row])); } catch (error) { /* no chips */ }
  }

  return h('div', { class: 'listen-page' },
    h('header', { class: 'page-header' },
      h('div', null,
        h('p', { class: 'page-eyebrow', text: 'Episodes' }),
        h('h1', { class: 'page-title', text: 'Listen' }))),
    rows.length === 0
      ? emptyState('No episodes yet', 'They show up here after an episode is generated on the computer.')
      : h('ul', { class: 'episode-list reader-episodes' }, rows.map((episode) => {
        const pillar = pillars.get(episode.pillar_id);
        const playable = !!audioSrc(episode.audio_key);
        const paper = papers.get(episode.paper_id);
        return h('li', { class: 'episode-row' },
          h('span', { class: `episode-cover ${pillarClass(episode.pillar_id)}`, 'aria-hidden': 'true' }, episodeInitial(episode.title)),
          h('div', { class: 'episode-copy' },
            h('a', { class: 'episode-title', href: episodeHref(episode.id), text: episode.title }),
            h('span', { class: 'episode-meta', text: episodeMeta(episode, pillar?.name) }),
            paper ? h('span', { class: 'episode-meta reader-episode-paper', text: paper.title }) : null,
            h('span', { class: 'reader-episode-tags' },
              playable ? null : h('span', { class: 'chip chip-neutral', text: 'No audio' }),
              playable && episode.has_timeline ? h('span', { class: 'chip chip-accent', text: 'Read along' }) : null,
              savedChip(saved, episode.id))));
      })));
}

/** One claim's explanation, as in the desktop popover. */
function claimExplanation(parent, claim) {
  const presentation = Factcheck.claimPresentation(claim);
  parent.append(h('strong', { class: `factcheck-popover-title is-${presentation.tone}`, text: `${presentation.icon} ${presentation.label}` }));
  if (claim.claim) parent.append(h('p', { class: 'factcheck-claim', text: `“${claim.claim}”` }));
  if (claim.rationale) parent.append(h('p', { text: claim.rationale }));
  const evidence = Array.isArray(claim.evidence) ? claim.evidence.filter((item) => item && item.quote) : [];
  for (const item of evidence) {
    parent.append(h('blockquote', null, `“${item.quote}”`,
      item.chunk_index != null ? h('small', { text: `Paper text · chunk ${item.chunk_index}` }) : null));
  }
  if (!evidence.length && claim.verdict !== 'not_checkable') {
    parent.append(h('p', { class: 'factcheck-evidence-missing', text: 'No evidence quote was located in the stored paper text.' }));
  }
}

export async function episodeView(store, id, offline) {
  const library = await store.library();
  const entry = library.episodes.find((e) => e.id === id);
  if (!entry) {
    return h('div', { class: 'listen-page' },
      h('header', { class: 'page-header' }, h('div', null, h('p', { class: 'page-eyebrow', text: 'Listen' }), h('h1', { class: 'page-title', text: 'Episode not found' }))),
      emptyState('This episode is not in the library.', id, h('a', { class: 'btn btn-primary btn-md', href: '#/listen', text: 'Back to Listen' })));
  }
  const episode = await store.doc(entry.doc);
  const paper = library.papers.find((p) => p.id === episode.paper_id);
  const pillar = library.pillars.find((p) => p.id === episode.pillar_id);
  const lines = Core.scriptLines(episode.script);
  const src = audioSrc(episode.audio && episode.audio.key);
  const scriptSha = await sha256Hex(new TextEncoder().encode(episode.script || ''));
  const timeline = src ? episode.timeline : null;
  const timelineState = src ? RA.timelineState(timeline, scriptSha) : 'missing';
  const model = RA.buildModel(timeline, lines, scriptSha);
  const check = factcheckState(episode);
  const durationHint = Number(episode.audio && episode.audio.duration_seconds) || 0;
  let audio = null;
  // Null when the episode follows no notebook.
  const drawer = codeDrawer({
    store, episode, model, scriptSha, timeline, hasAudio: !!src,
    now: () => ({
      ms: audio && Number.isFinite(audio.currentTime) ? audio.currentTime * 1000 : 0,
      playing: !!audio && !audio.paused && !audio.ended,
    }),
  });

  const state = {
    following: true,
    programmaticUntil: 0,
    active: { line: -1, word: -1 },
    showSupported: false,
    popover: null,
    popoverSeq: 0,
    raf: 0,
    pendingSeek: null,
    scrubbing: false,
    lastSaved: 0,
  };
  const lineEls = {};
  const wordEls = {};

  // ---- the audio element and its controls ---------------------------------
  audio = src ? h('audio', { preload: 'metadata', src }) : null;
  const surface = h('div', { class: 'readalong-surface', tabindex: '0', 'aria-label': 'Episode text, highlighted as it is spoken' });
  const note = h('p', { class: 'readalong-note', role: 'status', hidden: true });
  const nowButton = h('button', { type: 'button', class: 'readalong-now', hidden: true, text: 'Jump to now' });

  const toggle = h('button', { type: 'button', class: 'readalong-ctl readalong-play', 'aria-label': 'Play', text: 'Play' });
  const back = h('button', { type: 'button', class: 'readalong-ctl', 'aria-label': 'Back 15 seconds', text: '−15' });
  const forward = h('button', { type: 'button', class: 'readalong-ctl', 'aria-label': 'Forward 30 seconds', text: '+30' });
  const range = h('input', { type: 'range', min: '0', max: String(durationHint || 0), step: '1', value: '0', 'aria-label': 'Position' });
  const clock = h('span', { class: 'readalong-time' });
  const rate = h('select', { 'aria-label': 'Playback speed' },
    PLAYBACK_RATES.map((r) => h('option', { value: String(r), text: `${r}×` })));
  rate.value = String(rateOrDefault(stored(RATE_KEY)));

  const duration = () => (audio && Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : durationHint);

  function paintClock() {
    if (!audio) return;
    const time = audio.currentTime || 0;
    const total = duration();
    clock.textContent = total > 0 ? `${Core.formatClock(time)} / ${Core.formatClock(total)}` : Core.formatClock(time);
    range.max = String(total || 0);
    if (!state.scrubbing) range.value = String(time);
    range.setAttribute('aria-valuetext', Core.formatValueText(time, total));
  }

  function savePosition(force) {
    if (!audio) return;
    const at = Date.now();
    if (!force && at - state.lastSaved < 5000) return;
    state.lastSaved = at;
    remember(POSITION_KEY(id), audio.ended ? null : Math.floor(audio.currentTime || 0));
  }

  function seekTo(seconds) {
    if (!audio || seconds == null) return;
    const target = Math.max(0, seconds);
    if (audio.readyState >= 1) {
      try { audio.currentTime = target; } catch (error) { state.pendingSeek = target; }
    } else {
      // Metadata has not arrived (Safari loads none until asked): apply it when it does.
      state.pendingSeek = target;
    }
    paintClock();
  }

  function applyPendingSeek() {
    if (state.pendingSeek === null || !audio || audio.readyState < 1) return;
    const target = state.pendingSeek;
    state.pendingSeek = null;
    try { audio.currentTime = target; } catch (error) { /* ignore */ }
    paintClock();
    updatePosition(true);
  }

  function play() {
    const pending = audio.play();
    if (pending && typeof pending.catch === 'function') pending.catch(() => {});
  }

  /** Seek and play from there, and let auto-scroll follow again. */
  function playFrom(seconds) {
    if (!audio || seconds == null) return;
    seekTo(seconds);
    setFollowing(true);
    play();
    updatePosition(true);
  }

  // ---- the reading surface -------------------------------------------------
  const reducedMotion = () => !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

  function scrollTo(el, instant) {
    if (!el) return;
    const box = surface.getBoundingClientRect();
    const rect = el.getBoundingClientRect();
    const offset = rect.top - box.top + surface.scrollTop;
    const top = RA.scrollTopFor(offset, rect.height, surface.clientHeight, surface.scrollHeight, READ_ANCHOR);
    state.programmaticUntil = Date.now() + 900;
    if (instant || reducedMotion() || !surface.scrollTo) surface.scrollTop = top;
    else surface.scrollTo({ top, behavior: 'smooth' });
  }

  /** "Jump to now" is for a reader who has scrolled away, not one still looking at it. */
  function refreshNowButton() {
    if (state.following || !model || state.active.line === -1) {
      nowButton.hidden = true;
      return;
    }
    const el = lineEls[model.lines[state.active.line].line];
    if (!el) {
      nowButton.hidden = true;
      return;
    }
    const box = surface.getBoundingClientRect();
    const rect = el.getBoundingClientRect();
    nowButton.hidden = rect.bottom > box.top + 16 && rect.top < box.bottom - 16;
  }

  function setFollowing(on) {
    state.following = on;
    refreshNowButton();
  }

  function updatePosition(force) {
    if (!model || !audio) return;
    let pos = { line: -1, word: -1 };
    const time = audio.currentTime;
    if (Number.isFinite(time)) pos = RA.positionAt(model, time * 1000);
    if (!force && pos.line === state.active.line && pos.word === state.active.word) return;
    const lineChanged = pos.line !== state.active.line;
    if (drawer) drawer.update(false);
    if (state.active.line !== -1) {
      const prev = model.lines[state.active.line].line;
      if (lineChanged && lineEls[prev]) lineEls[prev].classList.remove('is-current');
      const prevWords = wordEls[prev];
      if (prevWords && prevWords[state.active.word]) prevWords[state.active.word].classList.remove('is-now');
    }
    state.active = pos;
    if (pos.line === -1) {
      refreshNowButton();
      return;
    }
    const lineNo = model.lines[pos.line].line;
    const lineEl = lineEls[lineNo];
    if (lineEl) lineEl.classList.add('is-current');
    const words = wordEls[lineNo];
    if (words && pos.word !== -1 && words[pos.word]) words[pos.word].classList.add('is-now');
    if (lineChanged || force) {
      if (state.following) scrollTo(lineEl, false);
      else refreshNowButton();
    } else if (state.following && words && words[pos.word]) {
      // A long line wraps; keep the word in view, not only the line's top.
      const box = surface.getBoundingClientRect();
      const rect = words[pos.word].getBoundingClientRect();
      if (rect.bottom > box.bottom - 24 || rect.top < box.top + 24) scrollTo(words[pos.word], false);
    }
  }

  function loop() {
    state.raf = 0;
    updatePosition(false);
    if (audio && !audio.paused && !audio.ended) state.raf = requestAnimationFrame(loop);
  }

  function startLoop() {
    if (!state.raf) state.raf = requestAnimationFrame(loop);
  }

  // ---- fact-check marks ----------------------------------------------------
  function closePopover(returnFocus) {
    if (!state.popover) return;
    const { trigger, node } = state.popover;
    trigger.setAttribute('aria-expanded', 'false');
    trigger.removeAttribute('aria-controls');
    node.remove();
    state.popover = null;
    if (returnFocus) trigger.focus();
  }

  function openPopover(trigger, claims, row) {
    // A second tap on the same mark closes it: a popover that only opens is a trap.
    if (state.popover && state.popover.trigger === trigger) {
      closePopover(false);
      return;
    }
    closePopover(false);
    const popoverId = `factcheck-popover-${++state.popoverSeq}`;
    const close = h('button', { type: 'button', class: 'factcheck-popover-close', 'aria-label': 'Close evidence', text: '×' });
    close.addEventListener('click', () => closePopover(true));
    const popover = h('div', { id: popoverId, class: 'factcheck-popover', role: 'dialog', 'aria-label': 'Claim evidence' }, close);
    claims.forEach((claim, index) => {
      if (index) popover.append(h('hr'));
      claimExplanation(popover, claim);
    });
    const lineNo = Number(row.getAttribute('data-line'));
    if (model && model.byLine[lineNo] != null) {
      const playFromHere = h('button', { type: 'button', class: 'btn btn-ghost btn-sm factcheck-playfrom', text: 'Play from here' });
      playFromHere.addEventListener('click', () => playFrom(RA.seekSecondsForLine(model, lineNo)));
      popover.append(playFromHere);
    }
    trigger.setAttribute('aria-expanded', 'true');
    trigger.setAttribute('aria-controls', popoverId);
    row.append(popover);
    // Reading the evidence is the reader's choice: let go of auto-scroll so the
    // line they are on does not move, and bring the popover into view.
    setFollowing(false);
    state.programmaticUntil = Date.now() + 900;
    if (popover.scrollIntoView) popover.scrollIntoView({ block: 'nearest' });
    state.popover = { node: popover, trigger };
  }

  function paintSurface() {
    closePopover(false);
    const keepScroll = surface.scrollTop;
    state.programmaticUntil = Date.now() + 400;
    surface.replaceChildren();
    for (const key of Object.keys(lineEls)) delete lineEls[key];
    for (const key of Object.keys(wordEls)) delete wordEls[key];
    state.active = { line: -1, word: -1 };
    surface.classList.toggle('is-timed', !!model);
    const byLine = check.kind === 'done' ? Core.claimsByLine(check.claims, lines.length) : {};

    lines.forEach((textLine, lineNo) => {
      const tokens = RA.tokenize(textLine);
      if (!tokens) {
        if (/^\s*\[(?:MUSIC|SFX|TRANSITION)\]\s*$/i.test(textLine)) {
          surface.append(h('div', { class: 'readalong-break', 'aria-hidden': 'true', text: '• • •' }));
        }
        return;
      }
      const row = h('div', { class: 'readalong-line', id: `readalong-line-${lineNo}`, 'data-line': lineNo });
      const timed = model && model.byLine[lineNo] != null ? model.lines[model.byLine[lineNo]] : null;
      if (timed) {
        row.append(h('button', {
          type: 'button', class: 'readalong-seek', 'data-line': lineNo,
          'aria-label': `Play from ${RA.clockLabel(timed.start)}`, text: RA.clockLabel(timed.start),
        }));
      }
      const body = h('p', { class: 'readalong-text' });
      const words = [];
      const segments = Core.claimSegments(textLine, byLine[lineNo] || [], state.showSupported);
      for (const group of Core.groupTokens(tokens, segments)) {
        let parent = body;
        let mark = null;
        if (group.segment) {
          // A <button> is atomic inline content in Chrome: a mark that wraps
          // would drop to its own line as one block. A span that behaves as a
          // button wraps with the words around it.
          mark = h('span', { role: 'button', tabindex: '0' });
          if (group.words[0].index > 0) body.append(' ');
          body.append(mark);
          parent = mark;
        }
        group.words.forEach((entry, position) => {
          if ((mark && position > 0) || (!mark && entry.index > 0)) parent.append(' ');
          const span = h('span', { class: 'ra-w', 'data-w': entry.index, text: entry.token.text });
          if (timed && timed.words && !timed.words[entry.index].heard) span.classList.add('is-guess');
          words.push(span);
          parent.append(span);
        });
        if (!mark) continue;
        const claims = group.segment.claims;
        const presentation = Factcheck.claimPresentation(Factcheck.strongestClaim(claims) || claims[0]);
        mark.className = `factcheck-mark is-${presentation.tone}`;
        mark.setAttribute('aria-expanded', 'false');
        mark.setAttribute('aria-label', `${presentation.label}: ${mark.textContent}`);
        mark.addEventListener('click', () => openPopover(mark, claims, row));
        mark.addEventListener('keydown', (event) => {
          if (event.key !== 'Enter' && event.key !== ' ') return;
          event.preventDefault();
          openPopover(mark, claims, row);
        });
      }
      row.append(body);
      lineEls[lineNo] = row;
      wordEls[lineNo] = words;
      surface.append(row);
    });
    surface.scrollTop = keepScroll;
    updatePosition(true);
  }

  // A tap seeks, but a tap on a fact-check mark opens its evidence instead.
  surface.addEventListener('click', (event) => {
    const target = event.target;
    if (!model || !target.closest) return;
    if (target.closest('.factcheck-mark') || target.closest('.factcheck-popover')) return;
    const selection = window.getSelection && window.getSelection();
    if (selection && String(selection).length > 0) return;
    const row = target.closest('.readalong-line');
    if (!row) return;
    const lineNo = Number(row.getAttribute('data-line'));
    const word = target.closest('.ra-w');
    if (word && !target.closest('.readalong-seek')) playFrom(RA.seekSecondsForWord(model, lineNo, Number(word.getAttribute('data-w'))));
    else playFrom(RA.seekSecondsForLine(model, lineNo));
  });
  // A scroll the page did not start is the reader's. Let go of them.
  surface.addEventListener('scroll', () => {
    if (!state.following) refreshNowButton();
    if (Date.now() < state.programmaticUntil) return;
    if (state.following) setFollowing(false);
  }, { passive: true });
  for (const name of ['wheel', 'touchmove']) {
    surface.addEventListener(name, () => { if (state.following) setFollowing(false); }, { passive: true });
  }
  nowButton.addEventListener('click', () => {
    setFollowing(true);
    if (model && state.active.line !== -1) scrollTo(lineEls[model.lines[state.active.line].line], false);
  });

  // ---- wiring the audio ----------------------------------------------------
  const mediaSession = audio && 'mediaSession' in navigator ? navigator.mediaSession : null;
  const mediaActions = {
    play: () => play(),
    pause: () => audio.pause(),
    seekbackward: (details) => seekTo(audio.currentTime - ((details && details.seekOffset) || 15)),
    seekforward: (details) => seekTo(audio.currentTime + ((details && details.seekOffset) || 30)),
    seekto: (details) => { if (details && Number.isFinite(details.seekTime)) seekTo(details.seekTime); },
  };

  function paintToggle() {
    const playing = !audio.paused && !audio.ended;
    toggle.textContent = playing ? 'Pause' : 'Play';
    toggle.setAttribute('aria-label', playing ? 'Pause' : 'Play');
  }

  if (audio) {
    toggle.addEventListener('click', () => { if (audio.paused) play(); else audio.pause(); });
    back.addEventListener('click', () => seekTo(audio.currentTime - 15));
    forward.addEventListener('click', () => seekTo(audio.currentTime + 30));
    range.addEventListener('pointerdown', () => { state.scrubbing = true; });
    range.addEventListener('pointerup', () => { state.scrubbing = false; });
    range.addEventListener('change', () => { state.scrubbing = false; });
    range.addEventListener('input', () => {
      seekTo(Number(range.value));
      const total = duration();
      clock.textContent = `${Core.formatClock(Number(range.value))}${total > 0 ? ` / ${Core.formatClock(total)}` : ''}`;
      setFollowing(true);
      updatePosition(true);
    });
    rate.addEventListener('change', () => {
      audio.playbackRate = rateOrDefault(rate.value);
      remember(RATE_KEY, rate.value);
    });
    audio.addEventListener('loadedmetadata', () => { applyPendingSeek(); paintClock(); });
    audio.addEventListener('canplay', applyPendingSeek);
    audio.addEventListener('play', () => { paintToggle(); startLoop(); if (drawer) drawer.update(false); });
    audio.addEventListener('pause', () => { paintToggle(); savePosition(true); updatePosition(false); if (drawer) drawer.update(false); });
    audio.addEventListener('ended', () => { paintToggle(); savePosition(true); });
    audio.addEventListener('seeked', () => { paintClock(); updatePosition(false); });
    audio.addEventListener('timeupdate', () => { paintClock(); updatePosition(false); savePosition(false); });
    audio.addEventListener('error', () => {
      note.replaceChildren('The audio could not be played. Check your connection, or ', h('a', { href: '/', text: 'sign in again' }), '.');
      note.hidden = false;
    });
    audio.playbackRate = rateOrDefault(rate.value);

    const resume = resumeSeconds(stored(POSITION_KEY(id)), durationHint);
    if (resume > 0) state.pendingSeek = resume;
    if (mediaSession) {
      try {
        mediaSession.metadata = new MediaMetadata({ title: episode.title, artist: pillar ? pillar.name : 'NLP Reader', album: 'NLP Reader' });
        for (const [action, handler] of Object.entries(mediaActions)) mediaSession.setActionHandler(action, handler);
      } catch (error) { /* an action this browser does not know */ }
    }
  }

  // ---- the page ------------------------------------------------------------
  const noteText = readAlongNote({ hasAudio: !!src, timelineState, wordLevel: !!(model && model.wordLevel) });
  if (noteText) {
    note.textContent = noteText;
    note.hidden = false;
  }

  const supportedToggle = h('input', { type: 'checkbox', id: 'factcheck-show-supported' });
  supportedToggle.addEventListener('change', () => { state.showSupported = supportedToggle.checked; paintSurface(); });
  const issues = check.kind === 'done' ? Factcheck.issueClaims(check.claims) : [];

  function jumpToLine(lineNo) {
    const el = lineEls[lineNo];
    if (!el) return;
    for (const stale of surface.querySelectorAll('.readalong-line.is-target')) stale.classList.remove('is-target');
    el.classList.add('is-target');
    setFollowing(false);
    scrollTo(el, true);
    surface.scrollIntoView({ block: 'nearest', behavior: reducedMotion() ? 'auto' : 'smooth' });
  }

  function issueCard(claim) {
    const presentation = Factcheck.claimPresentation(claim);
    const evidence = Array.isArray(claim.evidence) ? claim.evidence : [];
    const jump = h('button', { type: 'button', class: 'btn btn-ghost btn-sm factcheck-jump', text: 'Jump to line' });
    jump.addEventListener('click', () => jumpToLine(Number(claim.line_index)));
    return h('article', { class: `factcheck-issue is-${presentation.tone}` },
      h('strong', { text: `${presentation.icon} ${presentation.label}` }),
      h('span', { class: 'factcheck-issue-line', text: `L${claim.line_index}` }),
      h('q', { text: claim.span_text || claim.claim || '' }),
      claim.rationale ? h('p', { text: claim.rationale }) : null,
      evidence.length && evidence[0].quote
        ? h('blockquote', { text: `Paper: “${evidence[0].quote}”` })
        : (claim.verdict !== 'not_checkable' ? h('p', { class: 'factcheck-evidence-missing', text: 'Evidence quote not located.' }) : null),
      jump);
  }

  const factcheckPanel = h('section', { class: 'factcheck-panel', 'aria-labelledby': 'factcheck-title' },
    h('div', { class: 'factcheck-head' },
      h('div', null,
        h('h2', { id: 'factcheck-title', class: 'section-title', text: 'Fact check' }),
        h('p', { class: 'factcheck-status', role: 'status', text: ({
          none: 'Not fact-checked yet. No claim in this episode has been graded.',
          stale: 'The script changed after it was fact-checked, so the results are not shown.',
          done: 'Completed against the stored paper text.',
        })[check.kind] }))),
    check.kind === 'done' ? h('div', { class: 'factcheck-summary' },
      h('p', { text: Factcheck.factcheckSummary(check.claims) }),
      check.check.created_at ? h('p', { class: 'factcheck-meta', text: `Checked ${formatDate(check.check.created_at)}` }) : null) : null,
    check.kind === 'done' ? h('label', { class: 'factcheck-toggle', for: 'factcheck-show-supported' }, supportedToggle, h('span', { text: 'Show supported claims' })) : null,
    check.kind === 'done' ? h('div', { class: 'reader-issues' },
      issues.length === 0
        ? h('p', { class: 'field-help', text: 'No high, medium, or low severity flags. Opinions and supported claims are counted separately above.' })
        : issues.map(issueCard)) : null);

  const source = episode.source_material;
  const sourceWarnings = source && Array.isArray(source.warnings) ? source.warnings : [];

  const saveButton = offline && src ? saveControl(offline, episode.id) : null;

  const root = h('div', { class: 'listen-page episode-page', 'data-episode-id': episode.id },
    h('nav', { class: 'crumb', 'aria-label': 'Breadcrumb' },
      h('a', { href: '#/listen', text: 'Listen' }),
      pillar ? [h('span', { 'aria-hidden': 'true', text: '/' }),
        h('span', { class: `crumb-pillar ${pillarClass(pillar.id)}` },
          h('span', { class: 'pillar-dot', 'aria-hidden': 'true' }), pillar.name)] : null),
    h('article', { class: 'card player-card reader-player' },
      h('div', { class: `player-cover ${pillarClass(episode.pillar_id)}`, 'aria-hidden': 'true' }, episodeInitial(episode.title)),
      h('div', { class: 'player-body' },
        h('h1', { class: 'player-title', text: episode.title }),
        h('p', { class: 'episode-meta', text: episodeMeta({ duration_seconds: durationHint, created_at: episode.created_at }, pillar?.name) }),
        paper ? h('p', { class: 'episode-meta' }, 'From ', h('a', { href: paperHref(paper.id), text: paper.title })) : null,
        offline && src ? saveButton : null)),
    sourceWarnings.length
      ? callout('warn', 'Written from limited material', h('ul', null, sourceWarnings.map((w) => h('li', { text: w }))))
      : null,
    h('section', { class: 'reader-script-section', 'aria-labelledby': 'script-heading' },
      h('h2', { id: 'script-heading', class: 'visually-hidden', text: 'Script' }),
      h('div', { class: 'readalong' },
        note,
        h('div', { class: 'reader-surface-wrap' }, surface, nowButton),
        audio ? h('div', { class: 'readalong-bar reader-bar' },
          audio,
          h('div', { class: 'reader-bar-scrub' }, range, clock),
          h('div', { class: 'reader-bar-buttons' }, back, toggle, forward, rate)) : null)),
    drawer ? drawer.root : null,
    episode.key_points && episode.key_points.length
      ? h('section', { class: 'reader-keypoints' },
        h('h2', { class: 'section-title', text: 'Key points' }),
        h('ul', null, episode.key_points.map((point) => h('li', { text: point }))))
      : null,
    factcheckPanel);

  paintSurface();
  paintClock();
  if (audio) paintToggle();
  refreshNowButton();

  root.dispose = () => {
    if (saveButton) saveButton.dispose();
    if (state.raf) cancelAnimationFrame(state.raf);
    state.raf = 0;
    closePopover(false);
    if (drawer) drawer.dispose();
    if (audio) {
      savePosition(true);
      audio.pause();
      audio.removeAttribute('src');
      audio.load();
    }
    if (mediaSession) {
      for (const action of Object.keys(mediaActions)) {
        try { mediaSession.setActionHandler(action, null); } catch (error) { /* ignore */ }
      }
      mediaSession.metadata = null;
    }
  };
  return root;
}
