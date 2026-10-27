/*
 * The Playground tab on a paper: a notebook's study steps as cards, read-only.
 *
 * Each code cell is a card with its one-line explanation (hidden for a
 * self-quiz, revealed per card), the code with tap-a-line notes, the saved
 * output and each "try this" as a before/after diff. Nothing runs here.
 *
 * The cards are drawn with the app's own playground.js (loaded as a classic
 * script before this module, index.html, after the playground-core.js it
 * takes lineDiff from): renderCode for the lines and notes,
 * renderOutput and buildTree for saved output, so a sanitized table is built
 * here exactly as on the computer. Everything else is createElement +
 * textContent.
 */
import { h } from './dom.js';
import { droppedNote, droppedSummary, outputForRender } from './lib.js';

const PG = window.Playground;

function storedHide() {
  try { return localStorage.getItem(PG.HIDE_KEY) === 'true'; } catch (error) { return false; }
}

function rememberHide(on) {
  try { localStorage.setItem(PG.HIDE_KEY, on ? 'true' : 'false'); } catch (error) { /* this view only */ }
}

export function renderSavedOutput(out) {
  const shaped = outputForRender(out);
  return shaped ? PG.renderOutput(shaped) : h('p', { class: 'pg-out-note', text: droppedNote(out) });
}

function renderTry(source, attempt) {
  const diff = h('div', { class: 'pg-diff', 'aria-label': 'Original and changed code, line by line' },
    PG.lineDiff(source, attempt.code).map((row) =>
      h('div', {
        class: `pg-diff-line is-${row.op}`,
        'aria-label': row.op === 'same' ? null : `${row.op === 'add' ? 'added' : 'removed'}: ${row.text}`,
      },
      h('span', { class: 'pg-diff-mark', 'aria-hidden': 'true', text: row.op === 'add' ? '+' : row.op === 'del' ? '−' : ' ' }),
      h('code', { text: row.text || ' ' }))));
  return h('details', { class: 'pg-try' },
    h('summary', { text: `Try: ${attempt.label}` }),
    attempt.expect ? h('p', { class: 'pg-expect', text: attempt.expect }) : null,
    diff);
}

function renderMarkdown(cell) {
  const parts = PG.markdownParts(cell.source);
  if (!parts) return null;
  if (!parts.rest) return h('div', { class: 'pg-md', text: parts.head });
  return h('details', { class: 'pg-md' }, h('summary', { text: parts.head }), h('div', { class: 'pg-md-body', text: parts.rest }));
}

function renderCard(cell, guide, state) {
  const count = cell.execution_count != null ? `In [${cell.execution_count}]` : `Cell ${cell.index}`;
  const head = h('header', { class: 'pg-card-head' }, h('span', { class: 'pg-count', text: count }));
  const card = h('article', { class: 'card pg-card', id: `pg-cell-${cell.index}`, 'data-cell': cell.index }, head);

  if (guide) {
    const explain = h('p', { class: 'pg-explain', text: guide.explanation });
    const reveal = h('button', { type: 'button', class: 'btn btn-ghost btn-sm pg-reveal', text: 'Show explanation' });
    reveal.addEventListener('click', () => {
      explain.hidden = false;
      reveal.hidden = true;
    });
    explain.hidden = state.hide;
    reveal.hidden = !state.hide;
    state.explanations.push({ explain, reveal });
    head.append(reveal);
    card.append(explain);
  } else if (state.hasGuide) {
    card.append(h('p', { class: 'pg-out-note', text: 'Not explained yet.' }));
  }

  const notes = guide ? guide.line_notes || [] : [];
  card.append(PG.renderCode(cell.source, notes));
  if (notes.length) card.append(h('p', { class: 'pg-hint', text: 'Tap a highlighted line for its note.' }));

  const outputs = cell.outputs || [];
  if (outputs.length) {
    card.append(h('div', { class: 'pg-outputs' },
      h('p', { class: 'pg-label', text: 'Saved output' }),
      outputs.map(renderSavedOutput)));
  }

  if (guide) {
    for (const attempt of guide.tries || []) card.append(renderTry(cell.source, attempt));
    if (guide.quiz) card.append(h('p', { class: 'pg-quiz' }, h('strong', { text: 'Can you answer: ' }), guide.quiz));
  }
  return card;
}

/**
 * The tab's content for one published playground. `cell` is the cell a link
 * asked for; call `root.focusCell()` once the page is on screen.
 */
export function playgroundPanel(pg, { cell = null } = {}) {
  const steps = pg.steps || [];
  const state = { hide: storedHide(), hasGuide: steps.length > 0, explanations: [] };
  const guide = PG.explanationsByCell(steps);

  const hideButton = h('button', { type: 'button', class: 'btn btn-secondary btn-sm', 'aria-pressed': 'false' });
  function paintHide() {
    hideButton.textContent = state.hide ? 'Show explanations' : 'Hide explanations';
    hideButton.setAttribute('aria-pressed', state.hide ? 'true' : 'false');
  }
  hideButton.addEventListener('click', () => {
    state.hide = !state.hide;
    rememberHide(state.hide);
    for (const { explain, reveal } of state.explanations) {
      explain.hidden = state.hide;
      reveal.hidden = !state.hide;
    }
    paintHide();
  });
  paintHide();

  const sections = [];
  let number = 0;
  for (const group of PG.groupCells(pg.cells || [], steps)) {
    const section = h('section', { class: 'pg-step' });
    if (group.step) {
      number += 1;
      section.id = `pg-step-${number}`;
      section.append(
        h('p', { class: 'pg-step-kicker', text: `Step ${number}` }),
        h('h3', { class: 'pg-step-title', text: group.step.title }),
        group.step.goal ? h('p', { class: 'pg-goal', text: group.step.goal }) : null);
    } else if (steps.length) {
      section.append(
        h('h3', { class: 'pg-step-title', text: 'Rest of the notebook' }),
        h('p', { class: 'pg-goal', text: 'These cells are not in the study guide.' }));
    }
    for (const c of group.cells) {
      const node = c.type === 'code' ? renderCard(c, guide[c.index], state) : renderMarkdown(c);
      if (node) section.append(node);
    }
    sections.push(section);
  }

  // The steps are on this page, and the route lives in the hash: buttons, not #anchors.
  const nav = steps.length
    ? h('ol', { class: 'pg-steps-nav', 'aria-label': 'Study steps' }, steps.map((step, i) => {
      const go = h('button', { type: 'button', class: 'reader-step-link', text: step.title });
      go.addEventListener('click', () => {
        const target = sections.find((s) => s.id === `pg-step-${i + 1}`);
        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
      return h('li', null, go);
    }))
    : null;

  const dropped = droppedSummary(pg.dropped);
  const root = h('div', { class: 'pg-root reader-playground' },
    h('div', { class: 'pg-head' },
      h('h3', { class: 'pg-title', text: pg.title || 'Notebook' }),
      h('p', { class: 'reader-pg-note' }, h('strong', { text: 'Run code on your computer.' }),
        ' Here you can read each step, its code and the output it saved.'),
      pg.status === 'partial' ? h('p', { class: 'paper-muted', text: 'The study guide covers only part of the notebook.' }) : null,
      dropped ? h('p', { class: 'pg-out-note', text: dropped }) : null,
      state.hasGuide ? h('div', { class: 'pg-actions' }, hideButton) : null),
    nav,
    sections);

  root.focusCell = () => {
    if (cell == null) return;
    const card = root.querySelector(`#pg-cell-${cell}`);
    if (!card) return;
    card.classList.add('is-target');
    card.scrollIntoView({ block: 'start' });
  };
  return root;
}
