/* Tiny element builder. Everything is createElement + textContent: nothing here parses HTML. */

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (name === 'class') el.className = value;
    else if (name === 'text') el.textContent = value;
    else el.setAttribute(name, value === true ? '' : String(value));
  }
  for (const child of children.flat()) {
    if (child === undefined || child === null || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

const SVG_NS = 'http://www.w3.org/2000/svg';
const ICON_PATHS = {
  today: 'M4 10.5 12 4l8 6.5V20a1 1 0 0 1-1 1h-5v-6H10v6H5a1 1 0 0 1-1-1z',
  library: 'M5 4h5a2 2 0 0 1 2 2v13H7a2 2 0 0 0-2 2V4zM12 6h5a2 2 0 0 1 2 2v11h-7V6z',
  lessons: 'M6 3h8l4 4v14H6zM14 3v4h4',
};

export function icon(name) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'icon');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '1.75');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  const path = document.createElementNS(SVG_NS, 'path');
  path.setAttribute('d', ICON_PATHS[name] || '');
  svg.append(path);
  return svg;
}

export const chip = (label, variant = 'neutral') => h('span', { class: `chip chip-${variant}`, text: label });

export const emptyState = (title, message, action) =>
  h('div', { class: 'empty' },
    h('p', { class: 'empty-title', text: title }),
    h('p', { class: 'empty-message', text: message }),
    action || null);

export const callout = (kind, title, body) =>
  h('aside', { class: `callout callout-${kind}` },
    title ? h('p', { class: 'callout-title', text: title }) : null,
    h('div', { class: 'callout-body' }, body));

export const bullets = (items) => h('ul', null, (items || []).map((item) => h('li', { text: item })));
