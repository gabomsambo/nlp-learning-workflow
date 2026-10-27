// Just enough of a DOM to build the reader's pages in node --test: elements,
// text, fragments, attributes, classes, listeners and simple selectors. No
// layout, no HTML parser (nothing in the reader parses HTML).

class FakeNode {
  constructor() {
    this.childNodes = [];
    this.parentNode = null;
  }

  get firstChild() { return this.childNodes[0] || null; }

  appendChild(child) {
    if (child instanceof FakeFragment) {
      for (const node of [...child.childNodes]) this.appendChild(node);
      child.childNodes = [];
      return child;
    }
    if (child.parentNode) child.parentNode.childNodes = child.parentNode.childNodes.filter((n) => n !== child);
    child.parentNode = this;
    this.childNodes.push(child);
    return child;
  }

  append(...children) {
    for (const child of children) this.appendChild(typeof child === 'string' ? new FakeText(child) : child);
  }

  replaceChildren(...children) {
    for (const node of this.childNodes) node.parentNode = null;
    this.childNodes = [];
    this.append(...children);
  }

  remove() {
    if (this.parentNode) this.parentNode.childNodes = this.parentNode.childNodes.filter((n) => n !== this);
    this.parentNode = null;
  }

  get textContent() { return this.childNodes.map((n) => n.textContent).join(''); }

  set textContent(value) {
    this.childNodes = [];
    if (value !== null && value !== undefined && value !== '') this.appendChild(new FakeText(String(value)));
  }
}

class FakeText extends FakeNode {
  constructor(data) {
    super();
    this.data = data;
  }

  get textContent() { return this.data; }
}

class FakeFragment extends FakeNode {}

const REFLECTED = { className: 'class', id: 'id', title: 'title', href: 'href' };
const BOOLEAN = ['hidden', 'open', 'disabled'];

class FakeElement extends FakeNode {
  constructor(tag) {
    super();
    this.tagName = tag.toUpperCase();
    this.attributes = new Map();
    this.listeners = {};
    const element = this;
    this.classList = {
      add: (...names) => { for (const n of names) if (!this.classList.contains(n)) element.className = `${element.className} ${n}`.trim(); },
      remove: (...names) => { element.className = element.className.split(/\s+/).filter((c) => c && !names.includes(c)).join(' '); },
      contains: (name) => element.className.split(/\s+/).includes(name),
      toggle: (name, force) => {
        const on = force === undefined ? !element.classList.contains(name) : !!force;
        if (on) element.classList.add(name);
        else element.classList.remove(name);
        return on;
      },
    };
  }

  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.has(name) ? this.attributes.get(name) : null; }
  hasAttribute(name) { return this.attributes.has(name); }
  removeAttribute(name) { this.attributes.delete(name); }

  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  dispatch(type, event = {}) { for (const fn of this.listeners[type] || []) fn({ target: this, preventDefault() {}, ...event }); }
  click() { this.dispatch('click'); }
  scrollIntoView() { this.scrolledIntoView = true; }

  get children() { return this.childNodes.filter((n) => n instanceof FakeElement); }

  matches(selector) {
    if (selector.startsWith('#')) return this.id === selector.slice(1);
    if (selector.startsWith('.')) return selector.slice(1).split('.').every((c) => this.classList.contains(c));
    return this.tagName === selector.toUpperCase();
  }

  querySelectorAll(selector) {
    const found = [];
    const walk = (node) => {
      for (const child of node.children) {
        if (child.matches(selector)) found.push(child);
        walk(child);
      }
    };
    walk(this);
    return found;
  }

  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

for (const [prop, attr] of Object.entries(REFLECTED)) {
  Object.defineProperty(FakeElement.prototype, prop, {
    get() { return this.getAttribute(attr) || ''; },
    set(value) { this.setAttribute(attr, value); },
  });
}
for (const prop of BOOLEAN) {
  Object.defineProperty(FakeElement.prototype, prop, {
    get() { return this.hasAttribute(prop); },
    set(value) { if (value) this.setAttribute(prop, ''); else this.removeAttribute(prop); },
  });
}

function memoryStorage() {
  const data = new Map();
  return {
    getItem: (key) => (data.has(key) ? data.get(key) : null),
    setItem: (key, value) => { data.set(key, String(value)); },
    removeItem: (key) => { data.delete(key); },
    clear: () => data.clear(),
  };
}

/** Put a fresh document, Node, window and localStorage on globalThis. */
export function installFakeDom() {
  const document = {
    readyState: 'complete',
    createElement: (tag) => new FakeElement(tag),
    createElementNS: (ns, tag) => new FakeElement(tag),
    createTextNode: (text) => new FakeText(String(text)),
    createDocumentFragment: () => new FakeFragment(),
    getElementById: () => null,
    addEventListener() {},
  };
  globalThis.document = document;
  globalThis.Node = FakeNode;
  globalThis.window = globalThis;
  globalThis.localStorage = memoryStorage();
  return document;
}
