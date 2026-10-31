/*
 * Confirm-before-spend. Loaded in <head> so it wraps window.fetch before any
 * page script runs; every POST to a route that spends money or the GPU then
 * shows one dialog with the day's remaining budget, and only an explicit
 * "Continue" sends the request, stamped with X-Nlp-Confirm: 1 (which the
 * server demands of anything arriving through the tunnel).
 *
 * The server enforces the daily cap whatever this script does. This is the
 * part a person sees: what is about to cost money, and how much is left.
 * Everything is built with createElement + textContent.
 */
(function (global) {
  'use strict';

  var SPEND_URL = '/api/spend';
  var CONFIRM_HEADER = 'X-Nlp-Confirm';

  // Keep in step with webui/services/spend_guard.py's users.
  var ACTIONS = [
    { re: /^\/pipeline\/run$/, title: 'Run the pipeline', what: 'Finds, reads and summarises new papers with Claude.' },
    { re: /^\/api\/pillars\/[^/]+\/select$/, title: 'Process selected papers', what: 'Reads each paper and writes notes, a lesson and a quiz with Claude.' },
    { re: /^\/api\/pillars\/[^/]+\/upload\/(url|pdf)$/, title: 'Add this paper', what: 'Processes the paper with Claude (summary, lesson, quiz).' },
    { re: /^\/api\/podcast\/generate$/, title: 'Generate a podcast script', what: 'Writing the script makes paid model calls. If “Also make audio after saving” is checked, it will also use IndexTTS on the local GPU after the script saves.' },
    { re: /^\/api\/podcast\/tts\/generate$/, title: 'Generate audio', what: 'Runs the speech model on the computer’s GPU for several minutes.' },
    { re: /^\/api\/podcast\/[^/]+\/factcheck$/, title: 'Check the facts', what: 'Checks every claim against the paper with Claude.' },
    { re: /^\/api\/playgrounds\/[^/]+\/explain$/, title: 'Generate the study guide', what: 'Explains every code cell with Claude Sonnet, about a dime per dozen cells.' },
    { re: /^\/api\/follow-along$/, title: 'Follow along with code', what: 'One quick Claude Haiku call matches the episode’s lines to the notebook’s cells, about a cent.' },
    { re: /^\/api\/playgrounds\/[^/]+\/explain-difference$/, title: 'Explain the difference', what: 'One short Claude Haiku call, a fraction of a cent.' }
  ];

  // Buttons that start those actions; each gets the remaining budget beside it.
  var BUTTON_SELECTOR = '#generate-btn, #generate-audio-btn, #factcheck-btn, #process-btn, #run-btn, #submit-file-upload, #submit-url-upload, #playground-explain-btn, #follow-start';

  function pathOf(input) {
    var raw = typeof input === 'string' ? input : (input && input.url) || '';
    try {
      return new URL(raw, global.location ? global.location.href : 'http://localhost').pathname;
    } catch (e) {
      return '';
    }
  }

  function methodOf(input, init) {
    var m = (init && init.method) || (input && typeof input !== 'string' && input.method) || 'GET';
    return String(m).toUpperCase();
  }

  function matchAction(method, path) {
    if (method !== 'POST') return null;
    for (var i = 0; i < ACTIONS.length; i++) {
      if (ACTIONS[i].re.test(path)) return ACTIONS[i];
    }
    return null;
  }

  function money(value) {
    var n = Number(value);
    if (!isFinite(n)) return '$?';
    return '$' + n.toFixed(2);
  }

  function budgetText(budget) {
    if (!budget) return 'Budget unavailable';
    if (budget.exhausted) return 'Daily cap reached (' + money(budget.spent_usd) + ' of ' + money(budget.cap_usd) + ')';
    return money(budget.remaining_usd) + ' left of ' + money(budget.cap_usd) + ' today';
  }

  function withConfirm(input, init) {
    var next = {};
    var key;
    for (key in (init || {})) next[key] = init[key];
    var base = (init && init.headers) || (input && typeof input !== 'string' && input.headers) || undefined;
    var headers = new Headers(base);
    headers.set(CONFIRM_HEADER, '1');
    next.headers = headers;
    return next;
  }

  function cancelled() {
    var message = 'Cancelled. Nothing was started.';
    return new Response(JSON.stringify({ detail: message, error: message, success: false }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function askToContinue(action, budget) {
    return new Promise(function (resolve) {
      var dialog = el('dialog', 'dialog');
      dialog.setAttribute('aria-labelledby', 'spend-guard-title');
      var article = el('article');
      var header = el('header');
      var title = el('h2', null, action.title);
      title.id = 'spend-guard-title';
      header.appendChild(title);
      var body = el('div', 'dialog-body');
      body.appendChild(el('p', null, action.what));
      var stop = !!(budget && budget.exhausted);
      var line = el('p', 'field-help', budgetText(budget));
      line.setAttribute('role', 'status');
      body.appendChild(line);
      if (stop) body.appendChild(el('p', null, 'Nothing more can start until the cap resets or is raised.'));

      var actions = el('div', 'dialog-actions');
      var cancel = el('button', 'btn btn-secondary', stop ? 'OK' : 'Cancel');
      cancel.type = 'button';
      actions.appendChild(cancel);
      var go = null;
      if (!stop) {
        go = el('button', 'btn btn-primary', 'Continue');
        go.type = 'button';
        actions.appendChild(go);
      }
      body.appendChild(actions);
      article.appendChild(header);
      article.appendChild(body);
      dialog.appendChild(article);

      var answer = false;
      function finish() {
        if (dialog.open) dialog.close();
      }
      cancel.addEventListener('click', finish);
      if (go) go.addEventListener('click', function () { answer = true; finish(); });
      dialog.addEventListener('close', function () {
        dialog.remove();
        resolve(answer);
      });
      document.body.appendChild(dialog);
      dialog.showModal();
      (go || cancel).focus();
    });
  }

  var realFetch = typeof global.fetch === 'function' ? global.fetch.bind(global) : null;
  var cached = null;

  function loadBudget() {
    return realFetch(SPEND_URL, { headers: { Accept: 'application/json' }, cache: 'no-store' })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (body) { cached = body; return body; })
      .catch(function () { return null; });
  }

  function guardedFetch(input, init) {
    var action = matchAction(methodOf(input, init), pathOf(input));
    if (!action) return realFetch(input, init);
    return loadBudget().then(function (budget) {
      paintBudget(budget);
      return askToContinue(action, budget).then(function (ok) {
        if (!ok) return cancelled();
        return realFetch(input, withConfirm(input, init)).then(function (res) {
          loadBudget().then(paintBudget);
          return res;
        });
      });
    });
  }

  function paintBudget(budget) {
    if (typeof document === 'undefined') return;
    var targets = document.querySelectorAll('[data-spend-budget]');
    for (var i = 0; i < targets.length; i++) {
      targets[i].textContent = budgetText(budget);
      targets[i].setAttribute('data-exhausted', budget && budget.exhausted ? 'true' : 'false');
    }
  }

  function placeBudgetChips() {
    var buttons = document.querySelectorAll(BUTTON_SELECTOR);
    if (!buttons.length) return;
    for (var i = 0; i < buttons.length; i++) {
      var chip = el('span', 'spend-budget');
      chip.setAttribute('data-spend-budget', '');
      buttons[i].insertAdjacentElement('afterend', chip);
    }
    if (cached) paintBudget(cached);
    else loadBudget().then(paintBudget);
  }

  if (realFetch && typeof document !== 'undefined') global.fetch = guardedFetch;

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', placeBudgetChips);
    else placeBudgetChips();
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      matchAction: matchAction,
      pathOf: pathOf,
      methodOf: methodOf,
      budgetText: budgetText,
      money: money
    };
  }
})(typeof window !== 'undefined' ? window : globalThis);
