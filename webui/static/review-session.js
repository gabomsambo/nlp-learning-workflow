/*
 * One-card review session for /quiz/.
 *
 * The server sends the due cards as JSON. This file shows one of them, reveals
 * the answer, and posts the same /api/quiz/review body the old page posted.
 * Difficulty words come from PaperModal.difficultyLabel — that function is the
 * only mapping, shared with the paper modal. Nothing here builds markup from
 * strings: questions and answers are textContent.
 *
 * Shortcut help overlay
 * ---------------------
 * The shell owns window.NlpShortcuts.register({id, group, label, keys, match,
 * run}). Review registers Space and 1–4 there, and the shell dispatches them
 * (it already ignores text fields). When that registry is absent, the same
 * list is left on window.NLPReviewShortcuts and this file handles the keys.
 * The buttons and the #review-shortcuts dialog work either way.
 */
(function (global) {
  'use strict';

  var REVIEW_SHORTCUTS = [
    { id: 'review-reveal', keys: 'Space', label: 'Reveal the answer' },
    { id: 'review-again', keys: '1', label: 'Rate Again' },
    { id: 'review-hard', keys: '2', label: 'Rate Hard' },
    { id: 'review-good', keys: '3', label: 'Rate Good' },
    { id: 'review-easy', keys: '4', label: 'Rate Easy' }
  ];

  var TEXT_INPUT = {
    text: 1, search: 1, email: 1, url: 1, password: 1, tel: 1, number: 1
  };

  function isTextField(el) {
    if (!el || typeof el !== 'object') return false;
    if (el.isContentEditable) return true;
    var tag = String(el.tagName || '').toUpperCase();
    if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
    if (tag !== 'INPUT') return false;
    var type = String(el.type || 'text').toLowerCase();
    return !type || TEXT_INPUT[type] === 1;
  }

  function shortcutCommand(event) {
    if (!event || event.metaKey || event.ctrlKey || event.altKey) return null;
    if (isTextField(event.target)) return null;
    if (event.key === ' ' || event.key === 'Spacebar') return { type: 'reveal' };
    if (event.key === '1' || event.key === '2' || event.key === '3' || event.key === '4') {
      return { type: 'rate', rating: Number(event.key) };
    }
    return null;
  }

  function dialogOpen() {
    var doc = global.document;
    if (!doc || typeof doc.querySelector !== 'function') return false;
    return !!doc.querySelector('dialog[open]');
  }

  function shellEntries(handlers) {
    var reveal = handlers && handlers.reveal ? handlers.reveal : function () {};
    var rate = handlers && handlers.rate ? handlers.rate : function () {};
    function keyIs(name) {
      return function (event) {
        return !!event && event.key === name && !dialogOpen();
      };
    }
    return [
      {
        id: 'review-reveal',
        group: 'Review',
        label: 'Reveal the answer',
        keys: 'Space',
        match: function (event) {
          return !!event && (event.key === ' ' || event.key === 'Spacebar') && !dialogOpen();
        },
        run: reveal
      },
      { id: 'review-again', group: 'Review', label: 'Rate Again', keys: '1', match: keyIs('1'), run: function () { rate(1); } },
      { id: 'review-hard', group: 'Review', label: 'Rate Hard', keys: '2', match: keyIs('2'), run: function () { rate(2); } },
      { id: 'review-good', group: 'Review', label: 'Rate Good', keys: '3', match: keyIs('3'), run: function () { rate(3); } },
      { id: 'review-easy', group: 'Review', label: 'Rate Easy', keys: '4', match: keyIs('4'), run: function () { rate(4); } }
    ];
  }

  function registerShortcuts(registry, handlers) {
    var target = registry || global.NlpShortcuts || global.NLPShortcuts;
    if (target && typeof target.register === 'function') {
      shellEntries(handlers).forEach(function (entry) { target.register(entry); });
      return 'registered';
    }
    global.NLPReviewShortcuts = REVIEW_SHORTCUTS;
    return 'hook';
  }

  function progressLabel(index, total) {
    return index + ' of ' + total;
  }

  function labelDifficulty(level) {
    var modal = global.PaperModal;
    if (!modal || typeof modal.difficultyLabel !== 'function') return null;
    return modal.difficultyLabel(level);
  }

  function summariseSession(results) {
    var counts = { 1: 0, 2: 0, 3: 0, 4: 0 };
    var next = null;
    var rows = results || [];
    rows.forEach(function (row) {
      var rating = Number(row.rating);
      if (counts[rating] !== undefined) counts[rating] += 1;
      if (!row.nextReview) return;
      var time = Date.parse(row.nextReview);
      if (isNaN(time)) return;
      if (!next || time < Date.parse(next)) next = row.nextReview;
    });
    return { reviewed: rows.length, counts: counts, nextDue: next };
  }

  function formatNextReview(iso) {
    if (!iso) return 'Soon';
    var date = new Date(iso);
    if (isNaN(date.getTime())) return 'Soon';
    return date.toLocaleString();
  }

  function ratingLabel(card, rating) {
    var intervals = card.intervals || [];
    for (var i = 0; i < intervals.length; i++) {
      if (Number(intervals[i].rating) === Number(rating)) return intervals[i].label;
    }
    return String(rating);
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null && text !== '') node.textContent = String(text);
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function mount(root, data) {
    var cards = (data && data.cards) || [];
    var index = 0;
    var revealed = false;
    var finished = false;
    var busy = false;
    var shownAt = Date.now();
    var results = [];

    var progressEl = root.querySelector('#review-progress');
    var meter = root.querySelector('#review-meter');
    var cardEl = root.querySelector('#review-card');
    var statusEl = root.querySelector('#review-status');
    var errorEl = root.querySelector('#review-error');
    if (!cardEl) return;

    function setError(message) {
      if (!errorEl) return;
      if (!message) {
        errorEl.textContent = '';
        errorEl.hidden = true;
        return;
      }
      errorEl.hidden = false;
      errorEl.textContent = message;
    }

    function setStatus(message) {
      if (statusEl) statusEl.textContent = message || '';
    }

    function reveal() {
      if (revealed || finished) return;
      revealed = true;
      var revealBtn = cardEl.querySelector('[data-reveal]');
      var answer = cardEl.querySelector('.review-answer');
      var ratings = cardEl.querySelector('.review-ratings');
      if (revealBtn) revealBtn.hidden = true;
      if (answer) {
        answer.hidden = false;
        answer.tabIndex = -1;
        if (answer.focus) answer.focus();
      }
      if (ratings) ratings.hidden = false;
    }

    function renderEnd() {
      finished = true;
      clear(cardEl);
      var summary = summariseSession(results);
      cardEl.appendChild(el('h2', 'review-end-title', 'Session complete'));
      var noun = summary.reviewed === 1 ? ' card.' : ' cards.';
      cardEl.appendChild(el('p', 'review-end-count', 'You reviewed ' + summary.reviewed + noun));
      cardEl.appendChild(el(
        'p',
        'review-tally',
        'Again ' + summary.counts[1]
          + ' · Hard ' + summary.counts[2]
          + ' · Good ' + summary.counts[3]
          + ' · Easy ' + summary.counts[4]
      ));
      if (summary.nextDue) {
        cardEl.appendChild(el('p', 'review-next', 'Next review: ' + formatNextReview(summary.nextDue)));
      }
      var list = el('ol', 'review-results');
      results.forEach(function (row) {
        var item = el('li', 'review-result');
        item.appendChild(el('span', 'review-result-rating', row.ratingLabel));
        item.appendChild(el('span', 'review-result-question', row.question));
        item.appendChild(el('span', 'review-result-when', 'Next review: ' + formatNextReview(row.nextReview)));
        list.appendChild(item);
      });
      cardEl.appendChild(list);
      var actions = el('div', 'review-end-actions');
      var again = el('a', 'btn btn-primary', 'Review again');
      again.setAttribute('href', global.location.pathname + global.location.search);
      var today = el('a', 'btn btn-secondary', 'Back to Today');
      today.setAttribute('href', '/');
      actions.appendChild(again);
      actions.appendChild(today);
      cardEl.appendChild(actions);
      if (progressEl) progressEl.textContent = progressLabel(cards.length, cards.length);
      if (meter) {
        meter.value = cards.length;
        meter.setAttribute('aria-valuenow', String(cards.length));
        meter.setAttribute('aria-valuetext', progressLabel(cards.length, cards.length));
      }
      setStatus('Session complete. You reviewed ' + summary.reviewed + noun);
    }

    function renderCard() {
      var card = cards[index];
      if (!card) {
        renderEnd();
        return;
      }
      finished = false;
      revealed = false;
      busy = false;
      shownAt = Date.now();
      setError('');
      var position = index + 1;
      if (progressEl) progressEl.textContent = progressLabel(position, cards.length);
      if (meter) {
        meter.max = cards.length;
        meter.value = position;
        meter.setAttribute('aria-valuemin', '1');
        meter.setAttribute('aria-valuemax', String(cards.length));
        meter.setAttribute('aria-valuenow', String(position));
        meter.setAttribute('aria-valuetext', progressLabel(position, cards.length));
      }
      clear(cardEl);

      var meta = el('div', 'review-meta');
      if (card.pillarName) {
        var chipClass = 'chip chip-pillar';
        if (card.pillarId) chipClass += ' pillar-' + card.pillarId;
        var chip = el('span', chipClass, card.pillarName);
        if (card.pillarId) chip.setAttribute('title', card.pillarId);
        meta.appendChild(chip);
      }
      var diff = labelDifficulty(card.difficulty);
      if (diff) meta.appendChild(el('span', 'chip', diff));
      if (card.paperId) meta.appendChild(el('span', 'review-paper', card.paperId));
      cardEl.appendChild(meta);

      var question = el('h2', 'review-question', card.question);
      question.id = 'review-question';
      cardEl.setAttribute('aria-labelledby', 'review-question');
      cardEl.appendChild(question);

      var revealBtn = el('button', 'btn btn-primary', 'Show answer');
      revealBtn.type = 'button';
      revealBtn.setAttribute('data-reveal', 'true');
      cardEl.appendChild(revealBtn);

      var answer = el('div', 'review-answer');
      answer.hidden = true;
      answer.appendChild(el('p', 'review-answer-label', 'Answer'));
      answer.appendChild(el('p', 'review-answer-text', card.answer));
      cardEl.appendChild(answer);

      var ratings = el('div', 'review-ratings');
      ratings.hidden = true;
      (card.intervals || []).forEach(function (item) {
        var button = el('button', 'review-rate review-rate-' + item.rating);
        button.type = 'button';
        button.setAttribute('data-rating', String(item.rating));
        button.appendChild(el('span', 'review-rate-key', String(item.rating)));
        button.appendChild(el('span', 'review-rate-label', item.label));
        if (item.interval) button.appendChild(el('span', 'review-rate-interval', item.interval));
        if (item.hint) button.appendChild(el('span', 'review-rate-hint', item.hint));
        var name = item.label + (item.interval ? ', next review ' + item.interval : '');
        button.setAttribute('aria-label', name);
        ratings.appendChild(button);
      });
      cardEl.appendChild(ratings);
    }

    function rate(rating) {
      if (!revealed || busy || finished) return;
      var card = cards[index];
      if (!card) return;
      busy = true;
      var buttons = cardEl.querySelectorAll('[data-rating]');
      for (var i = 0; i < buttons.length; i++) buttons[i].disabled = true;
      var elapsed = Math.max(0, Date.now() - shownAt);
      fetch('/api/quiz/review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify({
          card_id: card.id,
          rating: rating,
          response_time_ms: elapsed
        })
      }).then(function (response) {
        return response.json().then(function (body) {
          return { ok: response.ok, body: body || {} };
        }, function () {
          return { ok: false, body: {} };
        });
      }).then(function (result) {
        var body = result.body || {};
        if (!result.ok || !body.success) {
          busy = false;
          for (var i = 0; i < buttons.length; i++) buttons[i].disabled = false;
          setError(body.message || 'Could not save this rating. Try again.');
          return;
        }
        var label = ratingLabel(card, rating);
        var when = formatNextReview(body.next_review_date);
        results.push({
          question: card.question,
          rating: rating,
          ratingLabel: label,
          nextReview: body.next_review_date || null
        });
        setStatus('Rated ' + label + '. Next review: ' + when);
        index += 1;
        if (index >= cards.length) renderEnd();
        else renderCard();
      }).catch(function () {
        busy = false;
        for (var i = 0; i < buttons.length; i++) buttons[i].disabled = false;
        setError('Could not save this rating. Try again.');
      });
    }

    cardEl.addEventListener('click', function (event) {
      var target = event.target && event.target.closest ? event.target.closest('button') : null;
      if (!target || !cardEl.contains(target)) return;
      if (target.hasAttribute('data-reveal')) {
        reveal();
        return;
      }
      var rating = target.getAttribute('data-rating');
      if (rating) rate(Number(rating));
    });

    var usingShell = !!(global.NlpShortcuts && typeof global.NlpShortcuts.register === 'function');
    registerShortcuts(null, {
      reveal: function () { reveal(); },
      rate: function (rating) { rate(rating); }
    });
    // The shell dispatches registered shortcuts. A second listener would rate
    // the same card twice.
    if (!usingShell) {
      document.addEventListener('keydown', function (event) {
        if (!root.isConnected || finished) return;
        var command = shortcutCommand(event);
        if (!command) return;
        event.preventDefault();
        if (command.type === 'reveal') reveal();
        else if (command.type === 'rate') rate(command.rating);
      });
    }

    renderCard();
  }

  function wireShortcutDialog() {
    var open = document.getElementById('review-shortcuts-open');
    var dialog = document.getElementById('review-shortcuts');
    if (!open || !dialog) return;
    open.addEventListener('click', function () {
      if (typeof dialog.showModal === 'function') dialog.showModal();
    });
    dialog.querySelectorAll('[data-dialog-close]').forEach(function (button) {
      button.addEventListener('click', function () { dialog.close(); });
    });
    dialog.addEventListener('click', function (event) {
      if (event.target === dialog) dialog.close();
    });
  }

  function boot() {
    registerShortcuts();
    wireShortcutDialog();
    var node = document.getElementById('review-session-data');
    var root = document.getElementById('review-session');
    if (!node || !root) return;
    var data;
    try {
      data = JSON.parse(node.textContent || '');
    } catch (e) {
      return;
    }
    if (!data || !data.cards || !data.cards.length) return;
    mount(root, data);
  }

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
    else boot();
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      isTextField: isTextField,
      shortcutCommand: shortcutCommand,
      registerShortcuts: registerShortcuts,
      progressLabel: progressLabel,
      labelDifficulty: labelDifficulty,
      summariseSession: summariseSession,
      formatNextReview: formatNextReview,
      REVIEW_SHORTCUTS: REVIEW_SHORTCUTS
    };
  }
})(typeof window !== 'undefined' ? window : globalThis);
