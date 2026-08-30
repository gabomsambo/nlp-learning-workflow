/*
 * The paper-detail modal, shared by /papers and /pillars/<id>.
 *
 * One modal, one endpoint (`GET /papers/details/{paper_id}`), two pages. It used to be
 * ~200 lines of inline <script> in papers.html; the pillar page's Recent Activity items
 * needed exactly the same surface, and a second copy would have drifted within a month.
 *
 * Deliberate choices, each with a failure mode behind it:
 *
 *  - Every node is built with createElement + textContent, and every attribute with
 *    setAttribute. The version this replaces assembled the whole modal out of template
 *    literals into innerHTML — paper titles, abstracts, note findings, takeaways and quiz
 *    text all interpolated raw. That is stored XSS, and it is the same defect PR #16 fixed
 *    in discovery.html and pillar_detail.html. escapeHtml() is NOT the fix: it escapes
 *    `&<>` and leaves quotes, so `title="${escapeHtml(x)}"` still injects. There is no
 *    escaping helper in this file on purpose — nothing here concatenates markup.
 *    https://cheatsheetseries.owasp.org/cheatsheets/DOM_based_XSS_Prevention_Cheat_Sheet.html
 *  - The PDF href is scheme-checked before it reaches setAttribute. url_pdf is
 *    database-supplied and `javascript:` in an href executes on click no matter how the
 *    node was created; textContent buys nothing there.
 *  - Quiz answers get ONE show/hide-all control, not a per-card reveal. These cards are
 *    read as a summary of the paper far more often than they are self-tested, so per-card
 *    reveal would be five clicks on the common path. Default is visible for the same
 *    reason, and the choice persists in localStorage so it does not reset on every open.
 *  - Answers are hidden with the `hidden` attribute, not a CSS class. `display: none` via
 *    a class hides them visually while leaving them in the accessibility tree, so a screen
 *    reader would still read out the answer the user asked to hide.
 */
(function (global) {
  'use strict';

  var DETAILS_URL = '/papers/details/';
  var REFRESH_URL = '/api/papers/';
  var ANSWERS_KEY = 'nlp:quizAnswersVisible';
  var DIFFICULTY_LABELS = ['Easy', 'Medium', 'Hard'];

  /* ---------------------------------------------------------------- helpers */

  /** 1-3 -> a word. Anything else -> null, so the badge is simply omitted. */
  function difficultyLabel(level) {
    var n = Number(level);
    if (!n || n < 1 || n > DIFFICULTY_LABELS.length) return null;
    return DIFFICULTY_LABELS[n - 1];
  }

  /** Author list -> one display string. Non-arrays and empties give ''. */
  function authorsLine(authors) {
    if (!Array.isArray(authors)) return '';
    return authors.filter(function (a) { return typeof a === 'string' && a.trim(); })
      .join(', ');
  }

  /** 0-1 confidence -> "84%". null when there is nothing to show. */
  function confidencePercent(score) {
    if (typeof score !== 'number' || isNaN(score)) return null;
    return Math.round(score * 100) + '%';
  }

  /**
   * True when a URL is safe to put in an href.
   *
   * Relative URLs are fine; absolute ones must be http/https/file. `file://` is allowed
   * because uploaded PDFs are stored as `file:///app/data/uploads/<hash>.pdf` (see
   * AGENTS.md) — the browser will refuse to follow it, but showing the path is honest and
   * it cannot execute. `javascript:` and `data:` are the ones that matter.
   */
  function isSafeHref(url) {
    if (typeof url !== 'string' || !url.trim()) return false;
    var scheme = /^([a-z][a-z0-9+.-]*):/i.exec(url.trim());
    if (!scheme) return true;
    var s = scheme[1].toLowerCase();
    return s === 'http' || s === 'https' || s === 'file';
  }

  /** The label a toggle should carry: it names the action, not the state. */
  function toggleLabel(answersVisible) {
    return answersVisible ? 'Hide answers' : 'Show answers';
  }

  /* ------------------------------------------------- answers-visible setting */

  /**
   * Read the persisted preference. Defaults to visible: reading the cards straight
   * through is the common use. localStorage throws in some privacy modes, so every
   * access is guarded and a failure just means the default.
   */
  function answersVisible() {
    try {
      return global.localStorage.getItem(ANSWERS_KEY) !== 'false';
    } catch (e) {
      return true;
    }
  }

  function setAnswersVisible(visible) {
    try {
      global.localStorage.setItem(ANSWERS_KEY, visible ? 'true' : 'false');
    } catch (e) {
      /* Preference is not persisted this session; the toggle still works in-page. */
    }
  }

  /* ------------------------------------------------------------ DOM building */

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null && text !== '') node.textContent = String(text);
    return node;
  }

  /** A labelled block: bold label, then the value as its own node. */
  function labelled(labelText, valueNode) {
    var wrap = el('div', 'modal-field');
    wrap.appendChild(el('span', 'metadata-label', labelText));
    wrap.appendChild(valueNode);
    return wrap;
  }

  function bulletList(items) {
    var ul = el('ul', 'json-list');
    items.forEach(function (item) {
      if (item === null || item === undefined || item === '') return;
      ul.appendChild(el('li', null, item));
    });
    return ul;
  }

  function section(headingText) {
    var s = el('div', 'paper-section');
    s.appendChild(el('h4', null, headingText));
    return s;
  }

  function difficultyBadge(level) {
    var label = difficultyLabel(level);
    if (!label) return null;
    var badge = el('span', 'difficulty-badge difficulty-' + Number(level), label);
    return badge;
  }

  function metadataItem(labelText, value) {
    var item = el('div', 'metadata-item');
    item.appendChild(el('span', 'metadata-label', labelText));
    item.appendChild(document.createTextNode(String(value)));
    return item;
  }

  /* ------------------------------------------------------------- renderers */

  function renderPaper(paper) {
    var s = section('📄 Paper Information');
    var grid = el('div', 'metadata-grid');

    if (paper.id) grid.appendChild(metadataItem('ID:', paper.id));
    var authors = authorsLine(paper.authors);
    if (authors) grid.appendChild(metadataItem('Authors:', authors));
    if (paper.year) grid.appendChild(metadataItem('Year:', paper.year));
    if (paper.venue) grid.appendChild(metadataItem('Venue:', paper.venue));
    if (paper.citation_count) grid.appendChild(metadataItem('Citations:', paper.citation_count));
    if (paper.pillar_id) grid.appendChild(metadataItem('Pillar:', paper.pillar_id));

    if (paper.url_pdf) {
      var pdfItem = el('div', 'metadata-item');
      pdfItem.appendChild(el('span', 'metadata-label', 'PDF:'));
      if (isSafeHref(paper.url_pdf)) {
        var link = el('a', null, 'View PDF');
        link.setAttribute('href', paper.url_pdf);
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
        pdfItem.appendChild(link);
      } else {
        pdfItem.appendChild(el('span', 'modal-muted', paper.url_pdf));
      }
      grid.appendChild(pdfItem);
    }

    s.appendChild(grid);
    if (paper.abstract) s.appendChild(labelled('Abstract:', el('p', null, paper.abstract)));
    return s;
  }

  function renderNotes(notes) {
    var s = section('📝 Research Notes');
    notes.forEach(function (note) {
      var card = el('div', 'modal-card');
      if (note.problem) card.appendChild(labelled('Problem:', el('p', null, note.problem)));
      if (note.method) card.appendChild(labelled('Method:', el('p', null, note.method)));
      if (Array.isArray(note.findings) && note.findings.length) {
        card.appendChild(labelled('Key Findings:', bulletList(note.findings)));
      }
      if (Array.isArray(note.limitations) && note.limitations.length) {
        card.appendChild(labelled('Limitations:', bulletList(note.limitations)));
      }
      if (Array.isArray(note.key_terms) && note.key_terms.length) {
        var terms = el('div', 'term-list');
        note.key_terms.forEach(function (term) {
          if (term) terms.appendChild(el('span', 'term-chip', term));
        });
        card.appendChild(labelled('Key Terms:', terms));
      }
      var confidence = confidencePercent(note.confidence_score);
      if (confidence) {
        card.appendChild(labelled('Confidence Score:', el('span', null, confidence)));
      }
      s.appendChild(card);
    });
    return s;
  }

  function renderLessons(lessons) {
    var s = section('🎓 Lessons & Takeaways');
    lessons.forEach(function (lesson) {
      var card = el('div', 'modal-card');
      if (lesson.tl_dr) {
        var tldr = el('div', 'lesson-tldr');
        tldr.appendChild(el('strong', null, 'TL;DR:'));
        tldr.appendChild(document.createTextNode(' ' + lesson.tl_dr));
        card.appendChild(tldr);
      }
      if (Array.isArray(lesson.takeaways) && lesson.takeaways.length) {
        card.appendChild(labelled('Key Takeaways:', bulletList(lesson.takeaways)));
      }
      if (Array.isArray(lesson.practice_ideas) && lesson.practice_ideas.length) {
        card.appendChild(labelled('Practice Ideas:', bulletList(lesson.practice_ideas)));
      }
      if (Array.isArray(lesson.connections) && lesson.connections.length) {
        card.appendChild(labelled('Connections:', bulletList(lesson.connections)));
      }

      var meta = el('div', 'modal-badge-row');
      var badge = difficultyBadge(lesson.difficulty);
      if (badge) {
        badge.textContent = 'Difficulty: ' + badge.textContent;
        meta.appendChild(badge);
      }
      if (lesson.estimated_time) {
        meta.appendChild(el('span', 'time-badge', '⏱️ ' + lesson.estimated_time + ' min read'));
      }
      if (meta.childNodes.length) card.appendChild(meta);

      s.appendChild(card);
    });
    return s;
  }

  /**
   * The quiz section, with the one control that shows or hides every answer in it.
   *
   * The control lives in the section header rather than on each card: see the file header
   * for why. It carries aria-expanded/aria-controls over the list it governs, and each
   * answer gets the `hidden` attribute so hiding removes it from the accessibility tree
   * as well as the page.
   */
  function renderQuizCards(quizCards) {
    var s = el('div', 'paper-section');
    var listId = 'modal-quiz-cards';
    var list = el('div', 'quiz-card-list');
    list.id = listId;

    var visible = answersVisible();
    var answerNodes = [];

    var toggle = el('button', 'quiz-answers-toggle', toggleLabel(visible));
    toggle.setAttribute('type', 'button');
    toggle.setAttribute('aria-controls', listId);
    toggle.setAttribute('aria-expanded', visible ? 'true' : 'false');

    function apply(show) {
      answerNodes.forEach(function (node) { node.hidden = !show; });
      toggle.textContent = toggleLabel(show);
      toggle.setAttribute('aria-expanded', show ? 'true' : 'false');
    }

    toggle.addEventListener('click', function () {
      visible = !visible;
      setAnswersVisible(visible);
      apply(visible);
    });

    var header = el('div', 'quiz-section-header');
    header.appendChild(el('h4', null, '🧠 Quiz Cards'));
    header.appendChild(toggle);
    s.appendChild(header);

    quizCards.forEach(function (quiz) {
      var card = el('div', 'quiz-card');
      card.appendChild(el('div', 'quiz-question', 'Q: ' + (quiz.question || '')));

      var answer = el('div', 'quiz-answer', 'A: ' + (quiz.answer || ''));
      answerNodes.push(answer);
      card.appendChild(answer);

      var badge = difficultyBadge(quiz.difficulty);
      if (badge) {
        var row = el('div', 'modal-badge-row');
        row.appendChild(badge);
        card.appendChild(row);
      }
      list.appendChild(card);
    });

    apply(visible);
    s.appendChild(list);
    return s;
  }

  function renderDetails(data, titleEl, contentEl) {
    var paper = data.paper || {};
    titleEl.textContent = paper.title || 'Paper details';
    contentEl.replaceChildren();

    contentEl.appendChild(renderPaper(paper));
    if (Array.isArray(data.notes) && data.notes.length) {
      contentEl.appendChild(renderNotes(data.notes));
    }
    if (Array.isArray(data.lessons) && data.lessons.length) {
      contentEl.appendChild(renderLessons(data.lessons));
    }
    if (Array.isArray(data.quiz_cards) && data.quiz_cards.length) {
      contentEl.appendChild(renderQuizCards(data.quiz_cards));
    }
  }

  function renderMessage(contentEl, headline, detail) {
    contentEl.replaceChildren();
    var box = el('div', 'modal-message');
    box.appendChild(el('p', null, headline));
    if (detail) box.appendChild(el('p', 'modal-muted', detail));
    contentEl.appendChild(box);
  }

  function setRefreshStatus(statusEl, message, kind) {
    if (!statusEl) return;
    statusEl.hidden = !message;
    statusEl.textContent = message || '';
    statusEl.classList.remove('is-success', 'is-info', 'is-error');
    if (message && kind) statusEl.classList.add(kind);
  }

  /* ------------------------------------------------------------------ wiring */

  /**
   * Wire the shared dialog up on a page.
   *
   * `triggerSelector` is matched with a delegated listener on document, so rows rendered
   * after init (the pillar page's search results, say) work without re-initialising.
   */
  function init(options) {
    var opts = options || {};
    var dialog = document.getElementById(opts.dialogId || 'paper-details-modal');
    if (!dialog) return null;

    var titleEl = document.getElementById(opts.titleId || 'modal-paper-title');
    var contentEl = document.getElementById(opts.contentId || 'modal-content');
    var closeEl = document.getElementById(opts.closeId || 'close-modal');
    var refreshBtn = document.getElementById(opts.refreshId || 'refresh-metadata-btn');
    var refreshStatusEl = document.getElementById(opts.refreshStatusId || 'metadata-refresh-status');
    var selector = opts.triggerSelector || '[data-paper-id]';
    var requestSeq = 0;
    var refreshSeq = 0;
    var currentPaperId = null;

    function loadDetails(paperId, seq) {
      return fetch(DETAILS_URL + encodeURIComponent(paperId))
        .then(function (response) {
          if (!response.ok) throw new Error('HTTP ' + response.status + ' ' + response.statusText);
          return response.json();
        })
        .then(function (data) {
          if (seq !== requestSeq) return null;
          renderDetails(data, titleEl, contentEl);
          return data;
        });
    }

    function open(paperId) {
      if (!paperId) return;
      currentPaperId = paperId;
      var seq = ++requestSeq;
      titleEl.textContent = 'Loading…';
      renderMessage(contentEl, 'Loading paper details…');
      setRefreshStatus(refreshStatusEl, '', null);
      if (refreshBtn) {
        refreshBtn.hidden = true;
        refreshBtn.disabled = true;
      }
      if (!dialog.open) dialog.showModal();

      loadDetails(paperId, seq)
        .then(function () {
          if (seq !== requestSeq) return;
          if (refreshBtn) {
            refreshBtn.hidden = false;
            refreshBtn.disabled = false;
          }
        })
        .catch(function (error) {
          if (seq !== requestSeq) return;
          titleEl.textContent = 'Paper details';
          renderMessage(contentEl, 'Could not load paper details.', error.message);
        });
    }

    function refreshMetadata() {
      if (!currentPaperId || !refreshBtn || refreshBtn.disabled) return;
      var seq = ++refreshSeq;
      refreshBtn.disabled = true;
      setRefreshStatus(refreshStatusEl, 'Refreshing metadata…', 'is-info');

      fetch(REFRESH_URL + encodeURIComponent(currentPaperId) + '/refresh-metadata', {
        method: 'POST',
      })
        .then(function (response) {
          return response.json().then(function (body) {
            if (!response.ok) {
              var detail = body && body.detail;
              var message = typeof detail === 'string'
                ? detail
                : 'HTTP ' + response.status + ' ' + response.statusText;
              throw new Error(message);
            }
            return body;
          });
        })
        .then(function (body) {
          if (seq !== refreshSeq) return;
          var kind = body.updated ? 'is-success' : 'is-info';
          setRefreshStatus(refreshStatusEl, body.message || 'Metadata refresh finished.', kind);
          return loadDetails(currentPaperId, requestSeq);
        })
        .then(function () {
          if (seq !== refreshSeq) return;
          refreshBtn.disabled = false;
        })
        .catch(function (error) {
          if (seq !== refreshSeq) return;
          setRefreshStatus(refreshStatusEl, error.message, 'is-error');
          refreshBtn.disabled = false;
        });
    }

    document.addEventListener('click', function (event) {
      var trigger = event.target.closest ? event.target.closest(selector) : null;
      if (!trigger || !dialog.isConnected) return;
      if (!trigger.dataset || !trigger.dataset.paperId) return;
      event.preventDefault();
      open(trigger.dataset.paperId);
    });

    if (closeEl) closeEl.addEventListener('click', function () { dialog.close(); });
    if (refreshBtn) refreshBtn.addEventListener('click', refreshMetadata);
    dialog.addEventListener('click', function (event) {
      if (event.target === dialog) dialog.close();
    });

    return { open: open };
  }

  global.PaperModal = {
    init: init,
    difficultyLabel: difficultyLabel,
    authorsLine: authorsLine,
    confidencePercent: confidencePercent,
    isSafeHref: isSafeHref,
    toggleLabel: toggleLabel,
    answersVisible: answersVisible,
    setAnswersVisible: setAnswersVisible,
  };

  // Also reachable from `node --test` (tests/js/). CommonJS on purpose, same as
  // run-progress.js: the browser loads this with a plain <script> tag and the project has
  // no build step. Only the DOM-free helpers are exported.
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      difficultyLabel: difficultyLabel,
      authorsLine: authorsLine,
      confidencePercent: confidencePercent,
      isSafeHref: isSafeHref,
      toggleLabel: toggleLabel,
    };
  }
})(typeof window !== 'undefined' ? window : globalThis);
