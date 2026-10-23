/*
 * Pillar search results. Built with createElement + textContent.
 *
 * paperPageHref is the single switch for the Library PR's paper page.
 * It returns null until GET /papers/{id} exists; View Paper then opens the
 * shared paper modal via .activity-item-link. Return
 * '/papers/' + encodeURIComponent(paperId) to navigate there instead.
 */
(function (global) {
  'use strict';

  function paperPageHref(paperId) {
    return null;
  }

  function enqueueMessage(status) {
    if (status === 'queued') return 'Queued for the next auto-run.';
    if (status === 'already_queued') return 'Already waiting for the next auto-run.';
    return 'Could not queue this paper.';
  }

  function currentPillarId() {
    var root = document.getElementById('pillar-page');
    if (root && root.dataset.pillarId) return root.dataset.pillarId;
    return global.location.pathname.split('/')[2];
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function authorsLine(authors) {
    if (!authors || !authors.length) return 'Unknown authors';
    var shown = authors.slice(0, 3).join(', ');
    return authors.length > 3 ? shown + ' et al.' : shown;
  }

  function emptyState(container, message, detail) {
    container.replaceChildren();
    var wrap = el('div', 'empty');
    wrap.appendChild(el('p', 'empty-message', message));
    if (detail) wrap.appendChild(el('p', 'empty-message', detail));
    container.appendChild(wrap);
  }

  function viewControl(paperId) {
    var href = paperPageHref(paperId);
    if (href) {
      var link = el('a', 'btn btn-secondary', 'View Paper');
      link.href = href;
      return link;
    }
    var button = el('button', 'btn btn-secondary activity-item-link', 'View Paper');
    button.type = 'button';
    button.dataset.paperId = paperId;
    return button;
  }

  function resultCard(result) {
    var card = el('article', 'search-result');
    var main = el('div', 'search-result-main');
    main.appendChild(el('h3', 'search-result-title', result.title || 'Untitled'));
    var meta = el('p', 'search-result-meta');
    meta.textContent = [
      authorsLine(result.authors),
      result.year ? String(result.year) : 'Unknown year',
      result.venue || ''
    ].filter(Boolean).join(' · ');
    main.appendChild(meta);
    card.appendChild(main);

    var score = el('p', 'search-result-score');
    score.appendChild(el('span', 'search-result-score-value', result.score == null ? '' : String(result.score)));
    var caption = el('span', 'search-result-score-label', 'Relevance');
    caption.title = 'Vector similarity: ' + result.cosine_similarity + ', Keyword overlap: ' + result.keyword_overlap;
    score.appendChild(caption);
    card.appendChild(score);

    var actions = el('div', 'search-result-actions');
    var enqueue = el('button', 'btn btn-ghost enqueue-btn', 'Enqueue');
    enqueue.type = 'button';
    enqueue.dataset.paperId = result.paper_id;
    actions.appendChild(enqueue);
    actions.appendChild(viewControl(result.paper_id));
    var status = el('p', 'enqueue-status');
    status.setAttribute('role', 'status');
    actions.appendChild(status);
    card.appendChild(actions);
    return card;
  }

  function displaySearchResults(results, query) {
    var container = document.getElementById('search-results-container');
    if (!results || !results.length) {
      emptyState(container, 'No papers found for “' + query + '”', 'Try different keywords or broaden your search');
      return;
    }
    container.replaceChildren();
    var heading = el('h3', 'section-title', 'Found ' + results.length + ' result' + (results.length === 1 ? '' : 's'));
    container.appendChild(heading);
    var list = el('div', 'search-results');
    results.forEach(function (result) { list.appendChild(resultCard(result)); });
    container.appendChild(list);
  }

  function enqueuePaper(button) {
    var paperId = button.dataset.paperId;
    var status = button.parentElement.querySelector('.enqueue-status');
    button.disabled = true;
    if (status) status.textContent = 'Queuing…';
    fetch('/api/pillars/' + encodeURIComponent(currentPillarId()) + '/queue', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paper_id: paperId })
    })
      .then(function (response) {
        return response.json().catch(function () { return {}; }).then(function (body) {
          if (!response.ok) {
            var detail = body && body.detail;
            throw new Error(typeof detail === 'string' ? detail : 'Could not queue this paper');
          }
          if (status) status.textContent = enqueueMessage(body.status);
        });
      })
      .catch(function (error) {
        if (status) status.textContent = error.message;
      })
      .then(function () {
        button.disabled = false;
      });
  }

  function performSearch(query, topK) {
    var searchBtn = document.querySelector('.search-btn');
    var container = document.getElementById('search-results-container');
    if (searchBtn) {
      searchBtn.disabled = true;
      searchBtn.textContent = 'Searching…';
    }
    emptyState(container, 'Searching…');
    var url = '/api/pillars/' + encodeURIComponent(currentPillarId())
      + '/search?q=' + encodeURIComponent(query) + '&top_k=' + encodeURIComponent(topK);
    fetch(url)
      .then(function (response) {
        if (!response.ok) throw new Error('Search failed: ' + response.status + ' ' + response.statusText);
        return response.json();
      })
      .then(function (results) { displaySearchResults(results, query); })
      .catch(function (error) {
        container.replaceChildren();
        var wrap = el('div', 'empty');
        wrap.appendChild(el('p', 'empty-message', 'Search failed: ' + error.message));
        var retry = el('button', 'btn btn-secondary btn-sm', 'Try again');
        retry.type = 'button';
        retry.addEventListener('click', function () { global.location.reload(); });
        wrap.appendChild(retry);
        container.appendChild(wrap);
      })
      .then(function () {
        if (searchBtn) {
          searchBtn.disabled = false;
          searchBtn.textContent = 'Search';
        }
      });
  }

  function init() {
    var form = document.getElementById('search-form');
    if (form) {
      form.addEventListener('submit', function (event) {
        event.preventDefault();
        var data = new FormData(form);
        var query = String(data.get('q') || '').trim();
        var topK = parseInt(data.get('top_k') || '10', 10);
        if (!query) return;
        performSearch(query, topK);
      });
    }
    document.addEventListener('click', function (event) {
      var button = event.target.closest ? event.target.closest('.enqueue-btn') : null;
      if (!button || button.disabled) return;
      enqueuePaper(button);
    });
  }

  global.PillarSearch = {
    init: init,
    paperPageHref: paperPageHref,
    enqueueMessage: enqueueMessage
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { paperPageHref: paperPageHref, enqueueMessage: enqueueMessage };
  }
})(typeof window !== 'undefined' ? window : globalThis);
