/*
 * Library list: client-side search, pillar and status filters, j/k/Enter.
 *
 * The pillar query (?pillar=) is still a real link, so the filter works with
 * scripts off. With scripts on, the same chips filter the rows already on the
 * page and write the query back, which is the hundred newest papers rather
 * than a second request.
 *
 * j/k and Enter are extra. The rows are ordinary links, and the shortcut is
 * ignored while a text field has focus.
 */
(function (global) {
  'use strict';

  function matchesRow(row, query, pillar, status) {
    if (pillar && row.pillar !== pillar) return false;
    if (status === 'lesson' && !row.lesson) return false;
    if (status === 'cards' && !row.cards) return false;
    if (status === 'episode' && !row.episode) return false;
    if (query && row.search.indexOf(query) === -1) return false;
    return true;
  }

  function nextIndex(current, delta, count) {
    if (!count) return -1;
    if (current < 0) return delta > 0 ? 0 : count - 1;
    return (current + delta + count) % count;
  }

  function countLabel(count) {
    if (count === 1) return '1 paper';
    if (!count) return '';
    return count + ' papers';
  }

  function readRow(el) {
    return {
      pillar: el.getAttribute('data-pillar') || '',
      search: el.getAttribute('data-search') || '',
      lesson: el.getAttribute('data-lesson') === '1',
      cards: el.getAttribute('data-cards') === '1',
      episode: el.getAttribute('data-episode') === '1',
    };
  }

  function typingTarget(el) {
    if (!el || !el.tagName) return false;
    var tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
  }

  function boot() {
    if (typeof document === 'undefined') return;
    var list = document.getElementById('library-list');
    if (!list) return;
    var rows = Array.prototype.slice.call(list.querySelectorAll('.library-row'));
    var search = document.getElementById('library-search');
    var countEl = document.getElementById('library-count');
    var empty = document.getElementById('library-no-match');
    var pillarChips = Array.prototype.slice.call(
      document.querySelectorAll('.filter-chip[data-pillar]')
    );
    var statusChips = Array.prototype.slice.call(
      document.querySelectorAll('.filter-chip[data-status]')
    );
    var pillar = '';
    var status = '';
    var current = -1;

    pillarChips.forEach(function (chip) {
      if (chip.getAttribute('aria-current') === 'true') {
        pillar = chip.getAttribute('data-pillar') || '';
      }
    });

    function visibleRows() {
      return rows.filter(function (row) { return !row.hidden; });
    }

    function paintCurrent(index, scroll) {
      rows.forEach(function (row) { row.classList.remove('is-current'); });
      current = index;
      var visible = visibleRows();
      if (index < 0 || index >= visible.length) return;
      visible[index].classList.add('is-current');
      if (scroll && visible[index].scrollIntoView) {
        visible[index].scrollIntoView({ block: 'nearest' });
      }
    }

    function apply() {
      var query = search ? search.value.trim().toLowerCase() : '';
      var marked = rows.filter(function (row) {
        return row.classList.contains('is-current');
      })[0];
      var shown = 0;
      rows.forEach(function (el) {
        var match = matchesRow(readRow(el), query, pillar, status);
        el.hidden = !match;
        if (match) shown += 1;
      });
      if (countEl) countEl.textContent = countLabel(shown);
      if (empty) empty.hidden = shown !== 0 || rows.length === 0;
      var visible = visibleRows();
      var index = marked ? visible.indexOf(marked) : -1;
      if (marked && index < 0) index = visible.length ? 0 : -1;
      paintCurrent(index, false);
    }

    function selectPillar(next, href) {
      pillar = next || '';
      pillarChips.forEach(function (chip) {
        var on = (chip.getAttribute('data-pillar') || '') === pillar;
        if (on) chip.setAttribute('aria-current', 'true');
        else chip.removeAttribute('aria-current');
      });
      if (href && global.history && global.history.replaceState) {
        global.history.replaceState(null, '', href);
      }
      apply();
    }

    pillarChips.forEach(function (chip) {
      chip.addEventListener('click', function (event) {
        event.preventDefault();
        selectPillar(chip.getAttribute('data-pillar') || '', chip.getAttribute('href'));
      });
    });

    statusChips.forEach(function (chip) {
      chip.addEventListener('click', function () {
        status = chip.getAttribute('data-status') || '';
        statusChips.forEach(function (other) {
          other.setAttribute(
            'aria-pressed',
            (other.getAttribute('data-status') || '') === status ? 'true' : 'false'
          );
        });
        apply();
      });
    });

    if (search) search.addEventListener('input', apply);

    document.addEventListener('keydown', function (event) {
      if (typingTarget(event.target)) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      var visible = visibleRows();
      if (!visible.length) return;
      if (event.key === 'j' || event.key === 'k') {
        event.preventDefault();
        var delta = event.key === 'j' ? 1 : -1;
        var index = visible.indexOf(rows.filter(function (row) {
          return row.classList.contains('is-current');
        })[0]);
        paintCurrent(nextIndex(index, delta, visible.length), true);
      } else if (event.key === 'Enter') {
        if (event.target && event.target.closest && event.target.closest('a, button')) return;
        var open = visible[current] && visible[current].querySelector('.library-row-open');
        if (!open) return;
        event.preventDefault();
        open.click();
      }
    });

    apply();
  }

  if (typeof document === 'undefined') {
    /* node --test */
  } else if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

  var api = {
    matchesRow: matchesRow,
    nextIndex: nextIndex,
    countLabel: countLabel,
  };
  global.LibraryList = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
