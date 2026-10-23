/*
 * Lessons list: j/k move, Enter opens. The rows are ordinary links, so the
 * keys are an extra. They do nothing while a text field is focused, and the
 * ? overlay lists them. The shell does not dispatch these: this page does.
 */
(function (global) {
  'use strict';

  function nextIndex(current, delta, count) {
    if (!count) return -1;
    if (current < 0) return delta > 0 ? 0 : count - 1;
    return (current + delta + count) % count;
  }

  function typingTarget(el) {
    if (!el || !el.tagName) return false;
    var tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
  }

  function boot() {
    if (typeof document === 'undefined') return;
    var list = document.getElementById('lesson-list');
    if (!list) return;
    var rows = Array.prototype.slice.call(list.querySelectorAll('.lesson-row'));
    if (!rows.length) return;
    var current = -1;

    if (global.NlpShortcuts && typeof global.NlpShortcuts.register === 'function') {
      global.NlpShortcuts.register({
        id: 'lessons-next',
        group: 'Lessons',
        label: 'Next lesson',
        keys: 'J',
      });
      global.NlpShortcuts.register({
        id: 'lessons-prev',
        group: 'Lessons',
        label: 'Previous lesson',
        keys: 'K',
      });
      global.NlpShortcuts.register({
        id: 'lessons-open',
        group: 'Lessons',
        label: 'Open the lesson',
        keys: 'Enter',
      });
    }

    function paint(index, scroll) {
      rows.forEach(function (row) { row.classList.remove('is-current'); });
      current = index;
      if (index < 0 || index >= rows.length) return;
      rows[index].classList.add('is-current');
      if (scroll && rows[index].scrollIntoView) {
        rows[index].scrollIntoView({ block: 'nearest' });
      }
    }

    document.addEventListener('keydown', function (event) {
      if (typingTarget(event.target)) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === 'j' || event.key === 'k') {
        event.preventDefault();
        paint(nextIndex(current, event.key === 'j' ? 1 : -1, rows.length), true);
      } else if (event.key === 'Enter') {
        if (event.target && event.target.closest && event.target.closest('a, button')) return;
        var open = rows[current] && rows[current].querySelector('.lesson-open');
        if (!open) return;
        event.preventDefault();
        open.click();
      }
    });
  }

  if (typeof document === 'undefined') {
    /* node --test */
  } else if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

  var api = { nextIndex: nextIndex };
  global.LessonsList = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
