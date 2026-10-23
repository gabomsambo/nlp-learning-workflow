/*
 * New / Edit / Delete pillar dialogs, shared by the pillars list and the pillar page.
 *
 * The save control is type="button" and Enter is handled here. A submit button
 * inside the dialog used to also navigate to /pillars/?name=… while the fetch ran.
 */
(function (global) {
  'use strict';

  function validatePillarField(fieldName, value) {
    var text = value == null ? '' : String(value).trim();
    if (fieldName === 'name') {
      if (!text) return 'Name is required';
      if (text.length > 100) return 'Name must be 100 characters or less';
      return '';
    }
    if (fieldName === 'goal') {
      if (!text) return 'Goal is required';
      if (text.length < 10) return 'Goal must be at least 10 characters';
      if (text.length > 500) return 'Goal must be 500 characters or less';
      return '';
    }
    return '';
  }

  function init() {
    var modal = document.getElementById('pillar-modal');
    var deleteModal = document.getElementById('delete-modal');
    var form = document.getElementById('pillar-form');
    if (!modal || !form) return;

    var modalTitle = document.getElementById('modal-title');
    var currentPillarId = null;
    var isEditing = false;

    function showFeedback(fieldName, message) {
      var feedback = document.getElementById(fieldName + '-feedback');
      if (feedback) feedback.textContent = message || '';
      var input = document.getElementById('pillar-' + fieldName);
      if (input) input.setAttribute('aria-invalid', message ? 'true' : 'false');
    }

    function clearFeedback() {
      ['name', 'goal'].forEach(function (field) { showFeedback(field, ''); });
    }

    function validateField(fieldName) {
      var input = document.getElementById('pillar-' + fieldName);
      var message = validatePillarField(fieldName, input ? input.value : '');
      showFeedback(fieldName, message);
      return !message;
    }

    function validateForm() {
      return validateField('name') && validateField('goal');
    }

    function closeModal() {
      modal.close();
      form.reset();
      clearFeedback();
      currentPillarId = null;
      isEditing = false;
    }

    function openModal(pillarId) {
      currentPillarId = pillarId || null;
      isEditing = !!pillarId;
      if (modalTitle) modalTitle.textContent = isEditing ? 'Edit Pillar' : 'New Pillar';
      if (isEditing) {
        loadPillarData(pillarId);
      } else {
        form.reset();
        clearFeedback();
        var perDay = document.getElementById('pillar-papers-per-day');
        if (perDay) perDay.value = '2';
      }
      modal.showModal();
    }

    function loadPillarData(pillarId) {
      fetch('/api/pillars/' + encodeURIComponent(pillarId))
        .then(function (response) {
          if (!response.ok) throw new Error('Failed to load pillar data');
          return response.json();
        })
        .then(function (pillar) {
          document.getElementById('pillar-name').value = pillar.name || '';
          document.getElementById('pillar-goal').value = pillar.goal || '';
          document.getElementById('pillar-focus-areas').value = (pillar.focus_areas || []).join('\n');
          document.getElementById('pillar-papers-per-day').value = pillar.papers_per_day;
        })
        .catch(function (error) {
          showFeedback('name', error.message || 'Failed to load pillar data');
        });
    }

    function savePillar() {
      if (!validateForm()) return;
      var data = new FormData(form);
      var focusAreas = String(data.get('focus_areas') || '')
        .split('\n')
        .map(function (area) { return area.trim(); })
        .filter(function (area) { return area.length > 0; });
      var pillarData = {
        name: String(data.get('name') || '').trim(),
        goal: String(data.get('goal') || '').trim(),
        focus_areas: focusAreas,
        papers_per_day: parseInt(data.get('papers_per_day'), 10)
      };
      var saveBtn = document.getElementById('modal-save');
      saveBtn.disabled = true;
      saveBtn.textContent = 'Saving...';
      var url = isEditing ? '/api/pillars/' + encodeURIComponent(currentPillarId) : '/api/pillars/';
      fetch(url, {
        method: isEditing ? 'PATCH' : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(pillarData)
      })
        .then(function (response) {
          return response.json().catch(function () { return {}; }).then(function (body) {
            if (!response.ok) {
              var detail = body && body.detail;
              throw new Error(typeof detail === 'string' ? detail : 'Failed to save pillar');
            }
            global.location.reload();
          });
        })
        .catch(function (error) {
          showFeedback('name', error.message);
          saveBtn.disabled = false;
          saveBtn.textContent = 'Save Pillar';
        });
    }

    function closeDeleteModal() {
      if (deleteModal) deleteModal.close();
      currentPillarId = null;
    }

    function confirmDelete(button) {
      if (!deleteModal) return;
      currentPillarId = button.dataset.pillarId;
      var nameEl = document.getElementById('delete-pillar-name');
      if (nameEl) nameEl.textContent = button.dataset.pillarName || 'this pillar';
      deleteModal.showModal();
    }

    function deletePillar() {
      var deleteBtn = document.getElementById('delete-confirm');
      if (!deleteBtn || !currentPillarId) return;
      deleteBtn.disabled = true;
      deleteBtn.textContent = 'Deleting...';
      fetch('/api/pillars/' + encodeURIComponent(currentPillarId), { method: 'DELETE' })
        .then(function (response) {
          if (response.status === 204) {
            global.location.reload();
            return;
          }
          return response.json().catch(function () { return {}; }).then(function (body) {
            var detail = body && body.detail;
            throw new Error(typeof detail === 'string' ? detail : 'Failed to delete pillar');
          });
        })
        .catch(function (error) {
          global.alert('Failed to delete pillar: ' + error.message);
          deleteBtn.disabled = false;
          deleteBtn.textContent = 'Delete';
        });
    }

    document.querySelectorAll('.btn-new-pillar').forEach(function (btn) {
      btn.addEventListener('click', function () { openModal(); });
    });
    document.querySelectorAll('.btn-edit').forEach(function (btn) {
      btn.addEventListener('click', function () { openModal(btn.dataset.pillarId); });
    });
    document.querySelectorAll('.btn-delete').forEach(function (btn) {
      btn.addEventListener('click', function () { confirmDelete(btn); });
    });

    var closeBtn = document.getElementById('modal-close');
    var cancelBtn = document.getElementById('modal-cancel');
    if (closeBtn) closeBtn.addEventListener('click', closeModal);
    if (cancelBtn) cancelBtn.addEventListener('click', closeModal);
    document.getElementById('modal-save').addEventListener('click', savePillar);
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      savePillar();
    });

    var nameInput = document.getElementById('pillar-name');
    var goalInput = document.getElementById('pillar-goal');
    if (nameInput) nameInput.addEventListener('input', function () { validateField('name'); });
    if (goalInput) goalInput.addEventListener('input', function () { validateField('goal'); });

    ['delete-cancel', 'delete-cancel-btn'].forEach(function (id) {
      var button = document.getElementById(id);
      if (button) button.addEventListener('click', closeDeleteModal);
    });
    var deleteConfirm = document.getElementById('delete-confirm');
    if (deleteConfirm) deleteConfirm.addEventListener('click', deletePillar);
  }

  global.PillarForm = { init: init, validatePillarField: validatePillarField };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { validatePillarField: validatePillarField };
  }
})(typeof window !== 'undefined' ? window : globalThis);
