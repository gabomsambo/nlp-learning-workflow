/*
 * Upload panel for the pillar page. Same behaviour as the inline script it
 * replaced: explicit summarizer/quiz booleans, linked checkboxes, run progress
 * for kind=upload only, outcome box, recent uploads from pipeline_runs.
 *
 * pillar id comes from #pillar-page[data-pillar-id], with the old pathname
 * split as a fallback so a route that still looks like /pillars/{id} works.
 */
(function (global) {
  'use strict';

  function currentPillarId() {
    var root = document.getElementById('pillar-page');
    if (root && root.dataset.pillarId) return root.dataset.pillarId;
    return global.location.pathname.split('/')[2];
  }

// Upload Interface Functions
function setupUploadInterface() {
    const pillarId = currentPillarId();
    
    // File drop zone
    const dropZone = document.getElementById('file-drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileUploadForm = document.getElementById('file-upload-form');
    
    // Click to browse
    dropZone.addEventListener('click', (event) => {
        if (event.target === fileInput) return;
        fileInput.click();
    });
    
    // Drag and drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files);
        const pdfFiles = files.filter(file => file.type === 'application/pdf');
        
        if (pdfFiles.length === 0) {
            alert('Please select PDF files only.');
            return;
        }
        
        handleFileSelection(pdfFiles[0]); // Handle first file for now
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });
    
    // Form handlers
    document.getElementById('pdf-upload-form').addEventListener('submit', (e) => {
        e.preventDefault();
        submitFileUpload(pillarId);
    });
    
    document.getElementById('url-upload-form').addEventListener('submit', (e) => {
        e.preventDefault();
        submitUrlUpload(pillarId);
    });
    
    linkSummarizerAndQuiz('file-run-summarizer', 'file-generate-quiz');
    linkSummarizerAndQuiz('url-run-summarizer', 'url-generate-quiz');

    document.getElementById('cancel-file-upload').addEventListener('click', () => {
        fileUploadForm.style.display = 'none';
        dropZone.style.display = 'block';
        document.getElementById('pdf-upload-form').reset();
    });
}

function handleFileSelection(file) {
    const dropZone = document.getElementById('file-drop-zone');
    const fileUploadForm = document.getElementById('file-upload-form');
    
    // Hide drop zone, show form
    dropZone.style.display = 'none';
    fileUploadForm.style.display = 'block';
    
    // Pre-populate title from filename
    const filename = file.name.replace('.pdf', '');
    document.getElementById('file-title').value = filename;
    
    // Store file reference
    window.selectedFile = file;
}

/* --------------------------------------------------------- upload progress */

/**
 * The elements RunProgress renders into. Rebuilt per call rather than cached: the
 * poller keeps a reference for the life of the run, and a stale node from before a
 * re-render would be updated invisibly.
 */
function uploadProgressEls() {
    return {
        status: document.getElementById('upload-run-status'),
        stages: document.getElementById('upload-run-stages'),
        progress: document.getElementById('upload-run-bar'),
        cancel: document.getElementById('upload-cancel-btn'),
    };
}

function showUploadPanel(show) {
    const panel = document.getElementById('upload-progress-section');
    if (panel) panel.hidden = !show;
}

/**
 * Both upload buttons follow the one active upload run, because they both feed it.
 * `dataset.idleText` remembers the label rather than hardcoding it in two places.
 */
function setUploadBusy(busy) {
    ['submit-url-upload', 'submit-file-upload'].forEach((id) => {
        const btn = document.getElementById(id);
        if (!btn) return;
        if (busy && !btn.dataset.idleText) btn.dataset.idleText = btn.textContent;
        btn.disabled = busy;
        btn.textContent = busy ? 'Uploading…' : (btn.dataset.idleText || btn.textContent);
    });
}

let uploadRunConfirmed = false;

/**
 * Called on every poll. Its first job is to reject runs that are not uploads.
 *
 * RunProgress stores the active run id under one key shared by every page that uses
 * it, so a discovery run started on /discovery is exactly what reattach() finds here.
 * Rendering its stages under "Upload Progress" would be another run's progress under
 * this page's heading — a display that is wrong rather than merely empty.
 */
function onUploadRun(run) {
    if (!run) return;
    if (run.kind !== 'upload') {
        RunProgress.detach();
        showUploadPanel(false);
        setUploadBusy(false);
        return;
    }
    if (!uploadRunConfirmed) {
        uploadRunConfirmed = true;
        showUploadPanel(true);
        setUploadBusy(!RunProgress.isTerminal(run.status));
    }
}

/**
 * Fires for a finished run AND for a run that turned out not to exist (run === null).
 * Both must re-enable the buttons; RunProgress's 404 path is the only thing that can
 * report the second case, and skipping it here leaves the page stuck on "Uploading…".
 */
function onUploadFinished(run) {
    uploadRunConfirmed = false;
    setUploadBusy(false);
    if (!run) return;
    renderUploadOutcome(run);
    loadRecentUploads();
}

function startUploadRun(runId) {
    uploadRunConfirmed = true;
    const box = document.getElementById('upload-result');
    if (box) {
        box.replaceChildren();
        box.hidden = true;
    }
    showUploadPanel(true);
    setUploadBusy(true);
    RunProgress.attach(runId, uploadProgressEls(), onUploadFinished, onUploadRun);
}

/**
 * Read the 202 (or the failure) from an upload route and start following the run.
 */
async function dispatchUpload(url, init) {
    const response = await fetch(url, init);
    let payload = {};
    try {
        payload = await response.json();
    } catch (e) { /* a non-JSON error page; the status below still says enough */ }

    if (!response.ok || !payload.run_id) {
        throw new Error(payload.detail || payload.message || `HTTP ${response.status}`);
    }
    startUploadRun(payload.run_id);
}

function reportUploadDispatchFailure(error) {
    // The panel is the one place upload state is reported, so a dispatch that never
    // produced a run says so there rather than in an alert() the user must dismiss
    // before they can read the form again.
    showUploadPanel(true);
    setUploadBusy(false);
    const statusEl = document.getElementById('upload-run-status');
    if (statusEl) {
        statusEl.textContent = 'Could not start the upload — ' + error.message;
        statusEl.classList.add('is-danger');
    }
    const stages = document.getElementById('upload-run-stages');
    if (stages) stages.replaceChildren();
}

async function submitFileUpload(pillarId) {
    const formData = new FormData();
    const form = document.getElementById('pdf-upload-form');

    // Validate that we have a file
    if (!window.selectedFile) {
        alert('Please select a file first');
        return;
    }

    // Add file
    formData.append('file', window.selectedFile);

    // Add form data
    const CHECKBOX_FIELDS = ['run_summarizer', 'generate_quiz'];
    const formDataObj = new FormData(form);
    for (const [key, value] of formDataObj.entries()) {
        if (CHECKBOX_FIELDS.includes(key)) continue;
        if (value !== '') { // Only append non-empty values
            formData.append(key, value);
        }
    }

    // An unchecked checkbox is absent from FormData, and the server now defaults
    // these to true — so omitting them would silently override the user's choice
    // with the opposite of it. Send both explicitly, always.
    formData.append('run_summarizer', String(document.getElementById('file-run-summarizer').checked));
    formData.append('generate_quiz', String(document.getElementById('file-generate-quiz').checked));

    // Ensure title is provided (it's required)
    const title = formData.get('title');
    if (!title || title.trim() === '') {
        alert('Please provide a title for the paper');
        return;
    }

    setUploadBusy(true);
    try {
        await dispatchUpload(`/api/pillars/${pillarId}/upload/pdf`, {
            method: 'POST',
            body: formData,
        });
        // The bytes are on the server and the run owns them from here, so the form
        // can be cleared: re-submitting it would start a second upload of the same
        // paper, not retry this one.
        form.reset();
        document.getElementById('file-upload-form').style.display = 'none';
        document.getElementById('file-drop-zone').style.display = 'block';
        window.selectedFile = null;
    } catch (error) {
        console.error('Upload error:', error);
        reportUploadDispatchFailure(error);
    }
}

async function submitUrlUpload(pillarId) {
    const form = document.getElementById('url-upload-form');
    const formData = new FormData(form);

    const data = {
        url: formData.get('url'),
        title: formData.get('title') || null,
        authors: formData.get('authors') ? formData.get('authors').split(',').map(a => a.trim()) : null,
        run_summarizer: formData.get('run_summarizer') === 'on',
        generate_quiz: formData.get('generate_quiz') === 'on'
    };

    setUploadBusy(true);
    try {
        await dispatchUpload(`/api/pillars/${pillarId}/upload/url`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        form.reset();
    } catch (error) {
        console.error('Upload error:', error);
        reportUploadDispatchFailure(error);
    }
}

/**
 * Report what the finished upload actually did.
 *
 * Three facts, kept apart because collapsing them is the bug this path keeps
 * relearning: whether the paper reached the library (`result.added`), which
 * post-upload steps completed, and which failed. A failed run whose paper WAS added
 * is not an instruction to upload again — saying so is the whole point of this box.
 *
 * createElement + textContent throughout. Titles come from PDFs and arXiv and the
 * error strings are raw exception text; none of it may be interpolated into innerHTML.
 */
function renderUploadOutcome(run) {
    const box = document.getElementById('upload-result');
    if (!box) return;

    const result = run.result || {};
    const errors = result.errors || [];
    const added = Boolean(result.added);
    const nodes = [];

    const heading = document.createElement('strong');
    if (added) {
        heading.textContent = `In your library: ${result.title || result.paper_id || 'the paper'}`;
    } else {
        heading.textContent = 'Nothing was added to the library.';
    }
    nodes.push(heading);

    const done = result.actions_triggered || [];
    if (done.length) {
        const line = document.createElement('p');
        line.textContent = `Completed: ${done.join(', ')}`;
        nodes.push(line);
    }

    const reasons = errors.length ? errors : (added || !run.error ? [] : [run.error]);
    if (reasons.length) {
        const label = document.createElement('p');
        label.textContent = added ? 'These steps did not finish:' : 'Why:';
        nodes.push(label);

        const list = document.createElement('ul');
        reasons.forEach((message) => {
            const item = document.createElement('li');
            item.textContent = message;
            list.appendChild(item);
        });
        nodes.push(list);
    }

    if (added && reasons.length) {
        const note = document.createElement('p');
        note.textContent =
            'The paper is in the library. Nothing needs re-uploading; the steps above '
            + 'are what did not run.';
        nodes.push(note);
    }

    box.replaceChildren(...nodes);
    // The edge carries the same distinction as the words: added, added-but-incomplete
    // (warn, never "start again"), or nothing added. Classes, so the three themes
    // paint it from tokens rather than raw colour names.
    box.classList.remove('outcome-ok', 'outcome-warn', 'outcome-danger');
    box.classList.add(!added ? 'outcome-danger' : (reasons.length ? 'outcome-warn' : 'outcome-ok'));
    box.hidden = false;
    box.scrollIntoView({ block: 'nearest' });
}

/**
 * Quiz cards are generated from the summarizer's PaperNote, so the quiz option only
 * means anything with the summarizer on. That was previously enforced by the backend
 * silently doing nothing. Here the two boxes move together, so the constraint is
 * visible before the upload rather than inferred from a missing quiz afterwards.
 * The backend still reports the combination if an API client sends it directly.
 */
function linkSummarizerAndQuiz(summarizerId, quizId) {
    const summarizer = document.getElementById(summarizerId);
    const quiz = document.getElementById(quizId);
    if (!summarizer || !quiz) return;

    quiz.addEventListener('change', () => {
        if (quiz.checked) summarizer.checked = true;
    });
    summarizer.addEventListener('change', () => {
        if (!summarizer.checked) quiz.checked = false;
    });
}

/**
 * Recent uploads, read from pipeline_runs.
 *
 * This list used to read GET /uploads/recent, which served an in-memory dict on a
 * process-local singleton: empty on every page load, empty after every restart, and
 * invisible to the scheduler container. It printed "No recent uploads" no matter how
 * many papers had been uploaded. The run rows are the real record.
 *
 * Filtered client-side by kind because /api/pipeline-runs takes pillar_id and limit
 * and nothing else; the limit is raised to compensate so a pillar's uploads are not
 * pushed out of the window by its discovery runs.
 */
async function loadRecentUploads() {
    const pillarId = currentPillarId();
    const container = document.getElementById('upload-status-container');
    if (!container) return;

    const empty = (text) => {
        const wrap = document.createElement('div');
        wrap.className = 'empty-state';
        const line = document.createElement('p');
        line.textContent = text;
        wrap.appendChild(line);
        container.replaceChildren(wrap);
    };

    try {
        const response = await fetch(
            `/api/pipeline-runs?pillar_id=${encodeURIComponent(pillarId)}&limit=50`
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const runs = (await response.json()).filter((run) => run.kind === 'upload');

        if (!runs.length) {
            empty('No recent uploads');
            return;
        }

        // createElement + textContent throughout. Titles come from PDFs and arXiv and
        // `error` is raw exception text; a filename like `<img src=x onerror=...>.pdf`
        // executed on this page when it was built from a template literal. Note that
        // escaping would not be enough for the attributes below either — it leaves
        // quote characters intact, so only setAttribute/style assignment is safe.
        const frag = document.createDocumentFragment();
        runs.slice(0, 10).forEach((run) => {
            // This endpoint returns runs WITHOUT their stages and without the whole
            // `result` payload — see PostgrestClient.list_pipeline_runs, which
            // projects just the two scalars a row needs out of it. Rebuild the shape
            // summarise()/statusColor() expect so the wording here and in the live
            // panel above cannot drift apart.
            const listed = Object.assign({}, run, {
                result: run.result_title || run.result_added
                    ? { title: run.result_title, added: run.result_added === 'true' }
                    : null,
            });
            const item = document.createElement('div');
            item.className = 'upload-status-item';

            const icon = document.createElement('div');
            icon.className = 'upload-status-icon';
            icon.textContent = getStatusIcon(run.status);

            const content = document.createElement('div');
            content.className = 'upload-status-content';

            const name = document.createElement('h5');
            // `result` is written once at the end, and only when a paper was added,
            // so a live row and a row whose upload never got that far both arrive
            // with no title. They are different states and say so; neither prints a
            // run id at the user, which is what this list used to do.
            name.textContent = run.result_title
                || (RunProgress.isTerminal(run.status) ? 'Upload' : 'Upload in progress');

            const message = document.createElement('p');
            // summarise() counts steps out of `run.stages`, which are not in this
            // response, so a live run would read "Step 0 of 0". A terminal upload's
            // summary comes from `result` instead, so only the live case needs its
            // own line here.
            message.textContent = RunProgress.isTerminal(run.status)
                ? RunProgress.summarise(listed)
                : 'In progress — ' + RunProgress.stageLabel(run.current_stage || '');
            const tone = { green: 'is-ok', darkorange: 'is-warn', crimson: 'is-danger' }[RunProgress.statusColor(listed)];
            if (tone) message.classList.add(tone);

            content.append(name, message);

            const when = document.createElement('small');
            when.style.opacity = '.7';
            when.textContent = run.created_at
                ? new Date(run.created_at).toLocaleString()
                : '';
            content.appendChild(when);

            item.append(icon, content);
            frag.appendChild(item);
        });

        container.replaceChildren(frag);
    } catch (error) {
        console.error('Failed to load recent uploads:', error);
        // Not "No recent uploads": an unreachable database and an empty list are
        // different facts, and this list showed the second for both for its whole life.
        empty('Could not load recent uploads — ' + error.message);
    }
}

//: pipeline_runs statuses, not the retired UploadStatus vocabulary.
function getStatusIcon(status) {
    switch (status) {
        case 'pending': return '⏳';
        case 'running': return '⚙️';
        case 'succeeded': return '✅';
        case 'failed': return '❌';
        case 'cancelled': return '🚫';
        case 'interrupted': return '⚠️';
        default: return '📄';
    }
}


  function init() {
    setupUploadInterface();
    loadRecentUploads();
    RunProgress.reattach(uploadProgressEls(), onUploadFinished, onUploadRun);
  }

  global.PillarUpload = { init: init, currentPillarId: currentPillarId };
})(typeof window !== 'undefined' ? window : globalThis);
