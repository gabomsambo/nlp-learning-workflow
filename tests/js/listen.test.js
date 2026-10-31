/*
 * Listen page helpers. DOM wiring is exercised in the browser.
 * Run with `node --test tests/js/*.test.js`.
 */
const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const {
  episodeInitial,
  paperPageHref,
  audioHref,
  formatDuration,
  formatCreated,
  formatValueText,
  scriptSegments,
  scriptLines,
  claimBucket,
  claimPresentation,
  resolveSpan,
  claimSegments,
  strongestClaim,
  issueClaims,
  factcheckSummary,
  isFactcheckStale,
  requestedLine,
  groupEpisodes,
  viewMode,
  rowFromParts,
} = require('../../webui/static/listen.js');

const SOURCE = fs.readFileSync(path.join(__dirname, '../../webui/static/listen.js'), 'utf8');
const TEMPLATE = fs.readFileSync(path.join(__dirname, '../../webui/templates/podcast.html'), 'utf8');

test('paper links encode a colon and audio names cannot escape the directory', () => {
  assert.equal(paperPageHref('arxiv:1706.03762'), '/papers/arxiv%3A1706.03762');
  assert.equal(paperPageHref(''), null);
  assert.equal(audioHref('episode.mp3'), '/api/podcast/audio/episode.mp3');
  assert.equal(audioHref('../secret.mp3'), null);
  assert.equal(audioHref('nested/ep.mp3'), null);
});

test('duration, dates, and the scrubber label', () => {
  assert.equal(formatDuration(125), '2:05');
  assert.equal(formatDuration(0), '');
  assert.equal(formatDuration(null), '');
  assert.equal(formatCreated('2026-10-23T22:01:00+00:00'), '23 Oct 2026');
  assert.equal(formatValueText(192, 778), '3 minutes 12 of 12 minutes 58');
});

test('script cues are segments, not an HTML string', () => {
  const parts = scriptSegments('Hello\n[HOST]: Welcome\n[PAUSE]\nTail');
  assert.deepEqual(parts.map((part) => part.kind), ['text', 'host', 'text', 'cue', 'text']);
  assert.equal(parts[1].text, '[HOST]:');
  assert.equal(parts[3].text, '[PAUSE]');
  assert.equal(parts.some((part) => part.text.includes('<')), false);
});

test('the script is split into the lines the fact-check engine numbered', () => {
  // The engine numbers claims by the index of Python's script.splitlines(),
  // and the TTS assembler walks the same list. A line off by one paints every
  // highlight on the wrong words, so these are Python's answers, not guesses.
  assert.deepEqual(scriptLines(''), []);
  assert.deepEqual(scriptLines('a'), ['a']);
  assert.deepEqual(scriptLines('a\n'), ['a']);
  assert.deepEqual(scriptLines('a\nb'), ['a', 'b']);
  assert.deepEqual(scriptLines('a\nb\n'), ['a', 'b']);
  assert.deepEqual(scriptLines('a\n\n'), ['a', '']);
  assert.deepEqual(scriptLines('\n'), ['']);
  assert.deepEqual(scriptLines('a\r\nb'), ['a', 'b']);
  assert.deepEqual(scriptLines('a\rb'), ['a', 'b']);
  assert.deepEqual(scriptLines('a\vb'), ['a', 'b']);
  assert.deepEqual(scriptLines('a\fb'), ['a', 'b']);
  assert.deepEqual(scriptLines('a\x85b'), ['a', 'b']);
  assert.deepEqual(scriptLines('a\u2028b'), ['a', 'b']);
  assert.deepEqual(scriptLines('a\u2029b'), ['a', 'b']);
});

test('a span whose offsets do not produce its own text is re-anchored', () => {
  // Python counts a line in code points; JavaScript counts UTF-16 code units.
  // The emoji is one code point and two code units, so Python's offsets for
  // "61.7" are 7..11 and JavaScript's are 8..12. The engine stores the
  // Python ones, and the highlight would land on the wrong words.
  const line = '🎧 gave 61.7 here';
  assert.equal(line.slice(7, 11), ' 61.');
  assert.equal(line.slice(8, 12), '61.7');
  assert.deepEqual(resolveSpan(line, { span_start: 7, span_end: 11, span_text: '61.7' }), {
    start: 8,
    end: 12,
  });
  assert.deepEqual(resolveSpan(line, { span_start: 8, span_end: 12, span_text: '61.7' }), {
    start: 8,
    end: 12,
  });
  // A span that is not in this line at all is not highlighted.
  assert.equal(resolveSpan(line, { span_start: 0, span_end: 4, span_text: 'nope' }), null);
  // Out-of-range offsets are not painted either.
  assert.equal(resolveSpan(line, { span_start: -1, span_end: 3 }), null);
  assert.equal(resolveSpan(line, { span_start: 5, span_end: 500 }), null);
});

test('fact-check spans split a line at adjacent and overlapping boundaries', () => {
  const high = { id: 'high', span_start: 2, span_end: 8, severity: 'high', verdict: 'contradicted' };
  const medium = { id: 'medium', span_start: 5, span_end: 10, severity: 'medium', verdict: 'partially_supported' };
  const segments = claimSegments('0123456789ab', [high, medium], false);
  assert.deepEqual(segments.map((part) => [part.text, part.claims.map((claim) => claim.id)]), [
    ['01', []],
    ['234', ['high']],
    ['567', ['high', 'medium']],
    ['89', ['medium']],
    ['ab', []],
  ]);
  assert.equal(strongestClaim(segments[2].claims).id, 'high');
});

test('supported spans are opt-in and malformed anchors are ignored', () => {
  const supported = { id: 'ok', span_start: 0, span_end: 4, verdict: 'supported' };
  const malformed = { id: 'bad', span_start: -1, span_end: 99, verdict: 'contradicted', severity: 'high' };
  assert.deepEqual(claimSegments('text', [supported, malformed], false).map((part) => part.claims), [[]]);
  assert.deepEqual(
    claimSegments('text', [supported, malformed], true)[0].claims.map((claim) => claim.id),
    ['ok'],
  );
});

test('issue order is severity first then zero-based script position', () => {
  const ordered = issueClaims([
    { id: 'low', severity: 'low', line_index: 1, span_start: 0 },
    { id: 'later', severity: 'high', line_index: 8, span_start: 0 },
    { id: 'earlier', severity: 'high', line_index: 3, span_start: 4 },
    { id: 'supported', severity: '', verdict: 'supported', line_index: 0 },
    { id: 'opinion', severity: 'none', verdict: 'not_checkable', line_index: 0 },
  ]);
  assert.deepEqual(ordered.map((claim) => claim.id), ['earlier', 'later', 'low']);
});

test('claim buckets keep the verdicts apart', () => {
  assert.equal(claimBucket({ verdict: 'contradicted', severity: 'high' }), 'contradicted');
  assert.equal(claimBucket({ verdict: 'contradicted', severity: 'medium' }), 'misattributed');
  assert.equal(claimBucket({ verdict: 'unsupported', severity: 'high' }), 'not_in_paper');
  assert.equal(claimBucket({ verdict: 'unsupported', severity: 'low' }), 'outside');
  assert.equal(
    claimBucket({ verdict: 'partially_supported', error_type: 'wrong_attribution' }),
    'misattributed',
  );
  assert.equal(claimBucket({ verdict: 'partially_supported' }), 'overstated');
  assert.equal(claimBucket({ verdict: 'not_checkable' }), 'unchecked');
  assert.equal(claimBucket({ verdict: 'supported' }), 'supported');
  // A verdict this build does not know is shown as unclassified, not dropped.
  assert.equal(claimBucket({ verdict: 'brand_new' }), 'unclassified');
});

test('claim labels and summary do not imply that unchecked text was verified', () => {
  assert.equal(claimPresentation({ verdict: 'contradicted', severity: 'high' }).label, 'Contradicts the paper');
  assert.equal(claimPresentation({ verdict: 'not_checkable' }).label, 'Opinion · not checked');
  assert.equal(
    claimPresentation({ verdict: 'unsupported', severity: 'high' }).label,
    'Not in the paper',
  );
  assert.equal(
    claimPresentation({ verdict: 'unsupported', severity: 'low' }).label,
    'Outside the paper',
  );
  const summary = factcheckSummary([
    { verdict: 'supported' },
    { verdict: 'contradicted', severity: 'high' },
    { verdict: 'unsupported', severity: 'high', scope: 'paper' },
    { verdict: 'unsupported', severity: 'low', scope: 'background' },
    { verdict: 'partially_supported', severity: 'medium' },
    { verdict: 'not_checkable' },
  ]);
  assert.equal(
    summary,
    '6 claims checked · 1 contradict the paper · 1 not in the paper · 1 overstated · 1 outside the paper · 1 opinion (not checked) · 1 supported',
  );
});

test('a summary of nothing says so, and a single claim is not pluralised', () => {
  assert.equal(factcheckSummary([]), '0 claims checked');
  assert.equal(factcheckSummary([{ verdict: 'supported' }]), '1 claim checked · 1 supported');
});

test('staleness and deep-link lines require exact hashes and valid zero-based indexes', () => {
  assert.equal(isFactcheckStale('old', 'new'), true);
  assert.equal(isFactcheckStale('same', 'same'), false);
  assert.equal(isFactcheckStale('', 'new'), false);
  assert.equal(requestedLine('0', 4), 0);
  assert.equal(requestedLine('3', 4), 3);
  assert.equal(requestedLine('4', 4), null);
  assert.equal(requestedLine('-1', 4), null);
  assert.equal(requestedLine('1.5', 4), null);
});

test('episodes group newest pillar first', () => {
  const groups = groupEpisodes([
    { id: 'a', pillar_id: 'old', pillar_name: 'Old', created_sort: '2026-01-01T00:00:00' },
    { id: 'b', pillar_id: 'new', pillar_name: 'New', created_sort: '2026-08-01T00:00:00' },
    { id: 'c', pillar_id: 'new', pillar_name: 'New', created_sort: '2026-07-01T00:00:00' },
  ]);
  assert.deepEqual(groups.map((group) => group.pillar_name), ['New', 'Old']);
  assert.deepEqual(groups[0].episodes.map((ep) => ep.id), ['b', 'c']);
});

test('a paper deep link opens the studio and a script link stays on episodes', () => {
  assert.equal(viewMode(''), 'episodes');
  assert.equal(viewMode('?script=abc'), 'episodes');
  assert.equal(viewMode('?studio=1'), 'studio');
  assert.equal(viewMode('?paper=arxiv%3A1'), 'studio');
  assert.equal(viewMode('?run=abc'), 'studio');
});

test('a list row picks up duration from the episode detail', () => {
  const row = rowFromParts(
    { id: 's1', paper_id: 'arxiv:1', pillar_id: 'neural', title: 'Spoken', word_count: 40, created_at: '2026-10-23T00:00:00+00:00' },
    { script: { audio_metadata: { file_name: 'spoken.mp3', duration_seconds: 65 } } },
    { papers: { 'arxiv:1': { title: 'Attention', pillar_id: 'neural' } }, pillars: { neural: 'Neural' } },
  );
  assert.equal(row.paper_href, '/papers/arxiv%3A1');
  assert.equal(row.paper_title, 'Attention');
  assert.equal(row.pillar_name, 'Neural');
  assert.equal(row.duration_label, '1:05');
  assert.equal(row.audio_href, '/api/podcast/audio/spoken.mp3');
  assert.equal(row.created_label, '23 Oct 2026');
  assert.equal(episodeInitial(''), 'E');
});

test('the listen script does not build HTML or start a paid call by itself', () => {
  assert.equal(SOURCE.includes('innerHTML'), false);
  assert.equal(SOURCE.includes('models-architectures'), false);
  assert.equal((SOURCE.match(/\/api\/podcast\/generate/g) || []).length, 1);
  assert.equal((SOURCE.match(/\/api\/podcast\/tts\/generate/g) || []).length, 1);
  assert.equal(SOURCE.includes('unsaved_podcast.txt'), true);
  const bootTail = SOURCE.slice(SOURCE.lastIndexOf('refreshTtsStatus();'));
  assert.equal(bootTail.includes('/api/podcast/generate'), false);
  assert.equal(bootTail.includes('/api/podcast/tts/generate'), false);
});

test('the script form shows a voice picker and explicit default-on audio consent', () => {
  assert.match(TEMPLATE, /id="request-voice-select"/);
  assert.match(TEMPLATE, /id="request-preview-voice-btn"/);
  assert.match(TEMPLATE, /id="auto-audio" checked/);
  assert.match(SOURCE, /voice_path: requestVoiceSelect \? requestVoiceSelect\.value \|\| null : null/);
  assert.match(SOURCE, /auto_audio: !!\(autoAudioCheckbox && autoAudioCheckbox\.checked\)/);
  assert.match(SOURCE, /lastVoicePath = data\.last_voice_path \|\| null/);
});
