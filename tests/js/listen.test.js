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
  groupEpisodes,
  viewMode,
  rowFromParts,
} = require('../../webui/static/listen.js');

const SOURCE = fs.readFileSync(path.join(__dirname, '../../webui/static/listen.js'), 'utf8');

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
