/*
 * The one player: a controller around a single <audio>, driven by the bar and
 * the Listen page alike. The fake element below behaves the way the browser's
 * does where it matters here (a new src empties the element and resets its
 * position; metadata arrives later). Run with `node --test tests/js/*.test.js`.
 */
const { test } = require('node:test');
const assert = require('node:assert');

const PC = require('../../webui/static/player-core.js');
const RA = require('../../webui/static/readalong.js');

const EP_A = { script_id: 'aaaaaaaa-0000', src: '/api/podcast/audio/a.mp3', title: 'Episode A' };
const EP_B = { script_id: 'bbbbbbbb-0000', src: '/api/podcast/audio/b.mp3', title: 'Episode B' };

function fakeAudio() {
  const listeners = {};
  const el = {
    currentSrc: '',
    readyState: 0,
    duration: NaN,
    currentTime: 0,
    paused: true,
    playbackRate: 1,
    defaultPlaybackRate: 1,
    preload: '',
    srcAssignments: 0,
    playCalls: 0,
    playRefused: false,
    addEventListener(name, fn) { (listeners[name] = listeners[name] || []).push(fn); },
    emit(name) { (listeners[name] || []).forEach((fn) => fn({ type: name })); },
    play() {
      this.playCalls += 1;
      if (this.playRefused) return Promise.reject(new Error('NotAllowedError'));
      this.paused = false;
      this.emit('play');
      return Promise.resolve();
    },
    pause() {
      if (this.paused) return;
      this.paused = true;
      this.emit('pause');
    },
    // The browser's side of "the file's metadata arrived".
    metadata(duration) {
      this.readyState = 1;
      this.duration = duration;
      this.playbackRate = this.defaultPlaybackRate;
      this.emit('loadedmetadata');
    },
  };
  let src = '';
  Object.defineProperty(el, 'src', {
    get() { return src; },
    set(value) {
      src = value;
      el.srcAssignments += 1;
      el.currentSrc = 'http://localhost:8000' + value;
      el.readyState = 0;
      el.duration = NaN;
      el.currentTime = 0;
      el.playbackRate = el.defaultPlaybackRate;
      el.emit('emptied');
    },
  });
  return el;
}

function memoryStorage(initial) {
  const data = Object.assign({}, initial || {});
  return {
    data,
    getItem: (key) => (key in data ? data[key] : null),
    setItem: (key, value) => { data[key] = String(value); },
    removeItem: (key) => { delete data[key]; },
  };
}

function setup(over) {
  const audio = fakeAudio();
  const storage = memoryStorage();
  let clock = 1000000;
  const player = PC.createPlayer(Object.assign({
    audio, storage, base: 'http://localhost:8000/podcast/', now: () => clock,
  }, over || {}));
  return { audio, storage, player, tick: (ms) => { clock += ms; } };
}

test('two controls, one source: whatever the bar does is what the page shows', () => {
  const { audio, player } = setup();
  const bar = [];
  const page = [];
  player.subscribe((snap) => bar.push(snap));
  player.subscribe((snap) => page.push(snap));

  player.load(EP_A, { play: true });
  audio.metadata(600);
  assert.strictEqual(audio.srcAssignments, 1, 'one stream was opened');
  assert.strictEqual(audio.paused, false);

  // The page's pause button and the bar's play button call the same method.
  player.toggle();
  assert.strictEqual(audio.paused, true);
  assert.strictEqual(bar[bar.length - 1].paused, true);
  assert.strictEqual(page[page.length - 1].paused, true);
  player.toggle();
  assert.strictEqual(audio.paused, false);
  assert.strictEqual(bar[bar.length - 1].playing, true);
  assert.strictEqual(page[page.length - 1].playing, true);
  assert.strictEqual(audio.srcAssignments, 1, 'toggling never opens a second stream');
});

test('seeking from either control moves the one position', () => {
  const { audio, player } = setup();
  player.load(EP_A, { play: true });
  audio.metadata(600);

  player.seekTo(100);
  assert.strictEqual(audio.currentTime, 100);
  assert.strictEqual(player.snapshot().time, 100);

  player.seekBy(30);
  assert.strictEqual(audio.currentTime, 130);
  player.seekBy(-15);
  assert.strictEqual(player.snapshot().time, 115);

  player.seekBy(-1000);
  assert.strictEqual(player.snapshot().time, 0);
  player.seekTo(99999);
  assert.strictEqual(player.snapshot().time, 600, 'clamped to the episode length');
});

test('a seek before the metadata is in is kept and applied when it arrives', () => {
  const { audio, player } = setup();
  player.load(EP_A, { at: 95, play: false });
  assert.strictEqual(player.snapshot().ready, false);
  assert.strictEqual(player.snapshot().time, 95, 'the place is known before the file is');

  player.seekBy(30);
  assert.strictEqual(player.snapshot().time, 125);

  audio.metadata(600);
  assert.strictEqual(audio.currentTime, 125);
  assert.strictEqual(player.snapshot().ready, true);
  assert.strictEqual(player.snapshot().duration, 600);
});

test('speed set anywhere reaches the audio, every view, and the next episode', () => {
  const { audio, storage, player } = setup();
  const seen = [];
  player.subscribe((snap) => seen.push(snap.rate));
  player.load(EP_A, { play: true });
  audio.metadata(600);

  player.setRate(1.5);
  assert.strictEqual(audio.playbackRate, 1.5);
  assert.strictEqual(audio.defaultPlaybackRate, 1.5);
  assert.strictEqual(seen[seen.length - 1], 1.5);
  assert.strictEqual(JSON.parse(storage.getItem(PC.PLAYER_KEY)).rate, 1.5);

  player.setRate(1.37);
  assert.strictEqual(player.snapshot().rate, 1, 'a speed the selects do not offer falls back to 1x');
  player.setRate(1.25);

  player.load(EP_B, { play: true });
  audio.metadata(300);
  assert.strictEqual(audio.playbackRate, 1.25, 'the new episode keeps the chosen speed');
});

test('loading another episode replaces the track; the same episode keeps its place', () => {
  const { audio, player } = setup();
  player.load(EP_A, { play: true });
  audio.metadata(600);
  player.seekTo(200);

  // Pressing play on the episode already loaded (the page's Play at a line) does not reload it.
  player.load(EP_A, { at: 300, play: true });
  assert.strictEqual(audio.srcAssignments, 1);
  assert.strictEqual(audio.currentTime, 300);

  player.load(EP_B, { play: true });
  assert.strictEqual(audio.srcAssignments, 2);
  assert.strictEqual(player.snapshot().track.script_id, EP_B.script_id);
  assert.strictEqual(player.snapshot().time, 0);
});

test('only audio paths from the app are accepted', () => {
  const { audio, player } = setup();
  assert.strictEqual(player.load({ src: 'https://elsewhere.example/x.mp3', title: 'x' }, { play: true }), null);
  assert.strictEqual(player.load({ src: '/api/podcast/audio/../secret.mp3' }, { play: true }), null);
  assert.strictEqual(audio.srcAssignments, 0);
  assert.strictEqual(player.snapshot().track, null);
});

test('a view that throws does not stop the others', () => {
  const { audio, player } = setup();
  let reached = 0;
  player.subscribe(() => { throw new Error('broken view'); });
  player.subscribe(() => { reached += 1; });
  player.load(EP_A, { play: true });
  audio.metadata(60);
  assert.ok(reached >= 2);
});

test('the place survives a page load: restored paused at the exact time', () => {
  const first = setup();
  first.player.load(EP_A, { at: 0, play: true });
  first.audio.metadata(600);
  first.player.seekTo(137.5);
  first.player.pause();
  first.player.suspend(true);

  const next = setup({ storage: first.storage });
  const snap = next.player.restore();
  assert.strictEqual(snap.track.script_id, EP_A.script_id);
  assert.strictEqual(snap.time, 137.5);
  assert.strictEqual(snap.paused, true);
  assert.strictEqual(next.audio.playCalls, 0, 'a paused page does not start sound by itself');
  next.audio.metadata(600);
  assert.strictEqual(next.audio.currentTime, 137.5);
});

test('a page left while playing asks the next one to carry on, for a few seconds only', () => {
  const first = setup();
  first.player.load(EP_A, { play: true });
  first.audio.metadata(600);
  first.player.seekTo(42);
  first.player.suspend(true);
  const stamp = JSON.parse(first.storage.getItem(PC.PLAYER_KEY)).resume_until;
  assert.strictEqual(stamp, 1000000 + PC.RESUME_WINDOW_MS);

  const soon = setup({ storage: first.storage, now: () => 1000000 + 2000 });
  soon.player.restore();
  assert.strictEqual(soon.audio.playCalls, 1);
  soon.audio.metadata(600);
  assert.strictEqual(soon.audio.currentTime, 42, 'it carries on from where it was, not from the start');

  const late = setup({ storage: first.storage, now: () => 1000000 + PC.RESUME_WINDOW_MS + 1 });
  late.player.restore();
  assert.strictEqual(late.audio.playCalls, 0, 'an old stamp never starts sound');
});

test('a browser that refuses to autoplay leaves the bar paused at the saved place', async () => {
  const first = setup();
  first.player.load(EP_A, { play: true });
  first.audio.metadata(600);
  first.player.seekTo(88);
  first.player.suspend(true);

  const next = setup({ storage: first.storage, now: () => 1000000 + 500 });
  next.audio.playRefused = true;
  const snap = next.player.restore();
  await Promise.resolve();
  assert.strictEqual(snap.paused, true);
  assert.strictEqual(snap.time, 88);
});

test('pausing before leaving leaves no resume stamp', () => {
  const { audio, storage, player } = setup();
  player.load(EP_A, { play: true });
  audio.metadata(600);
  player.pause();
  player.suspend(true);
  assert.strictEqual(JSON.parse(storage.getItem(PC.PLAYER_KEY)).resume_until, undefined);
});

test('the saved state is throttled while playing and exact on pause and seek', () => {
  const { audio, storage, player, tick } = setup();
  player.load(EP_A, { play: true });
  audio.metadata(600);
  audio.currentTime = 10;
  audio.emit('timeupdate');
  const afterFirst = JSON.parse(storage.getItem(PC.PLAYER_KEY)).time;
  audio.currentTime = 10.4;
  tick(300);
  audio.emit('timeupdate');
  assert.strictEqual(JSON.parse(storage.getItem(PC.PLAYER_KEY)).time, afterFirst, 'within a second: not rewritten');
  tick(1200);
  audio.currentTime = 12;
  audio.emit('timeupdate');
  assert.strictEqual(JSON.parse(storage.getItem(PC.PLAYER_KEY)).time, 12);
  player.seekTo(300);
  assert.strictEqual(JSON.parse(storage.getItem(PC.PLAYER_KEY)).time, 300);
});

test('opening an episode: the player lands on it only when it is the one it holds', () => {
  const { audio, player } = setup();
  player.load(EP_A, { at: 95, play: true });
  audio.metadata(600);
  audio.currentTime = 95;

  const here = PC.landingFor(player.snapshot(), { script_id: EP_A.script_id, src: EP_A.src });
  assert.deepStrictEqual(here, { attached: true, time: 95, playing: true });

  const other = PC.landingFor(player.snapshot(), { script_id: EP_B.script_id, src: EP_B.src });
  assert.deepStrictEqual(other, { attached: false, time: 0, playing: false });

  // State saved before it carried a script id still matches by its audio path.
  const legacy = PC.landingFor({ track: { script_id: '', src: EP_A.src }, time: 7, playing: false },
    { script_id: EP_A.script_id, src: EP_A.src });
  assert.strictEqual(legacy.attached, true);
  assert.strictEqual(legacy.time, 7);

  assert.strictEqual(PC.landingFor(null, { src: EP_A.src }).attached, false);
  assert.strictEqual(PC.landingFor({ track: null }, { src: EP_A.src }).attached, false);
  assert.strictEqual(player.isCurrent({ src: EP_A.src }), true);
  assert.strictEqual(player.isCurrent({ src: EP_B.src }), false);
});

test('the place being opened maps to the line and word being spoken', () => {
  const script = [
    '[HOST]: Welcome back to the show.',
    '[HOST]: It scored three points.',
  ];
  const model = RA.buildModel({
    version: 1,
    script_sha256: 'abc',
    duration_ms: 9000,
    lines: [
      { line: 0, start_ms: 1000, end_ms: 3000, words: [[1000, 1400, 1], [1400, 1900, 1], [1900, 2100, 1], [2100, 2300, 1], [2300, 3000, 1]] },
      { line: 1, start_ms: 4000, end_ms: 8000, words: [[4000, 4500, 1], [4500, 5000, 1], [5000, 5600, 1], [5600, 8000, 1]] },
    ],
  }, script, 'abc');

  const { audio, player } = setup();
  player.load(EP_A, { at: 5.2, play: false });
  const landing = PC.landingFor(player.snapshot(), { script_id: EP_A.script_id, src: EP_A.src });
  assert.strictEqual(landing.attached, true);
  // Before any metadata: the saved place alone is enough to open on the right word.
  assert.deepStrictEqual(RA.positionAt(model, landing.time * 1000), { line: 1, word: 2 });
  audio.metadata(9);
  assert.deepStrictEqual(RA.positionAt(model, player.snapshot().time * 1000), { line: 1, word: 2 });
});

test('the bar links to the episode in read-along, never to the general list when it knows the id', () => {
  assert.strictEqual(
    PC.readAlongHref('b4269725-8ef4-4509-943d-4d576dedd3ae', '/podcast/'),
    '/podcast/?script=b4269725-8ef4-4509-943d-4d576dedd3ae&view=readalong',
  );
  assert.strictEqual(PC.readAlongHref('', '/podcast/?script=x'), '/podcast/?script=x');
  assert.strictEqual(PC.readAlongHref('../evil', 'javascript:alert(1)'), '/podcast/');
  assert.strictEqual(PC.readAlongHref('a b', undefined), '/podcast/');
});
