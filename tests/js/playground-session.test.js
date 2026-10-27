/*
 * The Playground's runner session (webui/static/playground-session.js),
 * driven through a fake fetch that plays the webui proxy.
 * Run with `node --test tests/js/*.test.js`.
 */
const { test } = require('node:test');
const assert = require('node:assert');

const { Session, BASE } = require('../../webui/static/playground-session.js');

function response(status, body) {
  return { status: status, json: () => Promise.resolve(body) };
}

/**
 * A fake proxy. `routes` maps "METHOD /path" to a function (init) -> [status, body]
 * or a promise of one. Every call is recorded.
 */
function fakeProxy(routes) {
  const calls = [];
  const fetch = (url, init) => {
    const key = init.method + ' ' + url.slice(BASE.length);
    calls.push({ key: key, body: init.body ? JSON.parse(init.body) : undefined, keepalive: !!init.keepalive });
    const route = routes[key];
    if (!route) return Promise.resolve(response(404, { detail: 'no route ' + key }));
    return Promise.resolve(route(init)).then(([status, body]) => response(status, body));
  };
  return { fetch, calls };
}

function okResult(extra) {
  return Object.assign({ status: 'ok', execution_ms: 5, execution_count: 1, outputs: [], truncated: false,
    dropped_outputs: 0, interrupted: false, kernel_restarted: false }, extra || {});
}

function deferred() {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  return { promise, resolve };
}

const HEALTH = () => [200, { status: 'ok', limits: { exec_timeout_s: 180, max_exec_timeout_s: 900 } }];
const CREATE = () => [201, { session_id: 's1', playground_id: 'pg' }];

test('health reads the limits; a down runner shows the start hint', async () => {
  const up = fakeProxy({ 'GET /health': HEALTH });
  const s = new Session({ playgroundId: 'pg', fetch: up.fetch });
  assert.deepStrictEqual(await s.health(), { exec_timeout_s: 180, max_exec_timeout_s: 900 });
  assert.strictEqual(s.timeoutS(), 180);
  assert.strictEqual(s.maxTimeoutS(), 900);

  const down = fakeProxy({ 'GET /health': () => [503, { detail: 'not reachable. Start it with ./workstation up' }] });
  const phases = [];
  const d = new Session({ playgroundId: 'pg', fetch: down.fetch, onChange: (x) => phases.push(x.phase) });
  assert.strictEqual(await d.health(), null);
  assert.strictEqual(d.phase, 'unavailable');
  assert.match(d.message, /workstation up/);
  assert.deepStrictEqual(phases, ['unavailable']);
});

test('start creates one session for the playground, even when asked twice', async () => {
  const p = fakeProxy({ 'POST /sessions': CREATE });
  const phases = [];
  const s = new Session({ playgroundId: 'pg-1', fetch: p.fetch, onChange: (x) => phases.push(x.phase) });
  const [a, b] = await Promise.all([s.start(), s.ensure()]);
  assert.ok(a && b);
  assert.strictEqual(s.id, 's1');
  assert.deepStrictEqual(p.calls.map((c) => c.key), ['POST /sessions']);
  assert.deepStrictEqual(p.calls[0].body, { playground_id: 'pg-1' });
  assert.deepStrictEqual(phases, ['starting', 'ready']);
});

test('a full runner (429) with nothing of ours to close is an error with the runner\'s reason', async () => {
  const p = fakeProxy({
    'POST /sessions': () => [429, { detail: 'The runner already has 3 sessions open' }],
    'GET /sessions': () => [200, { sessions: [
      { session_id: 'x1', playground_id: 'other', busy: false, last_active_at: '2026-10-27T01:00:00Z' },
      { session_id: 'x2', playground_id: 'pg', busy: true, last_active_at: '2026-10-27T01:00:00Z' },
    ] }],
  });
  const s = new Session({ playgroundId: 'pg', fetch: p.fetch });
  assert.strictEqual(await s.start(), false);
  assert.strictEqual(s.phase, 'error');
  assert.strictEqual(s.message, 'The runner already has 3 sessions open');
  // Another playground's session and a busy one are never closed.
  assert.ok(!p.calls.some((c) => c.key.startsWith('DELETE')));
});

test('a full runner closes this playground\'s oldest idle leftover and retries once', async () => {
  let creates = 0;
  const p = fakeProxy({
    'POST /sessions': () => (++creates === 1 ? [429, { detail: 'full' }] : [201, { session_id: 'fresh' }]),
    'GET /sessions': () => [200, { sessions: [
      { session_id: 'newer', playground_id: 'pg', busy: false, last_active_at: '2026-10-27T02:00:00Z' },
      { session_id: 'older', playground_id: 'pg', busy: false, last_active_at: '2026-10-27T01:00:00Z' },
      { session_id: 'theirs', playground_id: 'other', busy: false, last_active_at: '2026-10-26T01:00:00Z' },
    ] }],
    'DELETE /sessions/older': () => [200, { deleted: true }],
  });
  const s = new Session({ playgroundId: 'pg', fetch: p.fetch });
  assert.strictEqual(await s.start(), true);
  assert.strictEqual(s.id, 'fresh');
  assert.deepStrictEqual(p.calls.map((c) => c.key),
    ['POST /sessions', 'GET /sessions', 'DELETE /sessions/older', 'POST /sessions']);
});

test('execute starts a session first and returns the runner result', async () => {
  const p = fakeProxy({
    'POST /sessions': CREATE,
    'POST /sessions/s1/execute': () => [200, okResult({ execution_count: 3 })],
  });
  const s = new Session({ playgroundId: 'pg', fetch: p.fetch });
  let started = 0;
  const res = await s.execute('x = 1', 60, () => { started += 1; });
  assert.strictEqual(res.result.execution_count, 3);
  assert.strictEqual(started, 1);
  assert.deepStrictEqual(p.calls[1].body, { code: 'x = 1', timeout_s: 60 });
  await s.execute('  y = 2\n  print(y)');
  assert.strictEqual(p.calls[2].body.code, 'y = 2\nprint(y)');
  assert.strictEqual(s.phase, 'ready');
  s.markRan(4, res.generation);
  assert.ok(s.hasRan(4));
});

test('runs queue one at a time, so the runner never sees a second execute (409)', async () => {
  const gates = [deferred(), deferred()];
  let active = 0;
  let maxActive = 0;
  let n = 0;
  const p = fakeProxy({
    'POST /sessions': CREATE,
    'POST /sessions/s1/execute': () => {
      const gate = gates[n++];
      active += 1;
      maxActive = Math.max(maxActive, active);
      return gate.promise.then(() => { active -= 1; return [200, okResult()]; });
    },
  });
  const s = new Session({ playgroundId: 'pg', fetch: p.fetch });
  const first = s.execute('a');
  const second = s.execute('b');
  await new Promise((r) => setTimeout(r, 5));
  assert.strictEqual(s.phase, 'busy');
  assert.strictEqual(s.waiting, 2);
  gates[0].resolve();
  await first;
  gates[1].resolve();
  await second;
  assert.strictEqual(maxActive, 1);
  assert.deepStrictEqual(p.calls.filter((c) => c.key.endsWith('/execute')).map((c) => c.body.code), ['a', 'b']);
  assert.strictEqual(s.waiting, 0);
});

test('interrupt stops the running code and drops what is waiting', async () => {
  const gate = deferred();
  const p = fakeProxy({
    'POST /sessions': CREATE,
    'POST /sessions/s1/execute': () => gate.promise.then(() => [200, okResult({ status: 'error', interrupted: true })]),
    'POST /sessions/s1/interrupt': () => [200, { session_id: 's1', interrupted: true }],
  });
  const s = new Session({ playgroundId: 'pg', fetch: p.fetch });
  const running = s.execute('while True: pass');
  const waiting = s.execute('print(1)');
  await new Promise((r) => setTimeout(r, 5));
  assert.strictEqual(await s.interrupt(), true);
  gate.resolve();
  assert.strictEqual((await running).result.interrupted, true);
  assert.deepStrictEqual(await waiting, { cancelled: true });
  assert.strictEqual(p.calls.filter((c) => c.key.endsWith('/execute')).length, 1);
});

test('restart empties the ran set and a late result cannot mark the old kernel', async () => {
  const p = fakeProxy({
    'POST /sessions': CREATE,
    'POST /sessions/s1/execute': () => [200, okResult()],
    'POST /sessions/s1/restart': () => [200, { session_id: 's1', restarted: true }],
  });
  const s = new Session({ playgroundId: 'pg', fetch: p.fetch });
  const res = await s.execute('x = 1');
  s.markRan(2, res.generation);
  assert.ok(s.hasRan(2));
  assert.strictEqual(await s.restart(), true);
  assert.ok(!s.hasRan(2));
  s.markRan(2, res.generation); // the old generation
  assert.ok(!s.hasRan(2));
  assert.strictEqual(s.phase, 'ready');
});

test('a kernel the runner had to restart loses the ran set too', async () => {
  const p = fakeProxy({
    'POST /sessions': CREATE,
    'POST /sessions/s1/execute': (init) => [200, JSON.parse(init.body).code === 'boom'
      ? okResult({ status: 'error', kernel_restarted: true }) : okResult()],
  });
  const s = new Session({ playgroundId: 'pg', fetch: p.fetch });
  s.markRan(1, (await s.execute('x')).generation);
  await s.execute('boom');
  assert.ok(!s.hasRan(1));
});

test('an expired session (404) stops, and the next run starts a new one', async () => {
  let created = 0;
  let executes = 0;
  const p = fakeProxy({
    'POST /sessions': () => { created += 1; return [201, { session_id: 's' + created }]; },
    'POST /sessions/s1/execute': () => { executes += 1; return [404, { detail: 'Unknown session' }]; },
    'POST /sessions/s2/execute': () => [200, okResult()],
  });
  const s = new Session({ playgroundId: 'pg', fetch: p.fetch });
  const first = await s.execute('x');
  assert.strictEqual(first.status, 404);
  assert.match(first.error, /session had ended/);
  assert.strictEqual(s.phase, 'stopped');
  assert.strictEqual(s.id, null);
  const second = await s.execute('x');
  assert.ok(second.result);
  assert.strictEqual(s.id, 's2');
  assert.strictEqual(executes, 1);
});

test('a runner that is down answers the run with the start hint', async () => {
  const p = fakeProxy({ 'POST /sessions': () => [503, { detail: 'The playground runner is not reachable. Start it with ./workstation up' }] });
  const s = new Session({ playgroundId: 'pg', fetch: p.fetch });
  const res = await s.execute('x');
  assert.strictEqual(res.status, 503);
  assert.match(res.error, /workstation up/);
  assert.strictEqual(s.phase, 'unavailable');
});

test('stop deletes the session; leave does it with keepalive and without waiting', async () => {
  const p = fakeProxy({
    'POST /sessions': CREATE,
    'DELETE /sessions/s1': () => [200, { session_id: 's1', deleted: true }],
  });
  const s = new Session({ playgroundId: 'pg', fetch: p.fetch });
  await s.start();
  assert.strictEqual(await s.stop(), true);
  assert.strictEqual(s.phase, 'stopped');
  assert.strictEqual(s.id, null);

  await s.start();
  s.leave();
  assert.strictEqual(s.id, null);
  const last = p.calls[p.calls.length - 1];
  assert.strictEqual(last.key, 'DELETE /sessions/s1');
  assert.strictEqual(last.keepalive, true);
  // Nothing to close: no request.
  const before = p.calls.length;
  s.leave();
  assert.strictEqual(p.calls.length, before);
});

test('a network failure is a sentence, not an exception', async () => {
  const s = new Session({ playgroundId: 'pg', fetch: () => Promise.reject(new Error('offline')) });
  const res = await s.execute('x');
  assert.match(res.error, /Could not reach the app/);
  assert.strictEqual(s.phase, 'error');
  assert.match(s.message, /Could not reach the app/);
});

test('an exception inside a run is reported, and the queue keeps going', async () => {
  let n = 0;
  const p = fakeProxy({
    'POST /sessions': CREATE,
    'POST /sessions/s1/execute': () => { n += 1; if (n === 1) throw new Error('boom'); return [200, okResult()]; },
  });
  const s = new Session({ playgroundId: 'pg', fetch: p.fetch });
  const first = await s.execute('a');
  assert.match(first.error, /The page hit an error: boom/);
  assert.strictEqual(s.running, false);
  assert.ok((await s.execute('b')).result);
});
