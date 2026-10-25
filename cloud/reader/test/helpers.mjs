// Test fixtures: a throwaway RSA key, a JWT signer, and fakes for R2 and the assets binding.
import { createHash } from 'node:crypto';

export const TEAM = 'soft-wave-31cb.cloudflareaccess.com';
export const AUD = 'a'.repeat(64);
export const KID = 'test-key-1';
export const NOW_MS = Date.UTC(2026, 9, 25, 12, 0, 0);
export const NOW_S = Math.floor(NOW_MS / 1000);

const SIGN = { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' };

export async function makeKeyPair() {
  const pair = await crypto.subtle.generateKey(
    { ...SIGN, modulusLength: 2048, publicExponent: new Uint8Array([1, 0, 1]) },
    true,
    ['sign', 'verify'],
  );
  const jwk = await crypto.subtle.exportKey('jwk', pair.publicKey);
  return { privateKey: pair.privateKey, jwk: { ...jwk, kid: KID, alg: 'RS256', use: 'sig' } };
}

function b64url(bytes) {
  return Buffer.from(bytes).toString('base64url');
}

const text = (value) => new TextEncoder().encode(typeof value === 'string' ? value : JSON.stringify(value));

export async function signJwt(privateKey, claims, header = {}) {
  const head = { alg: 'RS256', kid: KID, typ: 'JWT', ...header };
  const signingInput = `${b64url(text(head))}.${b64url(text(claims))}`;
  const signature = await crypto.subtle.sign(SIGN, privateKey, text(signingInput));
  return `${signingInput}.${b64url(new Uint8Array(signature))}`;
}

export function userClaims(overrides = {}) {
  return {
    aud: [AUD],
    iss: `https://${TEAM}`,
    email: 'Reader@Example.com',
    exp: NOW_S + 3600,
    iat: NOW_S - 10,
    nbf: NOW_S - 10,
    type: 'app',
    ...overrides,
  };
}

export function serviceClaims(overrides = {}) {
  const claims = userClaims({ common_name: 'abc123.access', ...overrides });
  delete claims.email;
  return { ...claims, ...overrides };
}

/** Just enough of an R2 bucket: get() with an If-None-Match precondition. */
export function fakeBucket(files) {
  const etagOf = (body) => `"${createHash('md5').update(body).digest('hex')}"`;
  return {
    async get(key, options = {}) {
      if (!Object.hasOwn(files, key)) return null;
      const body = files[key];
      const httpEtag = etagOf(body);
      const base = { key, size: Buffer.byteLength(body), httpEtag };
      const sent = options.onlyIf && options.onlyIf.get ? options.onlyIf.get('if-none-match') : null;
      if (sent && sent === httpEtag) return base;
      return { ...base, body: new Blob([body]).stream() };
    },
  };
}

export function fakeAssets(files = { '/index.html': '<!doctype html><title>reader</title>' }) {
  return {
    async fetch(request) {
      const path = new URL(request.url).pathname;
      const hit = files[path === '/' ? '/index.html' : path];
      if (hit === undefined) return new Response('not found', { status: 404 });
      return new Response(hit, { headers: { 'content-type': 'text/html', 'cache-control': 'public, max-age=0' } });
    },
  };
}

export function makeEnv(overrides = {}) {
  return {
    TEAM_DOMAIN: TEAM,
    ACCESS_AUD: AUD,
    ALLOWED_EMAILS: 'reader@example.com, other@example.com',
    DOCS: fakeBucket({ 'manifest.json': '{"version":1}\n', 'library.json': '{"papers":[]}\n' }),
    ASSETS: fakeAssets(),
    ...overrides,
  };
}

export const INGEST_AUD = 'c'.repeat(64);
export const PUBLISHER = 'publisher123.access';

/** A writable R2 bucket: get, put, list (with custom metadata), delete. */
export function fakeStore(initial = {}) {
  const objects = new Map();
  const put = (key, body, options = {}, uploaded = Date.UTC(2026, 9, 25, 12, 0, 0)) => {
    const bytes = typeof body === 'string' ? new TextEncoder().encode(body) : new Uint8Array(body);
    objects.set(key, { bytes, customMetadata: options.customMetadata || {}, uploaded: new Date(uploaded) });
  };
  for (const [key, value] of Object.entries(initial)) put(key, value.body, { customMetadata: value.customMetadata }, value.uploaded);
  return {
    objects,
    seed: put,
    async put(key, body, options = {}) {
      // R2 checks the checksum it is given and refuses a body that does not match.
      if (options.sha256 && createHash('sha256').update(Buffer.from(body)).digest('hex') !== options.sha256) {
        throw new Error('put: The SHA-256 checksum you specified did not match what we received. (10037)');
      }
      put(key, body, options, NOW_MS);
    },
    async get(key) {
      const held = objects.get(key);
      if (!held) return null;
      return { key, size: held.bytes.length, httpEtag: `"${key}"`, body: new Blob([held.bytes]).stream() };
    },
    async list({ cursor, limit = 1000 } = {}) {
      const keys = Array.from(objects.keys()).sort();
      const start = cursor ? Number(cursor) : 0;
      const page = keys.slice(start, start + limit);
      return {
        objects: page.map((key) => ({
          key, size: objects.get(key).bytes.length, uploaded: objects.get(key).uploaded,
          customMetadata: objects.get(key).customMetadata,
        })),
        truncated: start + limit < keys.length,
        cursor: String(start + limit),
      };
    },
    async delete(keys) { for (const key of [].concat(keys)) objects.delete(key); },
  };
}

export function makeIngestEnv(overrides = {}) {
  return makeEnv({
    INGEST_AUD,
    INGEST_SERVICE_TOKENS: PUBLISHER,
    DOCS: fakeStore(),
    ...overrides,
  });
}

/**
 * A D1 stand-in for the review_events statements in src/reviews.js, so the tests run on any
 * Node. It reproduces the behaviour that matters (INSERT OR IGNORE on the unique event_id,
 * seq that only grows) and refuses a statement it does not know, so a changed query fails
 * loudly instead of silently matching nothing. `sqliteD1` below runs the same statements on
 * real SQLite where Node has it.
 */
export function fakeD1() {
  const rows = [];
  let seq = 0;
  const FULL = ['seq', 'event_id', 'card_id', 'rating', 'reviewed_at', 'device_id', 'response_time_ms', 'session_id', 'received_at'];
  const SHORT = FULL.slice(0, 6);
  const pick = (row, columns) => Object.fromEntries(columns.map((c) => [c, row[c]]));

  function execute(sql, args) {
    if (sql.startsWith('INSERT OR IGNORE INTO review_events')) {
      const [event_id, card_id, rating, reviewed_at, device_id, response_time_ms, session_id, received_at] = args;
      if (rows.some((r) => r.event_id === event_id)) return { results: [], meta: { changes: 0 } };
      seq += 1;
      rows.push({ seq, event_id, card_id, rating, reviewed_at, device_id, response_time_ms, session_id, received_at });
      return { results: [], meta: { changes: 1 } };
    }
    if (sql.includes('FROM review_events WHERE event_id IN')) {
      return { results: rows.filter((r) => args.includes(r.event_id)).map((r) => pick(r, SHORT)), meta: { changes: 0 } };
    }
    if (sql.startsWith('SELECT COALESCE(MAX(seq), 0) AS seq FROM review_events')) {
      return { results: [{ seq: rows.reduce((m, r) => Math.max(m, r.seq), 0) }], meta: { changes: 0 } };
    }
    if (sql.includes('FROM review_events WHERE seq > ? ORDER BY seq LIMIT ?')) {
      const columns = sql.includes('received_at') ? FULL : SHORT;
      const [after, limit] = args;
      return { results: rows.filter((r) => r.seq > after).slice(0, limit).map((r) => pick(r, columns)), meta: { changes: 0 } };
    }
    throw new Error(`fakeD1 does not know this statement: ${sql}`);
  }

  return {
    rows,
    prepare(sql) {
      let args = [];
      const stmt = {
        bind(...values) { args = values; return stmt; },
        async all() { return execute(sql, args); },
        async first() { const out = execute(sql, args); return out.results[0] ?? null; },
        async run() { return execute(sql, args); },
        _execute: () => execute(sql, args),
      };
      return stmt;
    },
    // Atomic like D1's: all statements run or none (the fake's statements cannot fail midway).
    async batch(statements) { return statements.map((s) => s._execute()); },
  };
}

let DatabaseSync = null;
try {
  ({ DatabaseSync } = await import('node:sqlite'));
} catch (error) {
  DatabaseSync = null;
}
export const HAS_SQLITE = DatabaseSync !== null;

/** The same interface over real SQLite (Node 22+), running the migration file as written. */
export function sqliteD1(migrationSql) {
  const db = new DatabaseSync(':memory:');
  db.exec(migrationSql);
  const run = (sql, args) => {
    const statement = db.prepare(sql);
    if (/^\s*(INSERT|UPDATE|DELETE)/i.test(sql)) {
      const out = statement.run(...args);
      return { results: [], meta: { changes: Number(out.changes) } };
    }
    return { results: statement.all(...args).map((row) => ({ ...row })), meta: { changes: 0 } };
  };
  return {
    db,
    prepare(sql) {
      let args = [];
      const stmt = {
        bind(...values) { args = values; return stmt; },
        async all() { return run(sql, args); },
        async first() { return run(sql, args).results[0] ?? null; },
        async run() { return run(sql, args); },
        _execute: () => run(sql, args),
      };
      return stmt;
    },
    async batch(statements) {
      db.exec('BEGIN');
      try {
        const out = statements.map((s) => s._execute());
        db.exec('COMMIT');
        return out;
      } catch (error) {
        db.exec('ROLLBACK');
        throw error;
      }
    },
  };
}
