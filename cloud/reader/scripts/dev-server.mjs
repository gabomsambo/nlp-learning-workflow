// Run the real Worker locally, no Cloudflare and no network:
//   node cloud/reader/scripts/dev-server.mjs <bundle-dir> [port]
// It signs a throwaway Access token for every request, so the JWT check still runs: a reader's
// for everything but /ingest, the publisher's service token for /ingest. Review events are kept
// in SQLite (the real migration file) when this Node has node:sqlite, else in memory, and are
// lost when the server stops. To pull them home with the real command:
//   CLOUD_PUBLISH_URL=http://127.0.0.1:8787 CF_ACCESS_CLIENT_ID=x CF_ACCESS_CLIENT_SECRET=x \
//     python -m nlp_pillars.cli cloud pull-reviews
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

import { createHandler } from '../src/worker.js';
import {
  AUD, HAS_SQLITE, INGEST_AUD, KID, PUBLISHER, TEAM, fakeD1, makeKeyPair, serviceClaims, signJwt, sqliteD1, userClaims,
} from '../test/helpers.mjs';

const bundle = process.argv[2];
const port = Number(process.argv[3] || 8787);
if (!bundle) {
  console.error('usage: node dev-server.mjs <bundle-dir> [port]');
  process.exit(2);
}
const publicDir = fileURLToPath(new URL('../public', import.meta.url));
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2', '.png': 'image/png', '.json': 'application/json' };

const pair = await makeKeyPair();
const handle = createHandler({ keyFor: async (kid) => (kid === KID ? pair.jwk : null) });

async function readSafe(root, relative) {
  const path = normalize(join(root, relative));
  if (!path.startsWith(root)) return null;
  try { return await readFile(path); } catch { return null; }
}

const migration = readFileSync(new URL('../migrations/0001_review_events.sql', import.meta.url), 'utf8');

const env = {
  TEAM_DOMAIN: TEAM,
  ACCESS_AUD: AUD,
  ALLOWED_EMAILS: 'dev@example.com',
  INGEST_AUD,
  INGEST_SERVICE_TOKENS: PUBLISHER,
  REVIEWS: HAS_SQLITE ? sqliteD1(migration) : fakeD1(),
  DOCS: {
    async head(key) {
      const body = await readSafe(bundle, key);
      return body ? { key, size: body.length, httpEtag: `"${body.length}-${key.length}"` } : null;
    },
    async get(key, options = {}) {
      const body = await readSafe(bundle, key);
      if (!body) return null;
      const httpEtag = `"${createHash('md5').update(body).digest('hex')}"`;
      const base = { key, size: body.length, httpEtag };
      if (options.onlyIf?.get('if-none-match') === httpEtag) return base;
      const range = options.range;
      const slice = range ? body.subarray(range.offset, range.offset + range.length) : body;
      return { ...base, body: new Blob([slice]).stream(), range };
    },
  },
  ASSETS: {
    async fetch(request) {
      const path = new URL(request.url).pathname;
      const body = await readSafe(publicDir, path === '/' ? 'index.html' : path);
      if (!body) return new Response('not found', { status: 404 });
      return new Response(body, { headers: { 'content-type': TYPES[extname(path === '/' ? 'index.html' : path)] || 'application/octet-stream' } });
    },
  },
};

createServer(async (req, res) => {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const body = Buffer.concat(chunks);
  const exp = Math.floor(Date.now() / 1000) + 3600;
  const ingest = req.url.startsWith('/ingest');
  const token = ingest
    ? await signJwt(pair.privateKey, serviceClaims({ aud: [INGEST_AUD], common_name: PUBLISHER, exp }))
    : await signJwt(pair.privateKey, userClaims({ email: 'dev@example.com', exp }));
  const headers = { ...req.headers, 'cf-access-jwt-assertion': token };
  delete headers.host;
  const init = { method: req.method, headers };
  if (body.length && req.method !== 'GET' && req.method !== 'HEAD') init.body = body;
  // The address the browser used, so the Worker's same-origin check sees the page's own origin.
  const response = await handle(new Request(`http://${req.headers.host || `localhost:${port}`}${req.url}`, init), env);
  res.writeHead(response.status, Object.fromEntries(response.headers));
  res.end(Buffer.from(await response.arrayBuffer()));
}).listen(port, '127.0.0.1', () => console.log(`reader dev server: http://localhost:${port}/`));
