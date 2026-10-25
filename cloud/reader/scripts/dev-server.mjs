// Run the real Worker locally, no Cloudflare and no network:
//   node cloud/reader/scripts/dev-server.mjs <bundle-dir> [port]
// It signs a throwaway Access token for every request, so the JWT check still runs.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

import { createHandler } from '../src/worker.js';
import { AUD, KID, TEAM, makeKeyPair, signJwt, userClaims } from '../test/helpers.mjs';

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

const env = {
  TEAM_DOMAIN: TEAM,
  ACCESS_AUD: AUD,
  ALLOWED_EMAILS: 'dev@example.com',
  DOCS: {
    async head(key) {
      const body = await readSafe(bundle, key);
      return body ? { key, size: body.length, httpEtag: `"${body.length}-${key.length}"` } : null;
    },
    async get(key, options = {}) {
      const body = await readSafe(bundle, key);
      if (!body) return null;
      const httpEtag = `"${body.length}-${key.length}"`;
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
  const token = await signJwt(pair.privateKey, userClaims({ email: 'dev@example.com', exp: Math.floor(Date.now() / 1000) + 3600 }));
  const request = new Request(`http://localhost:${port}${req.url}`, { method: req.method, headers: { ...req.headers, 'cf-access-jwt-assertion': token } });
  const response = await handle(request, env);
  res.writeHead(response.status, Object.fromEntries(response.headers));
  res.end(Buffer.from(await response.arrayBuffer()));
}).listen(port, () => console.log(`reader dev server: http://localhost:${port}/`));
