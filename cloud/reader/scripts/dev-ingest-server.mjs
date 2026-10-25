// The real Worker (reader and /ingest) over an in-memory R2 bucket, no Cloudflare and no network:
//   node cloud/reader/scripts/dev-ingest-server.mjs [port]      (port 0 picks a free one)
// Prints {"port": N} when it is listening. Every request is given a throwaway Access token so the
// JWT checks still run: /ingest requests get the publisher's service token, everything else a
// signed-in reader. The Python publisher's tests drive it, and so can you:
//   CLOUD_PUBLISH_URL=http://127.0.0.1:<port> CF_ACCESS_CLIENT_ID=x CF_ACCESS_CLIENT_SECRET=x \
//     python -m nlp_pillars.cli cloud publish
import { createServer } from 'node:http';

import { createHandler } from '../src/worker.js';
import {
  AUD, INGEST_AUD, KID, NOW_MS, PUBLISHER, TEAM, fakeAssets, fakeStore, makeKeyPair, serviceClaims, signJwt, userClaims,
} from '../test/helpers.mjs';

const port = Number(process.argv[2] || 0);
const pair = await makeKeyPair();
const handle = createHandler({ keyFor: async (kid) => (kid === KID ? pair.jwk : null) });
const store = fakeStore();
// The fake bucket stamps writes with a fixed clock; real time is what garbage collection compares against.
const checked = store.put.bind(store);
store.put = async (key, body, options) => { await checked(key, body, options); store.objects.get(key).uploaded = new Date(); };

const env = {
  TEAM_DOMAIN: TEAM,
  ACCESS_AUD: AUD,
  ALLOWED_EMAILS: 'dev@example.com',
  INGEST_AUD,
  INGEST_SERVICE_TOKENS: PUBLISHER,
  INGEST_MAX_BYTES: process.env.INGEST_MAX_BYTES || '5000000000',
  DOCS: store,
  ASSETS: fakeAssets(),
};

const server = createServer(async (req, res) => {
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
  const response = await handle(new Request(`http://localhost:${server.address().port}${req.url}`, init), env);
  res.writeHead(response.status, Object.fromEntries(response.headers));
  res.end(Buffer.from(await response.arrayBuffer()));
});

server.listen(port, '127.0.0.1', () => console.log(JSON.stringify({ port: server.address().port, now: NOW_MS })));
