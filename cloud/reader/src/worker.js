/*
 * nlp-reader: a read-only reader for the published bundle.
 *
 * Every request, static files included, passes the Access check here first
 * (`run_worker_first` in wrangler.toml). The Worker has an R2 binding and the
 * static-asset binding and nothing else: no model, no queue, no database, and
 * no request to anything but the team's Access key endpoint (certs.js).
 */

import { AccessError, authorize, verifyAccessJwt } from './access.js';
import { createKeyStore } from './certs.js';
import { serveDoc } from './docs.js';

const SECURITY_HEADERS = {
  'content-security-policy': [
    "default-src 'none'",
    "script-src 'self'",
    "style-src 'self'",
    "img-src 'self' data:",
    "font-src 'self'",
    "connect-src 'self'",
    "manifest-src 'self'",
    "base-uri 'none'",
    "form-action 'none'",
    "frame-ancestors 'none'",
  ].join('; '),
  'x-content-type-options': 'nosniff',
  'referrer-policy': 'no-referrer',
  'x-frame-options': 'DENY',
};

function reply(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
  });
}

function secured(response) {
  const out = new Response(response.body, response);
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) out.headers.set(name, value);
  // Never let a shared cache keep a response that was behind a login.
  if (!/^(private|no-store)/.test(out.headers.get('cache-control') || '')) out.headers.set('cache-control', 'private, no-cache');
  return out;
}

const stores = new Map();

function keyStoreFor(teamDomain) {
  if (!stores.has(teamDomain)) stores.set(teamDomain, createKeyStore(teamDomain));
  return stores.get(teamDomain);
}

/** Build a fetch handler. Tests inject `keyFor` and `now`; production uses the defaults. */
export function createHandler({ keyFor, now } = {}) {
  return async function handle(request, env) {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return secured(reply(405, { error: 'read only' }));
    }

    let identity;
    try {
      const store = keyFor ? null : keyStoreFor(env.TEAM_DOMAIN);
      identity = authorize(
        await verifyAccessJwt(request.headers.get('cf-access-jwt-assertion'), {
          teamDomain: env.TEAM_DOMAIN,
          audience: env.ACCESS_AUD,
          keyFor: keyFor || ((kid) => store.keyFor(kid)),
          now: now ? now() : Date.now(),
        }),
        env,
      );
    } catch (error) {
      const code = error instanceof AccessError ? error.code : 'verify-failed';
      console.warn(`access denied: ${code}`);
      if (code === 'misconfigured') return secured(reply(503, { error: 'reader is not configured' }));
      // The key endpoint was unreachable or answered nonsense: not the visitor's fault, and still closed.
      if (code === 'verify-failed') return secured(reply(503, { error: 'sign-in could not be checked' }));
      const forbidden = code === 'not-allowed';
      return secured(reply(forbidden ? 403 : 401, { error: forbidden ? 'forbidden' : 'sign in required' }));
    }

    const { pathname } = new URL(request.url);
    if (pathname === '/api/me') return secured(reply(200, { kind: identity.kind, email: identity.email || null }));
    if (pathname === '/api/manifest') return secured(await serveDoc(env.DOCS, 'manifest.json', request));
    if (pathname.startsWith('/api/doc/')) {
      let key;
      try {
        key = decodeURIComponent(pathname.slice('/api/doc/'.length));
      } catch (error) {
        return secured(reply(404, { error: 'not found' }));
      }
      return secured(await serveDoc(env.DOCS, key, request));
    }
    if (pathname.startsWith('/api/')) return secured(reply(404, { error: 'not found' }));

    return secured(await env.ASSETS.fetch(request));
  };
}

const handler = createHandler();

export default {
  fetch: (request, env) => handler(request, env),
};
