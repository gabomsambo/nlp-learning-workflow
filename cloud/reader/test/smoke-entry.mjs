// Entry for `wrangler dev` smoke runs only (never deployed). It is the real Worker
// with the key lookup swapped for a throwaway key, and it signs a token for every
// request, so the real R2 and assets bindings are exercised without a login.
import { createHandler } from '../src/worker.js';
import { KID, makeKeyPair, signJwt, userClaims } from './helpers.mjs';

let ready;
async function setup() {
  const pair = await makeKeyPair();
  return { pair, handle: createHandler({ keyFor: async (kid) => (kid === KID ? pair.jwk : null) }) };
}

export default {
  async fetch(request, env) {
    ready ??= setup();
    const { pair, handle } = await ready;
    const headers = new Headers(request.headers);
    if (!headers.has('x-smoke-anonymous')) {
      const claims = userClaims({ email: 'dev@example.com', exp: Math.floor(Date.now() / 1000) + 600 });
      headers.set('cf-access-jwt-assertion', await signJwt(pair.privateKey, claims));
    }
    return handle(new Request(request, { headers }), env);
  },
};
