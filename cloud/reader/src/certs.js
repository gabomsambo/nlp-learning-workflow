/*
 * The signing keys Cloudflare Access uses for this team, fetched from the
 * team's own certs endpoint. This is the only network request the Worker
 * makes: the host is checked against the team-domain shape and the path is
 * fixed, so nothing a caller sends can steer it.
 */

const TEAM_DOMAIN = /^[a-z0-9][a-z0-9-]*\.cloudflareaccess\.com$/;
const KEY_TTL_MS = 60 * 60 * 1000;
// A token naming a key we do not have forces one refetch, never more than this often.
const REFETCH_FLOOR_MS = 60 * 1000;

export function certsUrl(teamDomain) {
  if (typeof teamDomain !== 'string' || !TEAM_DOMAIN.test(teamDomain)) {
    throw new Error('TEAM_DOMAIN must look like <team>.cloudflareaccess.com');
  }
  return `https://${teamDomain}/cdn-cgi/access/certs`;
}

/**
 * Returns {keysFor(kid, now)}: the JWK for a key id, from memory when fresh.
 * `fetchImpl` is injectable so tests never touch the network.
 */
export function createKeyStore(teamDomain, fetchImpl = fetch) {
  const url = certsUrl(teamDomain);
  let keys = null;
  let fetchedAt = 0;
  let inflight = null;

  async function refresh(now) {
    if (!inflight) {
      inflight = (async () => {
        const response = await fetchImpl(url, { headers: { accept: 'application/json' } });
        if (!response.ok) throw new Error(`certs endpoint answered ${response.status}`);
        const body = await response.json();
        if (!body || !Array.isArray(body.keys)) throw new Error('certs endpoint returned no keys');
        keys = body.keys.filter((k) => k && k.kty === 'RSA' && typeof k.kid === 'string');
        fetchedAt = now;
      })().finally(() => { inflight = null; });
    }
    await inflight;
  }

  return {
    async keyFor(kid, now = Date.now()) {
      if (!keys || now - fetchedAt > KEY_TTL_MS) await refresh(now);
      let found = keys.find((k) => k.kid === kid);
      if (!found && now - fetchedAt >= REFETCH_FLOOR_MS) {
        await refresh(now);
        found = keys.find((k) => k.kid === kid);
      }
      return found || null;
    },
  };
}
