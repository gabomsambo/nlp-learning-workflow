# NLP Reader

A read-only copy of the library for a phone or iPad, served from the personal
Cloudflare account at `read.gabrielsambo.com` so it works with the computer off.
It shows what `python -m nlp_pillars.cli cloud export` writes: Today, Library,
Paper and Lessons. It has no audio yet, no reviews, and nothing that generates.

Plan and reasoning: the cloud-reader report, sections 4, 7 and 9.

## What is deployed

| Piece | Name | Notes |
|---|---|---|
| Worker | `nlp-reader` | `src/`, no dependencies. No `workers.dev` address, no preview URLs |
| R2 bucket | `nlp-reader-docs` | The exported documents. Nothing else is bound |
| Access app | `NLP Reader`, `read.gabrielsambo.com` | Allow the two captain emails, 30-day session. Separate from the tunnel's `NLP Learning` app |
| DNS | one proxied `AAAA read` record | Created by the Worker's custom domain; no other record is touched |

The Worker's bindings are `DOCS` (R2) and `ASSETS` (static files). It has no model,
queue, database or service binding, and its only outbound request is the team's
Access key endpoint (`src/certs.js`).

## How a request is checked

1. Cloudflare Access answers first. Without a session it redirects to the login
   page and the Worker is never invoked.
2. The Worker runs before any static file (`run_worker_first`) and verifies the
   `Cf-Access-Jwt-Assertion` token itself: RS256 signature against the team's keys,
   issuer, this application's AUD, expiry. So a mistake in the Access setup fails
   closed.
3. The email must be on the `ALLOWED_EMAILS` secret. With it unset nobody gets in
   (503). A service token is refused unless its client id is on
   `ALLOWED_SERVICE_TOKENS`, which is empty and exists only to check a deployment.
4. Only `GET`/`HEAD`. `/api/manifest` and `/api/doc/<key>` serve R2 objects whose
   names match what `cloud export` writes; anything else is a 404.

Documents come with an `ETag` and are revalidated with `If-None-Match`. The page
checks each document against the SHA-256 in the manifest, so it never shows a
mixture of two publishes.

## Day to day

Run from the repository root. `wrangler` is pinned to 4.147.0 and run through `npx`;
there is no `package.json` and no CI deploy.

```sh
export XDG_CONFIG_HOME=$HOME/.config-cloudflare-account2   # the personal login, never the default one

python -m nlp_pillars.cli cloud export --out /tmp/bundle   # on the computer (see AGENTS.md for the container form)
node cloud/reader/scripts/upload-bundle.mjs /tmp/bundle    # documents first, manifest last; audio is skipped

cloud/reader/scripts/build-assets.sh                       # copies tokens.css, app.css, fonts, icons from webui/static
cd cloud/reader && npx wrangler@4.147.0 deploy             # only when the code or the look changed
```

`public/static/` is generated from `webui/static/` by `build-assets.sh` and is not
committed, so the reader and the app cannot drift apart. Run it before `deploy`.

## First-time setup (done once)

```sh
npx wrangler@4.147.0 r2 bucket create nlp-reader-docs
# Zero Trust: add a self-hosted application for read.gabrielsambo.com with the same
# two-email Allow policy and a 720h session. Put its AUD in wrangler.toml.
npx wrangler@4.147.0 deploy
printf '%s' 'one@example.com,two@example.com' | npx wrangler@4.147.0 secret put ALLOWED_EMAILS
```

## Try it without Cloudflare

```sh
node cloud/reader/scripts/dev-server.mjs /tmp/bundle 8787   # the real Worker, signing its own throwaway tokens
node --test cloud/reader/test/*.test.mjs                     # JWT, Worker, key store, helpers
cloud/reader/scripts/smoke-dev.sh                            # `wrangler dev`: everything unauthenticated is refused
```

Use the glob, not `node --test cloud/reader/test/`: the directory form resolves as a
module and fails with `MODULE_NOT_FOUND`.

## Not here yet

Audio and read-along, the offline service worker and install manifest, reviewing on
the phone, the D1 event log, and automatic publishing. Objects removed from a newer
bundle are not deleted from R2 by the manual upload.
