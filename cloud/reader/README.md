# NLP Reader

A read-only copy of the library for a phone or iPad, served from the personal
Cloudflare account at `read.gabrielsambo.com` so it works with the computer off.
It shows what `python -m nlp_pillars.cli cloud export` writes: Today, Library,
Paper and Lessons. `cloud publish` keeps it current, and nothing in it generates.

Plan and reasoning: the cloud-reader report, sections 4, 7 and 9.

## What is deployed

| Piece | Name | Notes |
|---|---|---|
| Worker | `nlp-reader` | `src/`, no dependencies. No `workers.dev` address, no preview URLs |
| R2 bucket | `nlp-reader-docs` | The exported documents. Nothing else is bound |
| Access app | `NLP Reader`, `read.gabrielsambo.com` | Allow the two captain emails, 30-day session. Separate from the tunnel's `NLP Learning` app |
| Access app | `NLP Reader Ingest`, `read.gabrielsambo.com/ingest` | Service Auth only: one policy naming the `nlp-publisher` service token. No person can pass it |
| Service token | `nlp-publisher` | Its id and secret are `CF_ACCESS_CLIENT_ID` / `CF_ACCESS_CLIENT_SECRET` in the desktop's `.env`. It reaches `/ingest` and nothing else |
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
5. `/ingest/*` is separate. It is checked against its own Access application's AUD
   (`INGEST_AUD`), takes service tokens only, and the token's client id must be
   `INGEST_SERVICE_TOKENS`. A signed-in person gets 401 or 403 there, and the
   publisher's token gets 403 from Access on every reader path.

Documents come with an `ETag` and are revalidated with `If-None-Match`. The page
checks each document against the SHA-256 in the manifest, so it never shows a
mixture of two publishes.

## Publishing

`python -m nlp_pillars.cli cloud publish` (or the webui itself, when
`CLOUD_PUBLISH_ENABLED=true`) exports the bundle and sends what changed:

1. `GET /ingest/inventory` lists what R2 holds, with each object's SHA-256.
2. Each new or changed object goes up with `PUT /ingest/object/<key>` and an
   `x-content-sha256` header, audio first. R2 checks the hash itself and refuses a
   body that does not match; an audio file's key *is* its hash. Each object is
   recorded in `cloud_publish_state` as it lands, so an interrupted publish resumes.
3. `PUT /ingest/manifest` goes **last**, and the Worker refuses it while any object it
   names is missing or has a different hash. A reader never sees a half-published bundle.
4. The same call garbage-collects. A key the new manifest does not list is deleted once
   it has been unreferenced for 7 days (counted from its tombstone's `deleted_at`, or
   from its upload if it never had one), so a phone that was offline can still catch up.

Limits: 8 MiB per document, 40 MiB per audio file, and `INGEST_MAX_BYTES` (5 GB) for the
bucket. The bucket shares the account's 10 GB free tier with other projects. The publisher
refuses a bundle that would pass the cap before it uploads anything (`CLOUD_PUBLISH_MAX_BYTES`).

Turn automatic publishing on in the desktop's `.env`, then `./workstation up`:

```sh
CLOUD_PUBLISH_ENABLED=true
CLOUD_PUBLISH_URL=https://read.gabrielsambo.com
CF_ACCESS_CLIENT_ID=<nlp-publisher client id>
CF_ACCESS_CLIENT_SECRET=<its secret>
```

The webui then publishes about 45 seconds after a run finishes, a script is saved or a
review is made (a burst becomes one publish, never more than 5 minutes late), and checks
every 10 minutes for runs that finished in another process. A failed publish is retried
with a growing wait, shown on Activity as "Cloud reader", and never fails the run it follows.
Migration 026 (`cloud_publish_state`) must be applied by hand.

## Day to day

Run from the repository root. `wrangler` is pinned to 4.147.0 and run through `npx`;
there is no `package.json` and no CI deploy.

```sh
export XDG_CONFIG_HOME=$HOME/.config-cloudflare-account2   # the personal login, never the default one

python -m nlp_pillars.cli cloud publish                    # export, then send what changed (--dry-run, --full)

cloud/reader/scripts/build-assets.sh                       # copies tokens.css, app.css, fonts, icons from webui/static
cd cloud/reader && npx wrangler@4.147.0 deploy             # only when the code or the look changed
```

`public/static/` is generated from `webui/static/` by `build-assets.sh` and is not
committed, so the reader and the app cannot drift apart. Run it before `deploy`.

Rotate or revoke the publisher: in Zero Trust > Access > Service Auth, delete `nlp-publisher`
(revokes at once) or create a new token, put it in `.env` and in `INGEST_SERVICE_TOKENS`
(a client id, not a secret), and add it to the `NLP Reader Ingest` policy.

`scripts/upload-bundle.mjs` is the manual route (wrangler, documents only). Prefer `cloud publish`.

## First-time setup (done once)

```sh
npx wrangler@4.147.0 r2 bucket create nlp-reader-docs
# Zero Trust: add a self-hosted application for read.gabrielsambo.com with the same
# two-email Allow policy and a 720h session. Put its AUD in wrangler.toml.
npx wrangler@4.147.0 deploy
printf '%s' 'one@example.com,two@example.com' | npx wrangler@4.147.0 secret put ALLOWED_EMAILS
# Publishing: create the `nlp-publisher` service token, then a second self-hosted Access
# application for read.gabrielsambo.com/ingest with one `non_identity` policy that includes only
# that token. Put the application's AUD in INGEST_AUD and the token's client id in
# INGEST_SERVICE_TOKENS (wrangler.toml), and deploy.
```

## Try it without Cloudflare

```sh
node cloud/reader/scripts/dev-server.mjs /tmp/bundle 8787   # the real Worker, signing its own throwaway tokens
node cloud/reader/scripts/dev-ingest-server.mjs 8788        # the real Worker with a writable in-memory bucket; point `cloud publish` at it
node --test cloud/reader/test/*.test.mjs                     # JWT, Worker, key store, helpers
cloud/reader/scripts/smoke-dev.sh                            # `wrangler dev`: refuses everything unauthenticated; serves from simulated R2 and assets
```

Use the glob, not `node --test cloud/reader/test/`: the directory form resolves as a
module and fails with `MODULE_NOT_FOUND`.

## Not here yet

Playing the audio and read-along on the phone (the files are published), the offline
service worker and install manifest, reviewing on the phone, and the D1 event log.
