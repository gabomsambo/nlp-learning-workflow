# NLP Reader

A read-only copy of the library for a phone or iPad, served from the personal
Cloudflare account at `read.gabrielsambo.com` so it works with the computer off.
It shows what `python -m nlp_pillars.cli cloud export` writes: Today, Library,
Paper, Lessons and Listen (episodes with audio, read-along and fact-check
highlights). It has no reviews, no offline mode, and nothing that generates.

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
   names match what `cloud export` writes; `/api/audio/<sha256>.mp3` serves
   `audio/<sha256>.mp3` with byte ranges. Anything else is a 404.

Documents come with an `ETag` and are revalidated with `If-None-Match`. The page
checks each document against the SHA-256 in the manifest, so it never shows a
mixture of two publishes.

## Audio

`src/audio.js` streams an episode out of R2. Safari will not play, and cannot seek in,
an `<audio>` source unless the server answers `Range` with `206` and says
`Accept-Ranges: bytes`, so that is the one thing it does: one `Range` header becomes an
offset and a length for R2, and the answer carries `Content-Range`. A range past the end
is `416`; a header it cannot read (several ranges, nonsense) is ignored and the whole
file is sent, which HTTP allows. `If-Range` and `If-None-Match` are honoured, and `HEAD`
reports the same lengths with no body. The name is the hash of the bytes, so the
response is `private, max-age=31536000, immutable`. `media-src 'self'` is in the CSP.

The episode page (`public/js/episodes.js`) is the read-along from the app's Listen page:
the spoken word is highlighted, the text follows it, a tap on a word or a line's
timestamp seeks, "Jump to now" brings you back after you scroll away, and fact-check
marks open their evidence. The pure parts (`readalong.js`, `factcheck.js`, `listen-core.js`)
are the app's own files from `webui/static/`, copied by `build-assets.sh` and loaded as
classic scripts, so the two pages cannot disagree about a line number, a word or a
verdict. The DOM and the `<audio>` element are the reader's. It remembers where you
stopped and the playback speed in `localStorage`, and sets the lock-screen controls.
Leaving the page stops the audio; there is no mini player yet.

## Day to day

Run from the repository root. `wrangler` is pinned to 4.147.0 and run through `npx`;
there is no `package.json` and no CI deploy.

```sh
export XDG_CONFIG_HOME=$HOME/.config-cloudflare-account2   # the personal login, never the default one

python -m nlp_pillars.cli cloud export --out /tmp/bundle   # on the computer (see AGENTS.md for the container form)
node cloud/reader/scripts/upload-bundle.mjs /tmp/bundle    # audio, then documents, then the manifest last (--no-audio to skip the MP3s)

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
cloud/reader/scripts/smoke-dev.sh                            # `wrangler dev`: refuses everything unauthenticated; serves from simulated R2 and assets
```

Use the glob, not `node --test cloud/reader/test/`: the directory form resolves as a
module and fails with `MODULE_NOT_FOUND`.

## Not here yet

The offline service worker and install manifest, a mini player that outlives the page,
reviewing on the phone, the D1 event log, and automatic publishing. Objects removed from a newer
bundle are not deleted from R2 by the manual upload.
