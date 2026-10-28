# NLP Reader

A copy of the library for a phone or iPad, served from the personal Cloudflare
account at `read.gabrielsambo.com` so it works with the computer off.
It shows what `python -m nlp_pillars.cli cloud export` writes: Today, Library,
Paper (with a read-only Playground tab), Lessons and Listen (episodes with audio,
read-along, fact-check highlights and the code drawer), and cards can be reviewed on it. `cloud publish` keeps it current; a review
is the one thing that travels the other way. It has no offline mode and nothing that generates.

Plan and reasoning: the cloud-reader report, sections 4, 7 and 9.

## What is deployed

| Piece | Name | Notes |
|---|---|---|
| Worker | `nlp-reader` | `src/`, no dependencies. No `workers.dev` address, no preview URLs |
| R2 bucket | `nlp-reader-docs` | The exported documents |
| D1 database | `nlp-reader-reviews` | One table, `review_events`: the phone's reviews waiting for the computer. A mailbox, not a source of truth |
| Access app | `NLP Reader`, `read.gabrielsambo.com` | Allow the two captain emails, 30-day session. Separate from the tunnel's `NLP Learning` app |
| Access app | `NLP Reader Ingest`, `read.gabrielsambo.com/ingest` | Service Auth only: one policy naming the `nlp-publisher` service token. No person can pass it |
| Service token | `nlp-publisher` | Its id and secret are `CF_ACCESS_CLIENT_ID` / `CF_ACCESS_CLIENT_SECRET` in the desktop's `.env`. It reaches `/ingest` and nothing else |
| DNS | one proxied `AAAA read` record | Created by the Worker's custom domain; no other record is touched |

The Worker's bindings are `DOCS` (R2), `REVIEWS` (D1) and `ASSETS` (static files). It has
no model, queue or service binding, and its only outbound request is the team's Access key
endpoint (`src/certs.js`).

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
4. Only `GET`/`HEAD`, and one `POST`: `/api/reviews` (below). `/api/manifest` and
   `/api/doc/<key>` serve R2 objects whose names match what `cloud export` writes;
   `/api/audio/<sha256>.mp3` serves `audio/<sha256>.mp3` with byte ranges. Anything else is a 404.
5. `/ingest/*` is separate. It is checked against its own Access application's AUD
   (`INGEST_AUD`), takes service tokens only, and the token's client id must be
   `INGEST_SERVICE_TOKENS`. A signed-in person gets 401 or 403 there, and the
   publisher's token gets 403 from Access on every reader path. `GET /ingest/reviews`
   is how the computer collects the phone's reviews.

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

## Playground (read-only)

A paper whose notebook has a study guide (`ready` or `partial`, docs/PLAYGROUND.md) gets a
Playground tab: the steps, each code cell as a card with its explanation (Hide explanations
for a self-quiz, Show explanation per card), the code with tap-a-line notes, the saved output,
and each "try this" as a before/after diff with its expect line. Nothing runs on the phone, and
the tab says so once. `#/paper/<id>/playground?cell=<n>` opens it at one cell.

When an episode follows a notebook (a ready `playground_links` row), a Code drawer under the
player shows the card for the code being talked about. It follows playback the way the desktop
panel does, Pin keeps the card, and tapping it opens that cell on the Playground tab (leaving
the episode, which stops it: there is no mini player). A mapping made for an older version of
the script is not followed; the drawer says to match it again on the computer.

- **In the bundle.** `playgrounds/<uuid>.json` per playground: title, status, steps (explanation,
  line notes, tries, quiz and answer), every cell's source, and saved outputs in the sanitized form the app's
  tab uses (text, HTML as a tag tree, PNG). No notebook JSON, datasets, model or error. The paper
  document carries a `playground` summary and each episode a `follow_along` list (spans, `stale`).
  Same hashing as every other document, so an unchanged playground is not uploaded again.
- **Caps.** Per playground, `CLOUD_PLAYGROUND_MAX_IMAGE_BYTES` (default 3 MB of base64 figure data)
  and `CLOUD_PLAYGROUND_MAX_OUTPUT_BYTES` (1.5 MB of every other saved output, as JSON). What does
  not fit is left out in notebook order, replaced by a `dropped` marker the card shows, counted in
  the document's `dropped`, and named in the export report. A playground that would still pass the
  8 MiB document limit, or that trips the secret guard (a key pasted into a cell), is left out
  whole with a warning; the rest of the library still publishes.
- **Shared code.** The cards are drawn by the app's `playground.js` (`renderCode`, `renderOutput`,
  the sanitizer's `buildTree`, and `lineDiff` from `playground-core.js`) and the drawer's
  time -> span -> card lookup is `follow-along.js`'s (`normalizeSpans`, `followAt`, `cardFor`,
  `availability`), all copied by `build-assets.sh`. `playground-session.js` (the runner) is not.
- **Before migrations 028/029** the export publishes without playgrounds (or without follow-along)
  and the report says which migration is missing.
- **Offline.** There is no offline layer yet (below), so a playground is fetched like any other
  document. When the offline step lands, a downloaded episode's `follow_along` docs belong in it.

## Reviewing on the phone

The Review page schedules cards on the device, so it works with no connection. A card is the
computer's published snapshot (`schedule.json`) with the reviews made since folded on top.

```
phone                                   cloud                          computer
rate a card
  -> outbox (localStorage)
  -> POST /api/reviews ---------------> review_events (D1)
                                          seq 1, 2, 3 ...  <---------- cloud pull-reviews
                                                                        (start, after each publish,
                                                                         every 10 minutes)
                                                                          -> review_logs (origin=cloud)
                                                                          -> cards re-derived from the log
                                                                          -> cloud_sync_state.reviews_seq
              schedule.json  <-------------- cloud publish  <-----------  published with through_seq
```

- **An event** is `{event_id, card_id, rating 1-4, reviewed_at, device_id}` plus an optional
  response time and session id. Every field has a fixed shape, an unknown field is refused,
  and there is no free text, so nothing stored can carry an instruction. The Worker checks it
  (`src/reviews.js`) and the computer checks it again (`nlp_pillars/cloud/reviews.py`).
- **Idempotent.** `event_id` is unique in D1 and becomes `review_logs.id`, so an event sent or
  pulled twice records once. The Worker answers each event separately (`accepted`, `duplicate`
  or `rejected` with a reason), so one bad event does not hold up the others.
- **Two devices.** Every review counts. Events are ordered by `(reviewed_at, event_id)` and the
  card is re-folded from its whole log, so the order they arrive in does not matter and the
  phone, the iPad and the computer land on the same card.
- **Time window.** A review stamped more than 5 minutes ahead of when the cloud received it, or
  more than 30 days behind, is refused: it is a wrong clock, not a review. The phone drops it
  and says so.
- **The outbox** (`public/js/outbox.js`) is the only data on the phone that cannot be fetched
  again. A review is written there first and kept until the computer's snapshot includes it
  (`through_seq`). Its storage is an interface; localStorage today, IndexedDB with the offline step.
- **Scheduling.** `public/js/fsrs.js` is a port of py-fsrs 6.3.1's scheduler, fuzz off, held to
  `tests/fixtures/fsrs/sequences.json` and to 400 random histories (every due date, step and state
  identical; stability and difficulty within 1e-12). It is a port rather than ts-fsrs because
  ts-fsrs 5.4.2 gives a different due date in 38 of the fixture's 100 reviews: it counts calendar
  days where py-fsrs counts 24-hour periods, it does not raise stability for a same-day Hard, it
  repeats the first step's Hard interval on later steps, and it rounds to 8 decimals after every
  review.
- **The computer is the authority.** When a late event sorts before one the snapshot already
  holds, folding forward from the snapshot (what the phone does) can differ from re-folding the
  whole log (what the computer does). The phone's view is then provisional until the next publish.

On the computer: apply `docs/migrations/027_cloud_sync_state.sql` by hand, then the webui pulls
whenever `CLOUD_PUBLISH_ENABLED=true` (same service token as publishing). By hand:

```sh
python -m nlp_pillars.cli cloud pull-reviews --dry-run     # fetch and check, write nothing
python -m nlp_pillars.cli cloud pull-reviews
```

If a pull stops half-way (the database goes away), `cloud_sync_state.reviews_pending` stays ahead
of `reviews_seq` and `cloud export` refuses to publish a snapshot until the next pull finishes the
job. Events for a card that no longer exists are skipped and reported; the cursor moves past them.

## Day to day

Run from the repository root. `wrangler` is pinned to 4.147.0 and run through `npx`;
there is no `package.json` and no CI deploy.

```sh
export XDG_CONFIG_HOME=$HOME/.config-cloudflare-account2   # the personal login, never the default one

python -m nlp_pillars.cli cloud publish                    # export, then send what changed (--dry-run, --full)

python -m nlp_pillars.cli cloud export --out /tmp/bundle   # the manual route, on the computer (see AGENTS.md for the container form)
node cloud/reader/scripts/upload-bundle.mjs /tmp/bundle    # audio, then documents, then the manifest last (--no-audio to skip the MP3s)

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
npx wrangler@4.147.0 d1 create nlp-reader-reviews          # put its id in wrangler.toml
npx wrangler@4.147.0 d1 migrations apply nlp-reader-reviews --remote
# Publishing: create the `nlp-publisher` service token, then a second self-hosted Access
# application for read.gabrielsambo.com/ingest with one `non_identity` policy that includes only
# that token. Put the application's AUD in INGEST_AUD and the token's client id in
# INGEST_SERVICE_TOKENS (wrangler.toml), and deploy.
```

## Try it without Cloudflare

```sh
node cloud/reader/scripts/dev-server.mjs /tmp/bundle 8787   # the real Worker, signing its own throwaway tokens
node cloud/reader/scripts/dev-ingest-server.mjs 8788        # the real Worker with a writable in-memory bucket; point `cloud publish` at it
node --test cloud/reader/test/*.test.mjs                     # JWT, Worker, reviews, the scheduler against py-fsrs, helpers
cloud/reader/scripts/smoke-dev.sh                            # `wrangler dev`: refuses everything unauthenticated; serves from simulated R2, D1 and assets
```

Use the glob, not `node --test cloud/reader/test/`: the directory form resolves as a
module and fails with `MODULE_NOT_FOUND`.

`dev-server.mjs` also keeps review events (in SQLite, lost when it stops) and answers `/ingest/reviews`, so
the real pull can run against it: `CLOUD_PUBLISH_URL=http://127.0.0.1:8787 CF_ACCESS_CLIENT_ID=x
CF_ACCESS_CLIENT_SECRET=x python -m nlp_pillars.cli cloud pull-reviews`.

## Deploying

`wrangler deploy` ships whatever branch it is run from, and the Worker's code and bindings come
from that branch alone. Deploy from merged `main`, and apply the D1 migration first. Deploy the
Worker before the computer publishes a bundle with a new kind of key: a Worker older than the
Playground refuses `playgrounds/<uuid>.json` (404 on `PUT /ingest/object`), and the publish fails
and is retried until the deploy. A deploy
from a branch that lacks `REVIEWS` drops the binding and `/api/reviews` answers 503 until the
next deploy.

## Not here yet

The offline service worker and install manifest (a page that is open keeps working without a
connection; one that is closed needs it to load, and card text is fetched when the card is shown;
playgrounds are fetched the same way), and a mini player that outlives the page.
