# Cloud reader: operating guide

The reader at `https://read.gabrielsambo.com` is a read-only copy of the library on the
personal Cloudflare account, so the phone and iPad work with the computer off.
What it shows, how a request is checked and how publishing works are in
[`cloud/reader/README.md`](../cloud/reader/README.md). This page is what to do with it.

The computer is the source of truth and holds every key. The cloud holds a published copy
(R2), a mailbox of the phone's card reviews (D1) and the Worker. It has no model, no API
key, no queue and no way back to the computer; the only thing that travels toward the
computer is a review event.

## What stops the reader calling AI or an outside service

`python scripts/check_cloud_reader.py` runs in CI (job `guard`) and fails the build when:

- `cloud/reader/wrangler.toml` (or `test/wrangler.smoke.toml`) declares anything but the
  bindings `DOCS` (R2), `REVIEWS` (D1) and `ASSETS`, or a `[vars]` name not on its list;
- the Worker source in `cloud/reader/src` fetches anything but the team's Access key
  endpoint (`certs.js`), opens a socket, imports a package, or holds a URL literal other
  than `https://${teamDomain}`;
- the page in `cloud/reader/public` names an absolute URL or uses a WebSocket, XHR or beacon;
- the Worker's Content-Security-Policy loses `connect-src 'self'` or gains a wildcard.

It is a tripwire, and the binding list and the CSP are the boundary it holds. To allow
something new, a person edits the allow-lists at the top of the script in a diff a reviewer
reads. `tests/test_check_cloud_reader.py` holds a failing fixture for each rule.

## Setup

Done once; the commands and the Access application settings are under "First-time setup" in
the README. In short: the R2 bucket `nlp-reader-docs`, the D1 database `nlp-reader-reviews`
and its migration, two Access applications (`read.gabrielsambo.com` for people,
`read.gabrielsambo.com/ingest` for the service token only), the `ALLOWED_EMAILS` secret, and
a deploy of the Worker from merged `main`.

On the computer, in `.env`, then `./workstation up`:

    CLOUD_PUBLISH_ENABLED=true
    CLOUD_PUBLISH_URL=https://read.gabrielsambo.com
    CF_ACCESS_CLIENT_ID=<nlp-publisher client id>
    CF_ACCESS_CLIENT_SECRET=<its secret>

Apply migrations 026 and 027 by hand (`cloud_publish_state`, `cloud_sync_state`) before the
first publish. Check it with `python -m nlp_pillars.cli cloud publish --dry-run`, then
without `--dry-run`. Activity shows a "Cloud reader" row.

## Add a device

Access admits a person, not a device. On the new phone or iPad open the address, give an
email that is on the allow list, and enter the one-time PIN it sends. The session lasts 30
days; after that the page asks to sign in again. Nothing is installed or registered, and the
device name a review carries is made by the page.

To let a new person in, add their email in both places, because the Worker checks the list
itself as well as Access:

    printf '%s' 'one@example.com,two@example.com,new@example.com' \
      | npx wrangler@4.147.0 secret put ALLOWED_EMAILS      # from cloud/reader/

and add it to the Allow policy of the `NLP Reader` application in Zero Trust.

## Revoke a device or a person

- **Lost phone:** in Zero Trust, Users, revoke that person's session. The next request is
  sent to the login page. Content already downloaded to the phone stays until it is wiped,
  the same as any file, so the device lock matters.
- **A person who should no longer have access:** remove the email from the Allow policy and
  from the `ALLOWED_EMAILS` secret (command above, without the email).
- Reviews already sent are not lost: the computer pulls them whenever it is publishing.

## Rotate the publisher's token

The desktop holds one service token, `nlp-publisher`. It can reach `/ingest` and nothing else.

1. Zero Trust, Access, Service Auth: create a new token and copy the id and secret.
2. Put them in `.env` as `CF_ACCESS_CLIENT_ID` / `CF_ACCESS_CLIENT_SECRET`.
3. Put the new client id in `INGEST_SERVICE_TOKENS` in `cloud/reader/wrangler.toml`, add the
   token to the `NLP Reader Ingest` policy, and deploy the Worker (`npx wrangler@4.147.0 deploy`).
4. Recreate the stack so it reads `.env`, check with `cloud publish --dry-run`, then delete the
   old token in Zero Trust. Deleting it revokes it at once.

If the token leaks, delete it first. Publishing fails and retries until the new one is in
place; the reader keeps serving what it already holds.

## Turn it off

- **Stop publishing:** set `CLOUD_PUBLISH_ENABLED=false` in `.env` and recreate the stack.
  The reader keeps serving the last bundle. Reviews made on the phone wait in D1 and are
  pulled when publishing comes back; to take them now run
  `python -m nlp_pillars.cli cloud pull-reviews` first.
- **Stop serving:** `npx wrangler@4.147.0 secret delete ALLOWED_EMAILS` from `cloud/reader/`.
  With no list the Worker answers 503 to everyone, signed in or not. Putting the secret back
  turns it on again.
- **Remove it:** pull reviews, then delete the Worker (`npx wrangler@4.147.0 delete`, which
  also removes the custom domain), the two Access applications, the `nlp-publisher` token, the
  R2 bucket and the D1 database. Nothing on the computer depends on them, and `cloud export`
  still writes a bundle to a directory.

Cloud costs are storage only, and the publisher refuses a bundle past `CLOUD_PUBLISH_MAX_BYTES`
(default 5 GB of the 10 GB free tier, which other projects on the account share).
