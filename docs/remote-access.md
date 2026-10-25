# Remote access (phone and iPad)

The app runs on the captain's machine and reaches the GPU and the paid model
APIs. To use it away from the desk it goes out through a **Cloudflare Tunnel**
with **Cloudflare Access** in front. Nothing is opened on the router, and
nobody but the allowed emails can reach a page, let alone start GPU work or
spend money.

    phone / iPad ──HTTPS──> Cloudflare edge ── Access login (email PIN) ──┐
                                                                          │ tunnel
    desktop ───────────────> http://localhost:8000 <── cloudflared ───────┘
                              (published on 127.0.0.1 only)

It works only while the computer is on, the stack is up, and `cloudflared` is
running.

## The layers

1. **Cloudflare Access** (outside the app). One-time-PIN login for exactly two
   emails, default deny for everyone else, 30-day session so the phone does
   not ask every day.
2. **Loopback-only ports** (`docker-compose.yml`). `webui`, `qdrant` and
   `searxng` publish on `127.0.0.1`, like `db` and `postgrest` already did, so
   the LAN cannot reach them. The tunnel dials `127.0.0.1:8000`.
3. **Token check in the app** (`webui/access.py`). A request that came through
   the tunnel (it carries `Cf-*` / `X-Forwarded-*` headers) must have a valid
   `Cf-Access-Jwt-Assertion`: RS256 signature from the team's keys, the right
   issuer and application AUD, not expired. Missing or invalid is 401/403 and
   no route runs. If `CF_ACCESS_TEAM_DOMAIN` / `CF_ACCESS_AUD` are not set,
   remote requests are refused. Requests from the desktop
   (`http://localhost:8000`, no proxy headers) are unchanged.
4. **Spend guard** (`webui/services/spend_guard.py`, `nlp_pillars/spend.py`).
   Paper processing, uploads, the daily pipeline run, podcast scripts, audio,
   and fact-checks (including the automatic one after a script):
   - show a confirm dialog with the remaining budget (`spend-guard.js`); a
     request through the tunnel without the confirmation header is refused
     with 428;
   - are refused with 429 once today's recorded spend reaches
     `DAILY_SPEND_CAP_USD` (default `2`), and with 503 if the spend cannot be
     read.

   Spend is the sum of the `$` amounts the app already writes into stage
   details plus `podcast_fact_checks.cost_usd`, since midnight in
   `SPEND_CAP_TIMEZONE` (default UTC). It is a **lower bound**: a multi-paper
   run keeps only the last paper's cost in its stage row, and a failed
   fact-check records nothing. The cap stops the *next* action; one big run can
   overshoot it. Audio generation costs no API dollars but is gated the same
   way because it holds the GPU.

   `GET /api/spend` shows the numbers. CLI backfills (`factcheck backfill`,
   `reprocess`, `embeddings`, ...) are not reachable through the tunnel and are
   not capped; do not run them on a day you are watching the budget.

## Settings

In `.env` (never committed):

| Variable | Meaning |
| --- | --- |
| `CF_ACCESS_TEAM_DOMAIN` | Zero Trust team domain, e.g. `<team>.cloudflareaccess.com` |
| `CF_ACCESS_AUD` | Application Audience (AUD) tag of the Access application |
| `CF_ACCESS_ALLOWED_EMAILS` | Optional, comma-separated extra check on the token's email |
| `DAILY_SPEND_CAP_USD` | Daily cap, default `2`; `0` refuses everything costly |
| `SPEND_CAP_TIMEZONE` | Day boundary, IANA name, default `UTC` |

Tunnel credentials stay in `~/.cloudflared/`, outside the repo. After editing
`.env` or merging this change: `./workstation up --build`.

## Setup (once)

All of this is on the **personal** Cloudflare account
(`gabriel.m.sambo@gmail.com`), zone `gabrielsambo.com`, hostname
`nlp.gabrielsambo.com`.

1. **Zero Trust.** Enable the free plan (done: team `soft-wave-31cb`).
2. **cloudflared.** Install the pinned release binary as your user:
   ```
   mkdir -p ~/.local/bin ~/.cloudflared
   curl -fsSL -o ~/.local/bin/cloudflared \
     https://github.com/cloudflare/cloudflared/releases/download/<version>/cloudflared-linux-amd64
   chmod +x ~/.local/bin/cloudflared
   ```
3. **Tunnel.** Create a named, remotely-managed tunnel `nlp-learning`, give it
   the public hostname `nlp.gabrielsambo.com` -> `http://127.0.0.1:8000`
   (Zero Trust > Networks > Tunnels, or the API). Save the tunnel token in
   `~/.cloudflared/nlp-learning.token` (`chmod 600`).
4. **Service.** `~/.config/systemd/user/cloudflared-nlp.service`:
   ```
   [Unit]
   Description=Cloudflare Tunnel for the NLP app
   After=network-online.target

   [Service]
   ExecStart=/bin/sh -c 'exec %h/.local/bin/cloudflared tunnel --no-autoupdate run --token "$(cat %h/.cloudflared/nlp-learning.token)"'
   Restart=on-failure
   RestartSec=5

   [Install]
   WantedBy=default.target
   ```
   then `systemctl --user daemon-reload && systemctl --user enable --now cloudflared-nlp`.
   For it to start at boot rather than at first login:
   `loginctl enable-linger $USER`.
5. **Access application.** Zero Trust > Access > Applications > Add >
   Self-hosted: domain `nlp.gabrielsambo.com`, session duration 30 days,
   identity provider One-time PIN, one Allow policy including exactly
   `gabriel.m.sambo@gmail.com` and `sambogabriel370@gmail.com`. Copy the
   application's **AUD** tag into `CF_ACCESS_AUD` and the team domain into
   `CF_ACCESS_TEAM_DOMAIN`, then `./workstation up --build`.
6. **Check.** `curl -sI https://nlp.gabrielsambo.com/` must answer a redirect
   to `<team>.cloudflareaccess.com` (or the Access login page), and
   `docker logs nlp_webui` must show no request for it.

## Add a device

1. Open `https://nlp.gabrielsambo.com/` in Safari on the phone or iPad.
2. Enter one of the two allowed emails, then the 6-digit code Cloudflare mails
   you. The session lasts 30 days.
3. Share > **Add to Home Screen**. It opens full-screen from then on.

Costly buttons ask for confirmation and show the remaining budget beside them.

## Revoke access

- **One device or person:** Zero Trust > Access > Users (or the application's
  *Sessions*), revoke the user's sessions. They must sign in again.
- **Remove an email:** edit the Allow policy and delete it, then revoke
  sessions (existing sessions stay valid until revoked or expired).
- **Everything now:** turn the tunnel off (below).

## Turn the tunnel off

    systemctl --user disable --now cloudflared-nlp

The hostname then answers Cloudflare's "tunnel down" error and the app is
reachable only at `http://localhost:8000`. Turn it back on with `enable --now`.
To remove it for good, delete the tunnel and the Access application in the
dashboard.

## Troubleshooting

- **403 "Remote access is not configured"**: `CF_ACCESS_*` missing in the
  container. Check `docker exec nlp_webui env | grep CF_ACCESS` and rebuild.
- **403 "Access token not accepted"**: wrong AUD (a different Access
  application) or the team domain does not match the token's issuer.
- **429 on a button**: today's cap is reached; see `GET /api/spend`, or raise
  `DAILY_SPEND_CAP_USD`.
- **Wrong "remote or desktop" call**: the app classifies by proxy headers, not
  client IP, because Docker rewrites the source address of host-loopback
  connections to the bridge gateway. Anything that adds `X-Forwarded-*` in
  front of `localhost:8000` makes the desktop look remote.
