#!/bin/sh
# Two `wrangler dev` runs (local mode, bindings simulated):
#   1. the real Worker and wrangler.toml: everything without a valid Access token is refused;
#   2. test/smoke-entry.mjs, the same Worker with a throwaway signing key and a seeded
#      local R2, to see the real assets and R2 bindings serve the page, documents and ETags.
# The signed-in logic itself is covered by test/worker.test.mjs.
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
fixture="$here/../../tests/fixtures/cloud/golden"
cd "$here"
./scripts/build-assets.sh >/dev/null

log=$(mktemp)
state=$(mktemp -d)
pid=""
stop() { [ -z "$pid" ] || kill -- -"$pid" 2>/dev/null || true; pid=""; }
trap 'stop; rm -rf "$log" "$state"' EXIT INT TERM

# serve <port> <wrangler args...>: start wrangler dev in its own session and wait for it.
serve() {
  port=$1; shift
  stop
  setsid npx --yes wrangler@4.147.0 dev --local --port "$port" --ip 127.0.0.1 --persist-to "$state" "$@" >"$log" 2>&1 &
  pid=$!
  for _ in $(seq 1 60); do
    if curl -s -o /dev/null "http://127.0.0.1:$port/"; then return 0; fi
    sleep 1
  done
  echo "wrangler dev did not start"; cat "$log"; exit 1
}

fail=0
check() { # description expected-status curl-args...
  desc=$1; want=$2; shift 2
  got=$(curl -s -o /dev/null -w '%{http_code}' "$@")
  if [ "$got" = "$want" ]; then echo "ok   $desc ($got)"; else echo "FAIL $desc: wanted $want, got $got"; fail=1; fi
}
expect() { # description needle haystack
  case "$3" in *"$2"*) echo "ok   $1" ;; *) echo "FAIL $1: no '$2' in: $(printf %s "$3" | head -c 200)"; fail=1 ;; esac
}

echo "== real Worker, no identity"
base="http://127.0.0.1:${PORT:-8799}"
serve "${PORT:-8799}"
check "page without a token"            401 "$base/"
check "manifest without a token"        401 "$base/api/manifest"
check "static file without a token"     401 "$base/static/app.css"
check "garbage token"                   401 -H 'Cf-Access-Jwt-Assertion: not-a-jwt' "$base/api/manifest"
check "unsigned token (alg none)"       401 -H "Cf-Access-Jwt-Assertion: $(printf '{"alg":"none","kid":"x"}' | base64 | tr '+/' '-_' | tr -d '=\n').e30.x" "$base/"
check "write without a token"           405 -X PUT "$base/api/manifest"
expect "refusal body is generic" '"sign in required"' "$(curl -s "$base/api/manifest")"

echo "== same Worker, throwaway key, seeded R2"
for file in manifest.json library.json schedule.json; do
  npx --yes wrangler@4.147.0 r2 object put "nlp-reader-docs/$file" --file "$fixture/$file" \
    --content-type application/json --local --persist-to "$state" >/dev/null
done
npx --yes wrangler@4.147.0 r2 object put "nlp-reader-docs/papers/arxiv_3a1706.03762.json" \
  --file "$fixture/papers/arxiv_3a1706.03762.json" --content-type application/json --local --persist-to "$state" >/dev/null
npx --yes wrangler@4.147.0 d1 migrations apply nlp-reader-reviews --local --persist-to "$state" -c test/wrangler.smoke.toml >/dev/null
audio=$(cd "$fixture" && ls audio/*.mp3 | head -1)   # 22 bytes: enough to see real R2 slice it
npx --yes wrangler@4.147.0 r2 object put "nlp-reader-docs/$audio" --file "$fixture/$audio" \
  --content-type audio/mpeg --local --persist-to "$state" >/dev/null
base="http://127.0.0.1:$((${PORT:-8799} + 1))"
serve "$((${PORT:-8799} + 1))" -c test/wrangler.smoke.toml
expect "page is served by the assets binding" '<title>NLP Reader</title>' "$(curl -s "$base/")"
check "script module is served"          200 "$base/js/reader.js"
check "app.css is served from the build" 200 "$base/static/app.css"
check "font is served"                   200 "$base/static/fonts/inter-latin.woff2"
check "shared Listen helpers are served" 200 "$base/static/listen-core.js"
check "episode page code is served"      200 "$base/js/episodes.js"
expect "security headers on assets"      "frame-ancestors 'none'" "$(curl -s -D- -o /dev/null "$base/")"
expect "caller is reported"              '"email":"dev@example.com"' "$(curl -s "$base/api/me")"
etag=$(curl -s -D- -o /dev/null "$base/api/manifest" | tr -d '\r' | awk -F': ' 'tolower($1)=="etag"{print $2}')
[ -n "$etag" ] && echo "ok   manifest has an ETag ($etag)" || { echo "FAIL manifest has no ETag"; fail=1; }
check "unchanged manifest revalidates"   304 -H "If-None-Match: $etag" "$base/api/manifest"
check "a changed ETag gets the body"     200 -H 'If-None-Match: "other"' "$base/api/manifest"
check "document by key"                  200 "$base/api/doc/library.json"
check "document with an encoded id"      200 "$base/api/doc/papers%2Farxiv_3a1706.03762.json"
check "unpublished key"                  404 "$base/api/doc/episodes/00000000-0000-4000-8000-000000000000.json"
check "audio is not served"              404 "$base/api/doc/audio/abc.mp3"

echo "== audio: byte ranges against the simulated R2"
headers_of() { curl -s -D- -o /dev/null "$@" | tr -d '\r' | tr 'A-Z' 'a-z'; }   # workerd keeps header-name case
expect "audio advertises ranges"          'accept-ranges: bytes'         "$(headers_of "$base/api/$audio")"
expect "audio is audio/mpeg"              'content-type: audio/mpeg'     "$(headers_of "$base/api/$audio")"
expect "whole file is 200 of 22 bytes"    'content-length: 22'           "$(headers_of "$base/api/$audio")"
probe=$(headers_of -H 'Range: bytes=0-1' "$base/api/$audio")
expect "Safari's bytes=0-1 probe is 206"  'http/1.1 206'                 "$probe"
expect "206 names the slice"              'content-range: bytes 0-1/22'  "$probe"
expect "206 is two bytes long"            'content-length: 2'            "$probe"
expect "a middle range"                   'content-range: bytes 5-9/22'  "$(headers_of -H 'Range: bytes=5-9' "$base/api/$audio")"
expect "an open-ended range"              'content-range: bytes 10-21/22' "$(headers_of -H 'Range: bytes=10-' "$base/api/$audio")"
expect "a suffix range"                   'content-range: bytes 18-21/22' "$(headers_of -H 'Range: bytes=-4' "$base/api/$audio")"
check "a range past the end"              416 -H 'Range: bytes=22-' "$base/api/$audio"
sliced=$(curl -s -H 'Range: bytes=5-9' "$base/api/$audio" | wc -c)
[ "$sliced" = 5 ] && echo "ok   the body of bytes=5-9 is 5 bytes" || { echo "FAIL body was $sliced bytes"; fail=1; }
check "unpublished audio"                 404 "$base/api/audio/$(printf 'ab%.0s' $(seq 1 32)).mp3"
check "anonymous request is refused"     401 -H 'x-smoke-anonymous: 1' -H 'Cf-Access-Jwt-Assertion: nope' "$base/api/manifest"

echo "== review events, on the real D1 binding"
now=$(date -u +%Y-%m-%dT%H:%M:%S.000Z)
event='{"event_id":"11111111-1111-4111-8111-111111111111","card_id":"22222222-2222-4222-8222-222222222222","rating":3,"reviewed_at":"'$now'","device_id":"smoke-device-1"}'
post() { curl -s -H 'content-type: application/json' -H 'sec-fetch-site: same-origin' -d "$1" "$base/api/reviews"; }
expect "a review is accepted with its seq" '"status":"accepted","seq":1' "$(post "{\"events\":[$event]}")"
expect "the same review again is a duplicate" '"status":"duplicate","seq":1' "$(post "{\"events\":[$event]}")"
expect "a changed review with that id is refused" 'different content' "$(post "{\"events\":[$(printf %s "$event" | sed 's/"rating":3/"rating":1/')]}")"
expect "events can be read back" '"through_seq":1' "$(curl -s "$base/api/reviews?since=0")"
expect "after the tip there is nothing" '"events":[]' "$(curl -s "$base/api/reviews?since=1")"
check "a review with a malformed id is a per-event refusal (200)" 200 -H 'content-type: application/json' -d '{"events":[{"event_id":"x"}]}' "$base/api/reviews"
check "a cross-site post is refused"     403 -H 'content-type: application/json' -H 'sec-fetch-site: cross-site' -d "{\"events\":[$event]}" "$base/api/reviews"
check "a post that is not JSON is refused" 415 -H 'content-type: text/plain' -d 'x' "$base/api/reviews"
check "/ingest/reviews is not for a reader's session" 401 "$base/ingest/reviews"
exit $fail
