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
base="http://127.0.0.1:$((${PORT:-8799} + 1))"
serve "$((${PORT:-8799} + 1))" -c test/wrangler.smoke.toml
expect "page is served by the assets binding" '<title>NLP Reader</title>' "$(curl -s "$base/")"
check "script module is served"          200 "$base/js/reader.js"
check "app.css is served from the build" 200 "$base/static/app.css"
check "font is served"                   200 "$base/static/fonts/inter-latin.woff2"
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
check "anonymous request is refused"     401 -H 'x-smoke-anonymous: 1' -H 'Cf-Access-Jwt-Assertion: nope' "$base/api/manifest"
exit $fail
