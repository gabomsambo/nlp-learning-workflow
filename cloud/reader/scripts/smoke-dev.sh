#!/bin/sh
# Boot the Worker under `wrangler dev` (local mode: bindings are simulated) and
# check it refuses everything without a valid Access token. The signed-in paths
# are covered by test/worker.test.mjs, which signs its own tokens.
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
port=${PORT:-8799}
log=$(mktemp)
cd "$here"
./scripts/build-assets.sh >/dev/null
# Own session, so the whole wrangler process tree can be stopped at the end.
setsid npx --yes wrangler@4.147.0 dev --local --port "$port" --ip 127.0.0.1 >"$log" 2>&1 &
pid=$!
trap 'kill -- -$pid 2>/dev/null || true; rm -f "$log"' EXIT INT TERM

up=""
for _ in $(seq 1 60); do
  if curl -s -o /dev/null "http://127.0.0.1:$port/"; then up=yes; break; fi
  sleep 1
done
[ -n "$up" ] || { echo "wrangler dev did not start"; cat "$log"; exit 1; }

fail=0
check() { # description expected-status curl-args...
  desc=$1; want=$2; shift 2
  got=$(curl -s -o /dev/null -w '%{http_code}' "$@")
  if [ "$got" = "$want" ]; then echo "ok   $desc ($got)"; else echo "FAIL $desc: wanted $want, got $got"; fail=1; fi
}
base="http://127.0.0.1:$port"
check "page without a token"            401 "$base/"
check "manifest without a token"        401 "$base/api/manifest"
check "static file without a token"     401 "$base/static/app.css"
check "garbage token"                   401 -H 'Cf-Access-Jwt-Assertion: not-a-jwt' "$base/api/manifest"
check "unsigned token (alg none)"       401 -H "Cf-Access-Jwt-Assertion: $(printf '{"alg":"none","kid":"x"}' | base64 | tr '+/' '-_' | tr -d '=\n').e30.x" "$base/"
check "write without a token"           405 -X PUT "$base/api/manifest"
body=$(curl -s "$base/api/manifest")
case "$body" in *'"sign in required"'*) echo "ok   refusal body is generic" ;; *) echo "FAIL refusal body: $body"; fail=1 ;; esac
exit $fail
