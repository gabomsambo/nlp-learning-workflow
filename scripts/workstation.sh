#!/usr/bin/env bash
# Start, stop, and inspect the local NLP learning workstation.
#
# Stops containers with `docker compose stop` and leaves named volumes in
# place. The database and uploaded PDFs live in those volumes.
set -euo pipefail

SOURCE="${BASH_SOURCE[0]}"
while [[ -L "$SOURCE" ]]; do
  DIR="$(cd "$(dirname "$SOURCE")" && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd "$(dirname "$SOURCE")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${HOME:?HOME is not set}"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/nlp-workstation"
PID_FILE="$STATE_DIR/indextts.pid"
LOG_FILE="$STATE_DIR/indextts.log"
WEBUI_WAIT_SECONDS="${WEBUI_WAIT_SECONDS:-90}"
TTS_WAIT_SECONDS="${TTS_WAIT_SECONDS:-180}"
APP_URL="http://localhost:8000/"
WEBUI_HEALTH_URL="http://127.0.0.1:8000/health"
POSTGREST_URL_HOST="http://127.0.0.1:3000/"
SEARXNG_HEALTH_URL="http://127.0.0.1:8080/healthz"

if [[ -t 1 ]]; then
  C_OK=$'\033[32m'
  C_BAD=$'\033[31m'
  C_WARN=$'\033[33m'
  C_BOLD=$'\033[1m'
  C_OFF=$'\033[0m'
else
  C_OK=""
  C_BAD=""
  C_WARN=""
  C_BOLD=""
  C_OFF=""
fi

usage() {
  cat <<EOF
${C_BOLD}NLP learning workstation${C_OFF}

Usage:
  ./workstation up [--build] [--tts]   Start the stack and open the app
  ./workstation down                   Stop containers, keep database and uploads
  ./workstation status                 Show whether each dependency is up
  ./workstation logs [service]         Follow logs (webui, db, postgrest, searxng, qdrant, scheduler,
                                       playground-runner, tts)
  ./workstation install-desktop        Add an application-menu icon for this checkout
  ./workstation help                   Show this help

up starts the Compose stack in the background. It rebuilds the web image only
when you pass --build. If the image is older than the last commit that touched
webui/ or nlp_pillars/, up says so and names the rebuild command.

The playground runner (locked-down Python for paper playgrounds) starts with
the stack. Idle it is one small process; a Python kernel starts only when a
playground session opens. It has no host port: status reads its health check.

IndexTTS (podcast audio) runs on the host GPU and is off unless you pass --tts.
It is started only when nothing is already listening on its port.

down stops containers this checkout started. It does not delete them or their data.
EOF
}

fail() {
  echo "$*" >&2
  exit 1
}

die() {
  echo "$*" >&2
  exit 2
}

compose() {
  docker compose --project-directory "$ROOT" "$@"
}

require_docker() {
  if ! docker info >/dev/null 2>&1; then
    fail "Docker is not running. Start Docker, then run: ./workstation up"
  fi
}

dotenv_get() {
  local key="$1"
  local file="$ROOT/.env"
  local line
  [[ -f "$file" ]] || return 1
  line="$(grep -E "^[[:space:]]*${key}=" "$file" | tail -n1 || true)"
  [[ -n "$line" ]] || return 1
  line="${line#*=}"
  line="${line%$'\r'}"
  if [[ ${#line} -ge 2 && "$line" == \"*\" && "$line" == *\" ]]; then
    line="${line:1:${#line}-2}"
  elif [[ ${#line} -ge 2 && "$line" == \'*\' && "$line" == *\' ]]; then
    line="${line:1:${#line}-2}"
  fi
  printf '%s' "$line"
}

# Defaults written as ${VAR:-value} in docker-compose.yml. A ${VAR:?msg}
# form has no default. Exported variables win, then .env, then that default.
compose_default() {
  local key="$1"
  local line val
  line="$(grep -E "^[[:space:]]+${key}:" "$ROOT/docker-compose.yml" | head -n1 || true)"
  [[ -n "$line" ]] || return 0
  val="${line#*:}"
  val="${val#"${val%%[![:space:]]*}"}"
  val="${val%"${val##*[![:space:]]}"}"
  if [[ ${#val} -ge 2 && "$val" == \"*\" && "$val" == *\" ]]; then
    val="${val:1:${#val}-2}"
  fi
  if [[ "$val" =~ ^\$\{[A-Za-z_][A-Za-z0-9_]*:-(.*)\}$ ]]; then
    val="${BASH_REMATCH[1]}"
  elif [[ "$val" =~ ^\$\{[A-Za-z_][A-Za-z0-9_]*:\? ]]; then
    val=""
  fi
  printf '%s' "$val"
}

resolve() {
  local key="$1"
  local val
  val="${!key-}"
  if [[ -n "$val" ]]; then
    printf '%s' "$val"
    return
  fi
  val="$(dotenv_get "$key" || true)"
  if [[ -n "$val" ]]; then
    printf '%s' "$val"
    return
  fi
  compose_default "$key"
}

host_tts_url() {
  local url
  url="$(resolve INDEXTTS_URL)"
  [[ -n "$url" ]] || url="http://127.0.0.1:7860"
  url="${url%/}"
  url="${url/host.docker.internal/127.0.0.1}"
  printf '%s' "$url"
}

port_from_url() {
  local rest="${1#*://}"
  rest="${rest%%/*}"
  rest="${rest%%\?*}"
  if [[ "$rest" == *:* ]]; then
    printf '%s' "${rest##*:}"
  else
    printf '%s' "7860"
  fi
}

port_open() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    if ss -H -ltn "( sport = :${port} )" 2>/dev/null | grep -q .; then
      return 0
    fi
    return 1
  fi
  timeout 0.2 bash -c "echo >/dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1
}

http_get() {
  curl -sS --max-time 8 "$1" 2>/dev/null || true
}

webui_ready() {
  local body
  body="$(http_get "$WEBUI_HEALTH_URL")"
  [[ "$body" == *'"ok"'* ]]
}

indextts_ready() {
  local body
  body="$(http_get "${1}/gradio_api/info")"
  [[ "$body" == *"/gen_single"* ]]
}

wait_until() {
  local label="$1"
  local seconds="$2"
  shift 2
  local start=$SECONDS
  printf 'Waiting for %s' "$label"
  while (( SECONDS - start < seconds )); do
    if "$@"; then
      printf ' ready.\n'
      return 0
    fi
    printf '.'
    sleep 1
  done
  printf '\n'
  echo "${label} did not become ready within ${seconds}s."
  return 1
}

warn_if_image_stale() {
  local image_id created code_iso image_epoch code_epoch short
  image_id="$(compose images -q webui 2>/dev/null | head -n1 || true)"
  [[ -n "$image_id" ]] || return 0
  created="$(docker image inspect --format '{{.Created}}' "$image_id" 2>/dev/null || true)"
  [[ -n "$created" ]] || return 0
  code_iso="$(git -C "$ROOT" log -1 --format=%cI -- webui nlp_pillars 2>/dev/null || true)"
  [[ -n "$code_iso" ]] || return 0
  image_epoch="$(date -d "$created" +%s 2>/dev/null || true)"
  code_epoch="$(date -d "$code_iso" +%s 2>/dev/null || true)"
  [[ -n "$image_epoch" && -n "$code_epoch" ]] || return 0
  if (( code_epoch > image_epoch )); then
    short="$(git -C "$ROOT" log -1 --format=%h -- webui nlp_pillars)"
    printf '%sThe webui image is older than the last commit that changed webui/ or nlp_pillars/.%s\n' \
      "$C_WARN" "$C_OFF"
    printf '  image built: %s\n' "$created"
    printf '  code commit: %s (%s)\n' "$code_iso" "$short"
    printf '  Rebuild with: ./workstation up --build\n'
  fi
}

start_indextts_if_needed() {
  local url port cmd
  url="$(host_tts_url)"
  port="$(port_from_url "$url")"
  cmd="$(resolve INDEXTTS_START_COMMAND)"
  [[ -n "$cmd" ]] || fail "INDEXTTS_START_COMMAND is empty. Set it in .env or docker-compose.yml."
  if port_open "$port"; then
    echo "IndexTTS is already listening on port ${port}."
    return 0
  fi
  mkdir -p "$STATE_DIR"
  echo "Starting IndexTTS. Log: ${LOG_FILE}"
  # setsid gives this process its own group so down can signal only what we started.
  setsid nohup bash -c "$cmd" >>"$LOG_FILE" 2>&1 </dev/null &
  printf '%s\n' "$!" >"$PID_FILE"
}

stop_indextts_if_ours() {
  local pid pgid
  [[ -f "$PID_FILE" ]] || return 0
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    rm -f "$PID_FILE"
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    return 0
  fi
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ "$pgid" != "$pid" ]]; then
    echo "IndexTTS pid ${pid} is not a process this script started. Leaving it alone."
    return 0
  fi
  echo "Stopping IndexTTS (pid ${pid})."
  kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  local _
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.2
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 -- "-$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
}

open_browser() {
  echo "App: ${APP_URL}"
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$APP_URL" >/dev/null 2>&1 || true
  fi
}

# Host folder bind-mounted at /vault. Compose must not create it: a missing
# path becomes root:root and the container user cannot write the export.
vault_host_dir() {
  local dir tilde rest
  dir="$(resolve VAULT_HOST_DIR)"
  # resolve() can hand back the compose default with HOME still unexpanded.
  # shellcheck disable=SC2016
  if [[ -z "$dir" || "$dir" == *'${HOME}'* ]]; then
    dir="${HOME}/NLP-Vault"
  fi
  tilde='~'
  rest="${dir#"$tilde"}"
  if [[ "$rest" != "$dir" ]]; then
    dir="${HOME}${rest}"
  fi
  printf '%s' "$dir"
}

ensure_vault_host_dir() {
  local dir
  dir="$(vault_host_dir)"
  if [[ -z "$dir" || "$dir" != /* ]]; then
    echo "Vault directory must be an absolute path (got: ${dir}). Set VAULT_HOST_DIR." >&2
    return 1
  fi
  if [[ -e "$dir" && ! -d "$dir" ]]; then
    echo "Vault path exists and is not a directory: ${dir}" >&2
    return 1
  fi
  mkdir -p -- "$dir"
}

cmd_up() {
  local build=0 tts=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --build) build=1 ;;
      --tts) tts=1 ;;
      *)
        die "Unknown option: $1
Usage: ./workstation up [--build] [--tts]"
        ;;
    esac
    shift
  done
  require_docker
  ensure_vault_host_dir || fail "Could not create the Obsidian vault directory."
  echo "Starting the workstation…"
  local -a args=(up -d)
  if [[ "$build" -eq 1 ]]; then
    args+=(--build)
  fi
  if ! compose "${args[@]}"; then
    fail "The stack did not start. See why with: ./workstation logs"
  fi
  if [[ "$build" -eq 0 ]]; then
    warn_if_image_stale
  fi
  if [[ "$tts" -eq 1 ]]; then
    start_indextts_if_needed
  else
    local tts_port
    tts_port="$(port_from_url "$(host_tts_url)")"
    if port_open "$tts_port"; then
      echo "IndexTTS is already listening on port ${tts_port}."
    else
      echo "IndexTTS was not started by this command. Start it with: ./workstation up --tts"
    fi
  fi
  if ! wait_until "the app" "$WEBUI_WAIT_SECONDS" webui_ready; then
    fail "See why with: ./workstation logs webui"
  fi
  if [[ "$tts" -eq 1 ]]; then
    local tts_url
    tts_url="$(host_tts_url)"
    if ! wait_until "IndexTTS" "$TTS_WAIT_SECONDS" indextts_ready "$tts_url"; then
      fail "See why with: ./workstation logs tts"
    fi
  fi
  open_browser
}

cmd_down() {
  require_docker
  stop_indextts_if_ours
  echo "Stopping containers. Database and uploads stay on disk."
  if ! compose stop; then
    fail "Could not stop the containers. See why with: ./workstation logs"
  fi
}

row() {
  local level="$1" name="$2" state="$3" detail="$4"
  local colour="$C_OK"
  if [[ "$level" == "bad" ]]; then
    colour="$C_BAD"
  elif [[ "$level" == "warn" ]]; then
    colour="$C_WARN"
  fi
  printf '%s%-16s%s  %s%-4s%s  %s\n' "$C_BOLD" "$name" "$C_OFF" "$colour" "$state" "$C_OFF" "$detail"
}

fix_line() {
  printf '  %sfix:%s %s\n' "$C_WARN" "$C_OFF" "$1"
}

ps_line() {
  local svc="$1"
  compose ps -a --format '{{.State}}|{{.ExitCode}}|{{.Status}}' "$svc" 2>/dev/null | head -n1 || true
}

schedule_enabled() {
  local value
  value="$(resolve SCHEDULE_ENABLED)"
  value="${value,,}"
  case "$value" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

status_webui() {
  if webui_ready; then
    row ok "Web UI" "OK" "$WEBUI_HEALTH_URL"
  else
    row bad "Web UI" "DOWN" "no answer from ${WEBUI_HEALTH_URL}"
    fix_line "./workstation up"
  fi
}

status_postgres() {
  local line state _exit status_text
  if ! docker info >/dev/null 2>&1; then
    row bad "Postgres" "DOWN" "Docker is not running"
    fix_line "Start Docker, then ./workstation up"
    return
  fi
  line="$(ps_line db)"
  state="${line%%|*}"
  status_text="${line#*|*|}"
  if [[ "$status_text" == *"unhealthy"* || "$status_text" == *"starting"* ]]; then
    row bad "Postgres" "DOWN" "${status_text:-not healthy}"
    fix_line "./workstation logs db"
  elif [[ "$status_text" == *"healthy"* || "$state" == "running" ]]; then
    row ok "Postgres" "OK" "${status_text:-running}"
  else
    row bad "Postgres" "DOWN" "${status_text:-not running}"
    fix_line "./workstation up"
  fi
}

status_postgrest() {
  local body
  body="$(http_get "$POSTGREST_URL_HOST")"
  if [[ -n "$body" ]]; then
    row ok "PostgREST" "OK" "$POSTGREST_URL_HOST"
  else
    row bad "PostgREST" "DOWN" "no answer from ${POSTGREST_URL_HOST}"
    fix_line "./workstation up"
  fi
}

status_searxng() {
  local body
  body="$(http_get "$SEARXNG_HEALTH_URL")"
  if [[ "$body" == *"OK"* ]]; then
    row ok "SearXNG" "OK" "$SEARXNG_HEALTH_URL"
  else
    row bad "SearXNG" "DOWN" "no answer from ${SEARXNG_HEALTH_URL}"
    fix_line "./workstation logs searxng"
  fi
}

status_indextts() {
  local url port
  url="$(host_tts_url)"
  port="$(port_from_url "$url")"
  if indextts_ready "$url"; then
    row ok "IndexTTS" "OK" "ready on port ${port}"
  else
    row bad "IndexTTS" "DOWN" "opt-in, not running (uses the GPU)"
    fix_line "./workstation up --tts"
  fi
}

status_qdrant_probe() {
  local label url key body base down_hint
  label="$1"
  url="$2"
  key="$3"
  down_hint="$4"
  if [[ "$url" != http://* && "$url" != https://* ]]; then
    row bad "$label" "DOWN" "not configured"
    return
  fi
  base="${url%/}"
  if [[ -n "$key" ]]; then
    body="$(curl -sS --max-time 8 --header "api-key: ${key}" "${base}/" 2>/dev/null || true)"
  else
    body="$(http_get "${base}/")"
  fi
  if [[ "$body" == \{* ]]; then
    row ok "$label" "OK" "reachable"
  elif [[ "$body" == *"404 page not found"* ]]; then
    row bad "$label" "DOWN" "suspended free-tier cluster (plain-text 404)"
    fix_line "Resume the cluster in the Qdrant Cloud console"
  else
    row bad "$label" "DOWN" "could not reach ${base}/"
    fix_line "$down_hint"
  fi
}

status_qdrant() {
  local cloud key
  # Published host port of the compose `qdrant` service. Inside the
  # containers QDRANT_URL is http://qdrant:6333; this probe is from the host.
  status_qdrant_probe "Qdrant" "http://127.0.0.1:6333" "" "./workstation up"
  cloud="$(resolve QDRANT_CLOUD_URL)"
  if [[ -z "$cloud" ]]; then
    row ok "Qdrant Cloud" "OK" "not configured"
    return
  fi
  key="$(resolve QDRANT_CLOUD_API_KEY)"
  status_qdrant_probe "Qdrant Cloud" "$cloud" "$key" "Check QDRANT_CLOUD_URL in .env"
}

status_scheduler() {
  local line state exit_code status_text
  if ! docker info >/dev/null 2>&1; then
    row bad "Scheduler" "DOWN" "Docker is not running"
    fix_line "Start Docker, then ./workstation up"
    return
  fi
  line="$(ps_line scheduler)"
  if [[ -z "$line" ]]; then
    row bad "Scheduler" "DOWN" "no scheduler container"
    fix_line "./workstation up"
    return
  fi
  state="${line%%|*}"
  exit_code="${line#*|}"
  exit_code="${exit_code%%|*}"
  status_text="${line#*|*|}"
  if [[ "$state" == "running" ]]; then
    row ok "Scheduler" "OK" "${status_text:-running}"
  elif [[ "$state" == "exited" && "$exit_code" == "0" ]]; then
    if schedule_enabled; then
      row bad "Scheduler" "DOWN" "Exited (0) while scheduling is turned on"
      fix_line "./workstation logs scheduler"
    else
      row ok "Scheduler" "OK" "disabled — Exited (0) is normal"
    fi
  else
    row bad "Scheduler" "DOWN" "${status_text:-exited ${exit_code}}"
    fix_line "./workstation logs scheduler"
  fi
}

# No host port to probe: the runner is only on the internal playground
# network. Its container healthcheck calls /health from inside.
status_playground_runner() {
  local line state status_text
  if ! docker info >/dev/null 2>&1; then
    row bad "Playground" "DOWN" "Docker is not running"
    fix_line "Start Docker, then ./workstation up"
    return
  fi
  line="$(ps_line playground-runner)"
  if [[ -z "$line" ]]; then
    row bad "Playground" "DOWN" "no playground runner container"
    fix_line "./workstation up"
    return
  fi
  state="${line%%|*}"
  status_text="${line#*|*|}"
  if [[ "$state" == "running" && "$status_text" == *"(healthy)"* ]]; then
    row ok "Playground" "OK" "runner ${status_text}"
  elif [[ "$state" == "running" && "$status_text" == *"starting"* ]]; then
    row warn "Playground" "WAIT" "runner starting"
  else
    row bad "Playground" "DOWN" "runner ${status_text:-not running}"
    fix_line "./workstation logs playground-runner"
  fi
}

cmd_status() {
  echo "${C_BOLD}NLP workstation${C_OFF}"
  echo
  if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running. Start Docker, then: ./workstation up"
    echo
  fi
  status_webui
  status_postgres
  status_postgrest
  status_searxng
  status_indextts
  status_qdrant
  status_scheduler
  status_playground_runner
}

cmd_logs() {
  if [[ $# -gt 1 ]]; then
    die "logs takes at most one service name"
  fi
  local svc="${1:-}"
  if [[ "$svc" == "tts" ]]; then
    mkdir -p "$STATE_DIR"
    touch "$LOG_FILE"
    echo "Following ${LOG_FILE} (Ctrl-C to stop)"
    exec tail -n +1 -f "$LOG_FILE"
  fi
  require_docker
  # exec must be the docker binary. A shell function cannot be exec'd.
  if [[ -z "$svc" ]]; then
    exec docker compose --project-directory "$ROOT" logs -f
  fi
  exec docker compose --project-directory "$ROOT" logs -f "$svc"
}

cmd_install_desktop() {
  local dest="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
  local target="$dest/nlp-workstation.desktop"
  local template="$ROOT/scripts/nlp-workstation.desktop"
  local content exec_line
  [[ -f "$template" ]] || fail "Missing desktop template: ${template}"
  mkdir -p "$dest"
  if [[ "$ROOT" == *" "* ]]; then
    exec_line="Exec=\"${ROOT}/workstation\" up"
  else
    exec_line="Exec=${ROOT}/workstation up"
  fi
  content="$(<"$template")"
  content="${content//@EXEC@/$exec_line}"
  content="${content//@PATH@/Path=${ROOT}}"
  printf '%s\n' "$content" >"$target"
  chmod 644 "$target"
  if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$dest" >/dev/null 2>&1 || true
  fi
  echo "Installed ${target}"
  echo "It runs: ${ROOT}/workstation up"
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    ""|-h|--help|help)
      usage
      ;;
    up)
      shift
      cmd_up "$@"
      ;;
    down)
      shift
      [[ $# -eq 0 ]] || die "down takes no arguments"
      cmd_down
      ;;
    status)
      shift
      [[ $# -eq 0 ]] || die "status takes no arguments"
      cmd_status
      ;;
    logs)
      shift
      cmd_logs "$@"
      ;;
    install-desktop)
      shift
      [[ $# -eq 0 ]] || die "install-desktop takes no arguments"
      cmd_install_desktop
      ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
