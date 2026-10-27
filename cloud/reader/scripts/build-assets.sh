#!/bin/sh
# Copy the app's look into public/static/ so the reader and the app cannot drift.
# public/static/ is generated and gitignored; run this before `wrangler dev` or deploy.
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
src="$here/../../webui/static"
dest="$here/public/static"
rm -rf "$dest"
mkdir -p "$dest/icons"
cp "$src/tokens.css" "$src/app.css" "$dest/"
# The pure Listen helpers the episode page shares with the desktop Listen page.
cp "$src/factcheck.js" "$src/readalong.js" "$src/listen-core.js" "$dest/"
# The Playground tab's renderer (sanitized output, code lines, notes) and the
# Listen page's time -> span -> card lookup, read-only on the reader.
cp "$src/playground-core.js" "$src/playground.js" "$src/follow-along.js" "$dest/"
cp -R "$src/fonts" "$dest/fonts"
cp "$src/icons/apple-touch-icon.png" "$src/icons/icon-192.png" "$src/icons/icon-512.png" "$dest/icons/"
echo "public/static/ built from webui/static ($(du -sk "$dest" | cut -f1) KB)"
