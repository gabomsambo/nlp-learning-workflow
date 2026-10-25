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
cp -R "$src/fonts" "$dest/fonts"
cp "$src/icons/apple-touch-icon.png" "$src/icons/icon-192.png" "$src/icons/icon-512.png" "$dest/icons/"
echo "public/static/ built from webui/static ($(du -sk "$dest" | cut -f1) KB)"
