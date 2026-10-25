// Manual publish of a `cloud export` bundle to R2. Text documents only: audio is
// left alone until the publisher exists. The manifest goes up last, so a reader
// never holds a manifest that names a document that is not there yet.
//
//   node cloud/reader/scripts/upload-bundle.mjs <bundle-dir> [--dry-run]
//
// Needs the personal Cloudflare login: XDG_CONFIG_HOME=$HOME/.config-cloudflare-account2
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { isDocKey } from '../src/docs.js';

const WRANGLER = 'wrangler@4.147.0';
const BUCKET = 'nlp-reader-docs';

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const dir = args.find((arg) => !arg.startsWith('--'));
if (!dir) {
  console.error('usage: node upload-bundle.mjs <bundle-dir> [--dry-run]');
  process.exit(2);
}
const root = resolve(dir);

const manifestBytes = readFileSync(join(root, 'manifest.json'));
const manifest = JSON.parse(manifestBytes);
if (manifest.version !== 1 || !Array.isArray(manifest.docs)) {
  console.error('manifest.json is not a version 1 bundle manifest');
  process.exit(1);
}

// Check every file against the manifest before anything leaves the machine.
const uploads = [];
for (const entry of manifest.docs) {
  if (!isDocKey(entry.key) || entry.key === 'manifest.json') {
    console.error(`refusing ${entry.key}: not a document the reader serves`);
    process.exit(1);
  }
  const bytes = readFileSync(join(root, entry.key));
  const sha = createHash('sha256').update(bytes).digest('hex');
  if (sha !== entry.sha256) {
    console.error(`refusing ${entry.key}: file hash ${sha.slice(0, 12)} does not match the manifest`);
    process.exit(1);
  }
  uploads.push(entry.key);
}
uploads.push('manifest.json');

const audio = (manifest.audio || []).length;
console.log(`${uploads.length - 1} documents and the manifest${dryRun ? ' (dry run)' : ''}; ${audio} audio files not uploaded`);

for (const key of uploads) {
  if (dryRun) {
    console.log(`  would put ${key}`);
    continue;
  }
  execFileSync('npx', [
    '--yes', WRANGLER, 'r2', 'object', 'put', `${BUCKET}/${key}`,
    '--file', join(root, key), '--content-type', 'application/json', '--remote',
  ], { stdio: ['ignore', 'ignore', 'inherit'] });
  console.log(`  put ${key}`);
}
console.log('done');
