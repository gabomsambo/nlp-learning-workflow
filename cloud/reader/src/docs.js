/*
 * Reads published documents out of R2. Only the keys `cloud export` writes
 * are reachable: a name outside this list is a 404 whatever the bucket holds.
 * Audio is streamed by audio.js, never through here.
 */

const DOC_KEY = new RegExp(
  '^(?:manifest|library|schedule)\\.json$'
  + '|^papers/[A-Za-z0-9._-]{1,200}\\.json$'
  + '|^(?:episodes|playgrounds)/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\\.json$',
);

export function isDocKey(key) {
  return typeof key === 'string' && !key.includes('..') && DOC_KEY.test(key);
}

const JSON_HEADERS = {
  'content-type': 'application/json; charset=utf-8',
  'cache-control': 'private, no-cache',
};

function json(status, body) {
  return new Response(JSON.stringify(body), { status, headers: JSON_HEADERS });
}

/**
 * Serve one document with its ETag. `If-None-Match` is handed to R2 as a
 * precondition, so an unchanged document answers 304 without a body.
 */
export async function serveDoc(bucket, key, request) {
  if (!isDocKey(key)) return json(404, { error: 'not found' });
  const object = await bucket.get(key, { onlyIf: request.headers });
  if (object === null) {
    return json(404, { error: key === 'manifest.json' ? 'nothing published yet' : 'not found' });
  }
  const headers = new Headers(JSON_HEADERS);
  headers.set('etag', object.httpEtag);
  if (object.body === undefined || object.body === null) {
    // R2 returns the object without a body when If-None-Match matched.
    return new Response(null, { status: 304, headers });
  }
  headers.set('content-length', String(object.size));
  return new Response(request.method === 'HEAD' ? null : object.body, { status: 200, headers });
}
