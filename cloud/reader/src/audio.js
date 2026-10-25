/*
 * Streams a published episode out of R2 with byte ranges.
 *
 * Safari will not play, and cannot seek in, an <audio> source unless the
 * server answers `Range` with 206 and says `Accept-Ranges: bytes`. R2 does the
 * slicing; this file only turns the request header into an offset and a length
 * and the answer into the right status line. Only the names `cloud export`
 * gives audio are reachable: `audio/<sha256 of the bytes>.mp3`.
 */

const AUDIO_KEY = /^audio\/[0-9a-f]{64}\.mp3$/;

export function isAudioKey(key) {
  return typeof key === 'string' && AUDIO_KEY.test(key);
}

/**
 * One byte range of an object `size` bytes long, from a `Range` header.
 *
 * Returns `null` when the header should be ignored (absent, not `bytes`, more
 * than one range, or not parseable: the server then sends the whole file, which
 * RFC 9110 allows), `'unsatisfiable'` when it names bytes past the end, and
 * `{ start, end }` (both inclusive, clamped to the object) otherwise.
 */
export function parseRange(header, size) {
  if (typeof header !== 'string') return null;
  const match = /^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$/i.exec(header);
  if (!match) return null;
  const [, first, last] = match;
  if (first === '' && last === '') return null;
  if (first === '') {
    // A suffix: the last N bytes.
    const count = Number(last);
    if (!Number.isSafeInteger(count)) return null;
    if (count === 0 || size === 0) return 'unsatisfiable';
    return { start: Math.max(0, size - count), end: size - 1 };
  }
  const start = Number(first);
  if (!Number.isSafeInteger(start)) return null;
  if (start >= size) return 'unsatisfiable';
  let end = last === '' ? size - 1 : Number(last);
  if (!Number.isSafeInteger(end)) return null;
  if (end < start) return null; // an invalid range is ignored, not refused
  end = Math.min(end, size - 1);
  return { start, end };
}

const matchesEtag = (header, etag) =>
  typeof header === 'string' && header.split(',').some((part) => {
    const tag = part.trim();
    return tag === '*' || tag.replace(/^W\//, '') === etag;
  });

const AUDIO_HEADERS = {
  'accept-ranges': 'bytes',
  'content-type': 'audio/mpeg',
  // The name is the hash of the bytes, so the bytes behind it never change.
  'cache-control': 'private, max-age=31536000, immutable',
};

function problem(status, message, extra = {}) {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', ...extra },
  });
}

export async function serveAudio(bucket, key, request) {
  if (!isAudioKey(key)) return problem(404, 'not found');
  const head = await bucket.head(key);
  if (head === null) return problem(404, 'not found');

  const etag = head.httpEtag;
  const headers = new Headers(AUDIO_HEADERS);
  headers.set('etag', etag);

  if (matchesEtag(request.headers.get('if-none-match'), etag)) return new Response(null, { status: 304, headers });

  // If-Range: only honour the range when the client still holds this exact object.
  const ifRange = request.headers.get('if-range');
  const range = ifRange && ifRange.trim() !== etag ? null : parseRange(request.headers.get('range'), head.size);

  if (range === 'unsatisfiable') {
    return problem(416, 'range not satisfiable', { 'content-range': `bytes */${head.size}`, 'accept-ranges': 'bytes' });
  }

  const wanted = range
    ? { status: 206, start: range.start, length: range.end - range.start + 1 }
    : { status: 200, start: 0, length: head.size };
  headers.set('content-length', String(wanted.length));
  if (range) headers.set('content-range', `bytes ${range.start}-${range.end}/${head.size}`);

  if (request.method === 'HEAD') return new Response(null, { status: wanted.status, headers });

  const object = await bucket.get(key, range ? { range: { offset: wanted.start, length: wanted.length } } : {});
  // Removed between head() and get(): the same answer a missing object gets.
  if (object === null || object.body === undefined || object.body === null) return problem(404, 'not found');
  return new Response(object.body, { status: wanted.status, headers });
}
