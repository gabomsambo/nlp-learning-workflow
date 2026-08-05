-- Migration 008: carry the PDF URL on the paper_queue row.
--
-- Why: a queue row used to store only (paper_id, title, source), and source was
-- hardcoded to 'arxiv' for every candidate. pop_from_paper_queue therefore had
-- to rebuild the download URL as https://arxiv.org/pdf/<paper_id>.pdf, which is
-- correct only when paper_id really is an arXiv id. For anything else — a DOI,
-- or the fabricated `searxng_<hash>` ids SearXNG's HTML fallback used to emit —
-- it produced a URL that 404s, and the paper failed at ingest several stages
-- after the point where the identifier went wrong.
--
-- With this column the queue carries the URL the candidate was discovered with,
-- so a non-arXiv paper survives the round trip and add_to_paper_queue can
-- refuse anything that still has no downloadable URL.
--
-- Safe to run against a populated database: the column is nullable and existing
-- rows keep working — a NULL url_pdf with an arXiv-shaped paper_id still
-- resolves via the arXiv URL, exactly as before.

ALTER TABLE paper_queue ADD COLUMN IF NOT EXISTS url_pdf TEXT;

COMMENT ON COLUMN paper_queue.url_pdf IS
    'PDF URL the candidate was discovered with. NULL is allowed only for '
    'arXiv-id rows, where the URL is reconstructed from paper_id.';

-- PostgREST caches the schema; without this it answers inserts naming the new
-- column with PGRST204 "Could not find the ''url_pdf'' column".
NOTIFY pgrst, 'reload schema';
