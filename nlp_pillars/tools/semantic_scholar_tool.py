"""
Semantic Scholar Tool - search, paper details, references and citations.

Every call in the process goes through one shared pacer (``RequestPacer``):
calls queue behind it instead of failing, a 429 backs off and retries a
bounded number of times, and a burst of bulk work cannot starve an
interactive request. All of it is in-process; there is nothing to deploy.
"""

import hashlib
import heapq
import itertools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Sequence

import httpx

from ..schemas import PaperRef
from ..config import get_settings


logger = logging.getLogger(__name__)

# Logged once, at error, when a configured key is rejected. The key itself
# is never written. .env is not changed; renewing or removing it is the
# captain's call.
KEY_REJECTED_LOG = (
    "SEMANTIC_SCHOLAR_API_KEY was rejected by Semantic Scholar (HTTP 401/403). "
    "Running without a key for the rest of this process. "
    "The key in .env was not changed; renew or remove it."
)

#: Seconds between requests. Semantic Scholar documents 1000 requests/second
#: shared by every unauthenticated caller, "further throttled during heavy
#: use", and 1 request/second for an individual key. Keyless stays under that
#: on purpose because the shared pool is what throttles first.
KEYLESS_INTERVAL_SECONDS = 1.25
KEYED_INTERVAL_SECONDS = 1.0

#: Tries per call for 429, 5xx and transport failures.
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 2.0
#: A Retry-After longer than this is not waited out on a worker thread. The
#: call fails as "temporarily unavailable" instead.
MAX_RETRY_AFTER_SECONDS = 30.0

#: Callers waiting for a slot. Bulk callers get a smaller share so a burst
#: cannot fill the queue and refuse an interactive request.
MAX_WAITING = 64
MAX_BULK_WAITING = 32

INTERACTIVE = 0
BULK = 1

UNAVAILABLE_MESSAGE = (
    "Semantic Scholar citation data is temporarily unavailable "
    "(rate limited) - try again in a few minutes"
)

PAPER_FIELDS = (
    "paperId,externalIds,title,authors,year,citationCount,"
    "abstract,influentialCitationCount,openAccessPdf"
)
LINK_FIELDS = "paperId,externalIds,title,authors,year,citationCount,abstract"
#: S2's page cap for references/citations and its batch cap for /paper/batch.
PAGE_LIMIT = 100
BATCH_LIMIT = 500


class SemanticScholarError(Exception):
    """A Semantic Scholar call failed. An empty citation list is not this.

    A 404 is "this paper is not in the index" and stays an empty list.
    This exception is everything else, so a 403 or a 429 cannot be reported
    as zero citations.
    """


class SemanticScholarUnavailableError(SemanticScholarError):
    """Rate limited past the retry budget, or the queue is full. Try later."""


def _public_reason(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    if " for url " in text:
        text = text.split(" for url ", 1)[0]
    return text[:180]


class RequestPacer:
    """One request slot at a time, ``interval`` seconds apart, best priority first.

    ``acquire`` blocks until the caller holds the next slot. Waiters are
    ordered by (priority, arrival), so an interactive caller overtakes queued
    bulk callers. ``clock`` and ``sleep`` are injectable so pacing is testable
    without waiting.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_waiting: int = MAX_WAITING,
        max_bulk_waiting: int = MAX_BULK_WAITING,
    ):
        self._clock = clock
        self._sleep = sleep
        self.max_waiting = max_waiting
        self.max_bulk_waiting = max_bulk_waiting
        self._cond = threading.Condition(threading.Lock())
        self._waiting: list = []
        self._seq = itertools.count()
        self._next_at = 0.0

    def waiting(self) -> int:
        with self._cond:
            return len(self._waiting)

    def penalize(self, seconds: float) -> None:
        """Hold every caller back, not just the one that was rate limited."""
        with self._cond:
            self._next_at = max(self._next_at, self._clock() + seconds)

    def sleep(self, seconds: float) -> None:
        self._sleep(seconds)

    def acquire(self, priority: int, interval: float) -> None:
        with self._cond:
            bulk = sum(1 for p, _ in self._waiting if p == BULK)
            if len(self._waiting) >= self.max_waiting or (
                priority == BULK and bulk >= self.max_bulk_waiting
            ):
                raise SemanticScholarUnavailableError(
                    "Semantic Scholar citation data is temporarily unavailable "
                    "(too many requests queued) - try again in a few minutes"
                )
            ticket = (priority, next(self._seq))
            heapq.heappush(self._waiting, ticket)
            try:
                while True:
                    if self._waiting[0] == ticket:
                        wait = self._next_at - self._clock()
                        if wait <= 0:
                            heapq.heappop(self._waiting)
                            self._next_at = self._clock() + interval
                            self._cond.notify_all()
                            return
                        self._cond.release()
                        try:
                            self._sleep(wait)
                        finally:
                            self._cond.acquire()
                    else:
                        self._cond.wait()
            except BaseException:
                if ticket in self._waiting:
                    self._waiting.remove(ticket)
                    heapq.heapify(self._waiting)
                self._cond.notify_all()
                raise


_PACER = RequestPacer()
_REJECTED_KEYS: set = set()
_STATE_LOCK = threading.Lock()


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def reset_shared_state() -> None:
    """Forget rejected keys and queued state. For tests."""
    global _PACER
    with _STATE_LOCK:
        _REJECTED_KEYS.clear()
    _PACER = RequestPacer()


@dataclass
class LinkPage:
    """One page of references or citations.

    ``next_offset`` is None on the last page; pass it back as ``offset``.
    """

    items: List[tuple] = field(default_factory=list)
    next_offset: Optional[int] = None


class SemanticScholarTool:
    """Semantic Scholar client. Rate limiting is shared by every instance."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        priority: int = INTERACTIVE,
        pacer: Optional[RequestPacer] = None,
    ):
        """
        Args:
            api_key: Optional API key. Defaults to SEMANTIC_SCHOLAR_API_KEY.
            priority: ``INTERACTIVE`` (default) or ``BULK``. Bulk callers
                queue behind interactive ones.
            pacer: Test hook. Defaults to the process-wide pacer.
        """
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.priority = priority
        self._pacer = pacer

        key = (api_key or "").strip()
        if not key:
            settings = get_settings()
            key = (getattr(settings, "semantic_scholar_api_key", None) or "").strip()
        self._api_key = key or None

        self.client = httpx.Client(timeout=30.0)

        logger.info("SemanticScholarTool initialized")

    # -- key and pacing ---------------------------------------------------

    @property
    def key_rejected(self) -> bool:
        return bool(self._api_key) and _digest(self._api_key) in _REJECTED_KEYS

    def _active_key(self) -> Optional[str]:
        return None if self.key_rejected else self._api_key

    @property
    def headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        key = self._active_key()
        if key:
            headers["x-api-key"] = key
        return headers

    @property
    def pacer(self) -> RequestPacer:
        return self._pacer or _PACER

    def _reject_configured_key(self) -> None:
        """Treat a key Semantic Scholar refused as absent. Log once. Do not edit .env."""
        if not self._api_key:
            return
        with _STATE_LOCK:
            first = _digest(self._api_key) not in _REJECTED_KEYS
            _REJECTED_KEYS.add(_digest(self._api_key))
        if first:
            logger.error(KEY_REJECTED_LOG)

    @staticmethod
    def _retry_delay(response: Optional[httpx.Response], attempt: int) -> float:
        expo = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
        if response is not None:
            raw = response.headers.get("Retry-After")
            try:
                if raw is not None:
                    return max(float(raw), 0.0)
            except ValueError:
                pass
        return expo

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
    ) -> httpx.Response:
        """One logical call: queue for a slot, send, retry what is retryable.

        Returns the final response for the caller to judge (404 handling,
        ``raise_for_status``). Raises ``SemanticScholarUnavailableError``
        when 429 outlasts the retry budget or the queue is full, and
        ``SemanticScholarError`` when 5xx or the transport does.
        """
        pacer = self.pacer
        attempt = 0
        while True:
            key = self._active_key()
            pacer.acquire(
                self.priority, KEYED_INTERVAL_SECONDS if key else KEYLESS_INTERVAL_SECONDS
            )
            response: Optional[httpx.Response] = None
            try:
                response = self.client.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=self.headers,
                )
            except httpx.TransportError as exc:
                attempt += 1
                if attempt >= MAX_ATTEMPTS:
                    raise SemanticScholarError(_public_reason(exc)) from exc
                pacer.penalize(self._retry_delay(None, attempt))
                continue

            status = response.status_code
            if status in (401, 403) and key:
                self._reject_configured_key()
                continue  # now keyless; does not use a retry
            if status == 429:
                attempt += 1
                delay = self._retry_delay(response, attempt)
                logger.warning(
                    "Semantic Scholar rate limited (attempt %d/%d), backing off %.1fs",
                    attempt, MAX_ATTEMPTS, delay,
                )
                if attempt >= MAX_ATTEMPTS or delay > MAX_RETRY_AFTER_SECONDS:
                    raise SemanticScholarUnavailableError(UNAVAILABLE_MESSAGE)
                pacer.penalize(delay)
                continue
            if status >= 500:
                attempt += 1
                if attempt >= MAX_ATTEMPTS:
                    raise SemanticScholarError(f"Semantic Scholar answered HTTP {status}")
                pacer.penalize(self._retry_delay(response, attempt))
                continue
            return response

    def _json(self, response: httpx.Response) -> dict:
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise SemanticScholarError(_public_reason(exc)) from exc
        except ValueError as exc:
            raise SemanticScholarError("Semantic Scholar returned invalid JSON") from exc

    def _map_paper_id(self, paper_id: str) -> str:
        """
        Map our paper ID to Semantic Scholar format.

        Args:
            paper_id: Paper ID in our format (e.g., "2301.00001" for arXiv)

        Returns:
            Semantic Scholar paper ID format
        """
        # If it looks like an arXiv ID (numbers with dots)
        if paper_id and '.' in paper_id and paper_id.replace('.', '').replace('v', '').isdigit():
            return f"arXiv:{paper_id.split('v')[0]}"

        # If it looks like a DOI
        if paper_id and paper_id.startswith('10.'):
            return f"DOI:{paper_id}"

        # Otherwise return as-is (might be S2 corpus ID)
        return paper_id

    # -- public API -------------------------------------------------------

    def search(self, query: str, limit: int = 10, year: Optional[str] = None) -> List[PaperRef]:
        """Search for papers by keyword. Raises ``SemanticScholarError`` on failure."""
        params = {
            "query": query,
            "fields": PAPER_FIELDS,
            "limit": min(limit, PAGE_LIMIT),
        }
        if year:
            params["year"] = year

        logger.info(f"Searching Semantic Scholar for: {query}")
        data = self._json(self._request("GET", f"{self.base_url}/paper/search", params=params))

        papers = []
        for paper_data in data.get("data") or []:
            try:
                papers.append(self._convert_to_paper_ref(paper_data))
            except Exception as e:
                logger.warning(f"Failed to convert S2 result: {e}")
        logger.info(f"Found {len(papers)} papers from Semantic Scholar")
        return papers

    def get_paper(self, paper_id: str) -> Optional[PaperRef]:
        """One paper by id. None means Semantic Scholar does not have it;
        any other failure raises ``SemanticScholarError``."""
        s2_id = self._map_paper_id(paper_id)
        response = self._request(
            "GET", f"{self.base_url}/paper/{s2_id}", params={"fields": PAPER_FIELDS}
        )
        if response.status_code == 404:
            return None
        return self._convert_to_paper_ref(self._json(response))

    def get_papers_batch(self, paper_ids: Sequence[str]) -> List[Optional[PaperRef]]:
        """Details for many papers, one POST per 500 ids.

        The result is aligned with ``paper_ids``; an id S2 does not know is None.
        """
        ids = list(paper_ids)
        out: List[Optional[PaperRef]] = []
        for start in range(0, len(ids), BATCH_LIMIT):
            chunk = [self._map_paper_id(i) for i in ids[start:start + BATCH_LIMIT]]
            data = self._json(
                self._request(
                    "POST",
                    f"{self.base_url}/paper/batch",
                    params={"fields": PAPER_FIELDS},
                    json_body={"ids": chunk},
                )
            )
            if not isinstance(data, list) or len(data) != len(chunk):
                raise SemanticScholarError("Semantic Scholar batch answer did not match the request")
            out.extend(self._convert_to_paper_ref(d) if d else None for d in data)
        return out

    def get_references_page(
        self, paper_id: str, offset: int = 0, limit: int = PAGE_LIMIT
    ) -> LinkPage:
        """One page of papers this paper cites."""
        return self._link_page(paper_id, "references", "citedPaper", offset, limit)

    def get_citations_page(
        self, paper_id: str, offset: int = 0, limit: int = PAGE_LIMIT
    ) -> LinkPage:
        """One page of papers that cite this paper."""
        return self._link_page(paper_id, "citations", "citingPaper", offset, limit)

    def get_references(self, paper_id: str, limit: int = 50) -> List[tuple[PaperRef, bool]]:
        """Outgoing references as ``(PaperRef, is_influential)``, up to ``limit``."""
        return list(self.iter_references(paper_id, max_items=limit))

    def get_citations(self, paper_id: str, limit: int = 50) -> List[tuple[PaperRef, bool]]:
        """Incoming citations as ``(PaperRef, is_influential)``, up to ``limit``."""
        return list(self.iter_citations(paper_id, max_items=limit))

    def iter_references(self, paper_id: str, max_items: Optional[int] = None) -> Iterator[tuple]:
        return self._iter_links(self.get_references_page, paper_id, max_items)

    def iter_citations(self, paper_id: str, max_items: Optional[int] = None) -> Iterator[tuple]:
        return self._iter_links(self.get_citations_page, paper_id, max_items)

    @staticmethod
    def _iter_links(fetch_page, paper_id: str, max_items: Optional[int]) -> Iterator[tuple]:
        offset, yielded = 0, 0
        while max_items is None or yielded < max_items:
            want = PAGE_LIMIT if max_items is None else min(PAGE_LIMIT, max_items - yielded)
            page = fetch_page(paper_id, offset=offset, limit=want)
            for item in page.items:
                yield item
                yielded += 1
                if max_items is not None and yielded >= max_items:
                    return
            if page.next_offset is None:
                return
            offset = page.next_offset

    def _link_page(
        self, paper_id: str, kind: str, wrapper: str, offset: int, limit: int
    ) -> LinkPage:
        s2_id = self._map_paper_id(paper_id)
        response = self._request(
            "GET",
            f"{self.base_url}/paper/{s2_id}/{kind}",
            params={
                "fields": LINK_FIELDS,
                "limit": max(1, min(limit, PAGE_LIMIT)),
                "offset": max(0, offset),
            },
        )
        if response.status_code == 404:
            logger.warning(f"Paper not found in Semantic Scholar: {paper_id}")
            return LinkPage()
        data = self._json(response)

        items = []
        for row in data.get("data") or []:
            try:
                other = row.get(wrapper) or {}
                if other.get("paperId"):
                    items.append((self._convert_to_paper_ref(other), row.get("isInfluential", False)))
            except Exception as e:
                logger.warning(f"Failed to convert {kind} entry: {e}")
        logger.info(f"Found {len(items)} {kind} for paper {paper_id}")
        return LinkPage(items=items, next_offset=data.get("next"))

    def _convert_to_paper_ref(self, s2_paper: dict) -> PaperRef:
        """
        Convert Semantic Scholar paper data to PaperRef.

        Args:
            s2_paper: Paper data from S2 API

        Returns:
            PaperRef object
        """
        # Extract best paper ID (prefer arXiv, then DOI, then S2 corpus ID)
        external_ids = s2_paper.get("externalIds", {}) or {}

        paper_id = None
        if external_ids.get("ArXiv"):
            paper_id = external_ids["ArXiv"]
        elif external_ids.get("DOI"):
            paper_id = external_ids["DOI"]
        else:
            paper_id = s2_paper.get("paperId", "unknown")

        # Extract authors
        authors = []
        for author in s2_paper.get("authors", []):
            if author and author.get("name"):
                authors.append(author["name"])

        # Extract PDF URL
        pdf_url = None
        open_access = s2_paper.get("openAccessPdf")
        if open_access and isinstance(open_access, dict):
            pdf_url = open_access.get("url")
        elif external_ids.get("ArXiv"):
            # Construct arXiv PDF URL
            arxiv_id = external_ids["ArXiv"]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        return PaperRef(
            id=paper_id,
            title=s2_paper.get("title", "Unknown Title"),
            authors=authors,
            year=s2_paper.get("year"),
            venue=None,  # S2 doesn't return venue in basic fields
            url_pdf=pdf_url,
            abstract=s2_paper.get("abstract"),
            citation_count=s2_paper.get("citationCount", 0)
        )

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tool = SemanticScholarTool()

    # Example search
    papers = tool.search("transformer attention mechanism", limit=5)

    for paper in papers:
        print(f"Title: {paper.title}")
        print(f"Authors: {', '.join(paper.authors[:3])}")
        print(f"Year: {paper.year}")
        print(f"Citations: {paper.citation_count}")
        print("-" * 50)
