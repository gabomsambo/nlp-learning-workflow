"""Embedding providers for chunk vectors stored in Postgres.

Two providers implement one interface: OpenAI ``text-embedding-3-small`` and local
``BAAI/bge-base-en-v1.5`` through fastembed's quantized ONNX export. That ONNX
file is the model the local-knowledge report measured. sentence-transformers
would load the fp32 PyTorch checkpoint instead, which is a heavier download and
does not produce these vectors. onnxruntime is already in the image, so the
image grows by fastembed and three small libraries, not by the weights.

Weights download into ``EMBEDDING_CACHE_DIR`` on first use (about 208 MB) and
are not copied into the image. Qdrant does not embed: ``vectors.upsert_text``
projects the rows this module stored.

Selection is ``EMBEDDING_PROVIDER`` (``local`` or ``openai``) and
``EMBEDDING_MODEL``. The default provider is ``local``. A value of
``EMBEDDING_MODEL`` that starts with ``text-embedding-`` does not replace bge
while the provider is local. Set ``EMBEDDING_PROVIDER=openai`` to store OpenAI
vectors in ``chunk_embeddings`` and project those.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, runtime_checkable

logger = logging.getLogger(__name__)

MIGRATION = "docs/migrations/018_chunk_embeddings.sql"

# Pinned ONNX export. fastembed 0.9.0 names this model BAAI/bge-base-en-v1.5
# and downloads Qdrant/bge-base-en-v1.5-onnx-Q (model_optimized.onnx, int8).
# The Hugging Face repo sha below is the revision those bytes came from.
# The upstream fp32 checkpoint is BAAI/bge-base-en-v1.5 @ a5beb1e3e68b9ab74eb54cfd186867f64f240e1a.
ONNX_REPO = "Qdrant/bge-base-en-v1.5-onnx-Q"
ONNX_REVISION = "199291fdd1aa89faf9c20b722dc72ad5e17aa0d0"
LOCAL_MODEL_NAME = "BAAI/bge-base-en-v1.5"
LOCAL_MODEL_ID = f"{LOCAL_MODEL_NAME}@{ONNX_REVISION}"
# huggingface.co API usedStorage for that revision, 2026-10-24.
FIRST_RUN_DOWNLOAD_BYTES = 217_824_172

# Model card: queries only. The trailing space is part of the instruction.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_DOCUMENT_PREFIX = ""

# CPU batches. OpenAI stays at 100, matching vectors.EMBED_BATCH_SIZE.
LOCAL_BATCH_SIZE = 32
OPENAI_BATCH_SIZE = 100
OPENAI_DEFAULT_MODEL = "text-embedding-3-small"
OPENAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_WEIGHT_FILES = ("model_optimized.onnx", "tokenizer.json")


class EmbeddingError(Exception):
    """Embeddings were not computed or not stored.

    ``written`` is how many rows of this attempt were stored before the
    failure. Callers put the message on the run stage. A missing table names
    the migration; swallowing the error is how that used to look finished.
    """

    def __init__(self, message: str, *, written: int = 0):
        super().__init__(message)
        self.written = written


class EmbeddingConfigError(EmbeddingError):
    """``EMBEDDING_PROVIDER`` / ``EMBEDDING_MODEL`` does not name a provider."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """What ingest, upload and the backfill CLI call. Nothing else."""

    model_id: str
    dims: int
    query_prefix: str
    document_prefix: str
    batch_size: int

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed passage text. One vector per input, input order."""

    def embed_query(self, text: str) -> List[float]:
        """Embed one query, including the provider's query prefix."""


def resolve_embedding_selection(
    provider: Optional[str],
    model: Optional[str],
) -> tuple[str, str]:
    """Return ``(provider name, stored model_id)``.

    ``provider`` defaults to ``local``. An OpenAI model id
    (``text-embedding-…``) with the local provider does not replace the pinned
    bge model. ``EMBEDDING_PROVIDER=openai`` is how OpenAI vectors are stored.
    """
    name = (provider or "").strip().lower() or "local"
    requested = (model or "").strip()

    if name == "openai":
        chosen = requested or OPENAI_DEFAULT_MODEL
        if chosen == LOCAL_MODEL_NAME or chosen == LOCAL_MODEL_ID or chosen.startswith("BAAI/"):
            chosen = OPENAI_DEFAULT_MODEL
        if chosen not in OPENAI_DIMS:
            known = ", ".join(sorted(OPENAI_DIMS))
            raise EmbeddingConfigError(
                f"OpenAI embeddings do not know model {chosen!r}. Known: {known}."
            )
        return "openai", chosen

    if name != "local":
        raise EmbeddingConfigError(
            f"Unknown EMBEDDING_PROVIDER {name!r}. Use 'local' or 'openai'."
        )

    if not requested or requested.startswith("text-embedding-"):
        return "local", LOCAL_MODEL_ID
    if requested in (LOCAL_MODEL_NAME, LOCAL_MODEL_ID):
        return "local", LOCAL_MODEL_ID
    raise EmbeddingConfigError(
        f"Local embeddings only support {LOCAL_MODEL_NAME} "
        f"(stored as {LOCAL_MODEL_ID}). Got {requested!r}."
    )


def selection_for_backfill(
    requested: Optional[str],
    provider: str,
    model: str,
) -> tuple[str, str]:
    """Resolve a CLI ``--model`` override, or the active selection."""
    if not requested:
        return resolve_embedding_selection(provider, model)
    if requested.startswith("text-embedding-"):
        return resolve_embedding_selection("openai", requested)
    return resolve_embedding_selection("local", requested)


def active_model_id() -> str:
    """Stored model id for the current config. Does not load weights."""
    from .config import get_settings

    settings = get_settings()
    _provider, model_id = resolve_embedding_selection(
        settings.embedding_provider,
        settings.embedding_model,
    )
    return model_id


def cache_dir() -> Path:
    """Where ONNX weights live. Created on use.

    ``EMBEDDING_CACHE_DIR`` when set (compose mounts this on the uploads
    volume). Otherwise ``~/.cache/nlp-embeddings``, which is outside the
    image and outside the git worktree.
    """
    raw = os.getenv("EMBEDDING_CACHE_DIR", "").strip()
    path = Path(raw) if raw else Path.home() / ".cache" / "nlp-embeddings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def local_model_dir(root: Path) -> Path:
    """Directory of the pinned ONNX snapshot inside ``root``."""
    return root / f"Qdrant-bge-base-en-v1.5-onnx-Q-{ONNX_REVISION}"


def _weights_ready(dest: Path) -> bool:
    return all(
        (dest / name).is_file() and (dest / name).stat().st_size > 0
        for name in _WEIGHT_FILES
    )


def ensure_local_model(root: Path) -> Path:
    """Download the pinned ONNX snapshot once. Later calls stay offline.

    The first download is about 208 MB (``FIRST_RUN_DOWNLOAD_BYTES``).
    """
    dest = local_model_dir(root)
    if _weights_ready(dest):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading %s@%s into %s (about %s MB, first run only)",
        ONNX_REPO,
        ONNX_REVISION,
        dest,
        FIRST_RUN_DOWNLOAD_BYTES // 1_000_000,
    )
    _snapshot_download(
        repo_id=ONNX_REPO,
        revision=ONNX_REVISION,
        local_dir=str(dest),
    )
    if not _weights_ready(dest):
        raise EmbeddingError(
            f"Download of {ONNX_REPO}@{ONNX_REVISION} into {dest} "
            f"did not produce {', '.join(_WEIGHT_FILES)}."
        )
    return dest


def _snapshot_download(**kwargs):
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def get_embedding_provider(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> EmbeddingProvider:
    """Build the configured provider. Local weights load on the first embed."""
    if provider is None or model is None:
        from .config import get_settings

        settings = get_settings()
        provider = settings.embedding_provider if provider is None else provider
        model = settings.embedding_model if model is None else model
    name, model_id = resolve_embedding_selection(provider, model)
    if name == "openai":
        return OpenAIEmbeddingProvider(model_id)
    return LocalBgeEmbeddingProvider()


def _vectors_in_input_order(model: str, count: int, data: Sequence) -> List[List[float]]:
    """Place each OpenAI embedding by its ``index``.

    The same contract as ``vectors._embed_batch``: arrival order is not the
    mapping, a short or duplicated response is not a partial success.
    """
    if len(data) != count:
        raise EmbeddingError(
            f"Embedding model {model} returned {len(data)} vectors for {count} inputs"
        )
    ordered: List[Optional[List[float]]] = [None] * count
    for position, item in enumerate(data):
        idx = getattr(item, "index", None)
        if idx is None:
            idx = position
        if not isinstance(idx, int) or not 0 <= idx < count:
            raise EmbeddingError(
                f"Embedding model {model} returned an out-of-range index {idx!r} "
                f"for a batch of {count}"
            )
        if ordered[idx] is not None:
            raise EmbeddingError(
                f"Embedding model {model} returned index {idx} twice in one batch"
            )
        ordered[idx] = [float(x) for x in item.embedding]
    missing = [i for i, value in enumerate(ordered) if value is None]
    if missing:
        raise EmbeddingError(
            f"Embedding model {model} returned no vector for input(s) {missing}"
        )
    return [value for value in ordered if value is not None]


class OpenAIEmbeddingProvider:
    """``text-embedding-3-small`` by default. Same batching contract as Qdrant.

    Query and document prefixes are empty: the OpenAI models were not trained
    with the bge instruction. ``embed_documents`` strips text, which is what
    ``vectors._embed_batch`` does.
    """

    model_id = OPENAI_DEFAULT_MODEL
    dims = OPENAI_DIMS[OPENAI_DEFAULT_MODEL]
    query_prefix = ""
    document_prefix = ""
    batch_size = OPENAI_BATCH_SIZE

    def __init__(self, model_id: str = OPENAI_DEFAULT_MODEL, *, client=None):
        if model_id not in OPENAI_DIMS:
            known = ", ".join(sorted(OPENAI_DIMS))
            raise EmbeddingConfigError(
                f"OpenAI embeddings do not know model {model_id!r}. Known: {known}."
            )
        self.model_id = model_id
        self.dims = OPENAI_DIMS[model_id]
        self._client = client

    def _openai(self):
        if self._client is not None:
            return self._client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EmbeddingError(
                "OpenAI client not configured. Set OPENAI_API_KEY environment variable."
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        return self._client

    def _embed(self, texts: Sequence[str], prefix: str) -> List[List[float]]:
        if not texts:
            return []
        prepared = [f"{prefix}{text.strip()}" for text in texts]
        try:
            response = self._openai().embeddings.create(
                model=self.model_id,
                input=prepared,
            )
            data = list(response.data)
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embeddings with model {self.model_id}: {e}"
            ) from e
        vectors = _vectors_in_input_order(self.model_id, len(texts), data)
        for vec in vectors:
            if len(vec) != self.dims:
                raise EmbeddingError(
                    f"Embedding model {self.model_id} returned {len(vec)} dims, "
                    f"expected {self.dims}"
                )
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed(texts, self.document_prefix)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], self.query_prefix)[0]


class LocalBgeEmbeddingProvider:
    """Pinned ``BAAI/bge-base-en-v1.5`` ONNX, 768-d, CPU only.

    Constructing the provider does not download weights. The first
    ``embed_documents`` / ``embed_query`` does, into ``cache_dir``.

    fastembed's ``query_embed`` does not add the bge instruction (its own
    description of this model says the prefix is "not so necessary"). The
    model card still specifies it for retrieval queries, and passages take
    no instruction. This class applies that prefix itself and calls the
    passage ``embed`` path, so the instruction is added once.
    """

    model_id = LOCAL_MODEL_ID
    dims = 768
    query_prefix = BGE_QUERY_PREFIX
    document_prefix = BGE_DOCUMENT_PREFIX
    batch_size = LOCAL_BATCH_SIZE

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = cache_dir
        self._onnx = None

    def _load(self):
        if self._onnx is not None:
            return self._onnx
        root = self._cache_dir if self._cache_dir is not None else cache_dir()
        model_path = ensure_local_model(root)
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise EmbeddingError(
                "fastembed is not installed; local embeddings need it. "
                "It is pinned in requirements.txt."
            ) from e
        # providers forces CPU. Device.AUTO would take a CUDA provider if one
        # appeared, and the host GPU is reserved for IndexTTS.
        self._onnx = TextEmbedding(
            model_name=LOCAL_MODEL_NAME,
            cache_dir=str(root),
            specific_model_path=str(model_path),
            providers=["CPUExecutionProvider"],
            cuda=False,
        )
        return self._onnx

    def _embed(self, texts: Sequence[str], prefix: str) -> List[List[float]]:
        if not texts:
            return []
        prepared = [f"{prefix}{text.strip()}" for text in texts]
        try:
            raw = self._load().embed(
                prepared,
                batch_size=self.batch_size,
                parallel=None,
            )
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embeddings with model {self.model_id}: {e}"
            ) from e
        vectors: List[List[float]] = []
        for item in raw:
            if hasattr(item, "tolist"):
                item = item.tolist()
            vectors.append([float(x) for x in item])
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Embedding model {self.model_id} returned {len(vectors)} vectors "
                f"for {len(texts)} inputs"
            )
        for vec in vectors:
            if len(vec) != self.dims:
                raise EmbeddingError(
                    f"Embedding model {self.model_id} returned {len(vec)} dims, "
                    f"expected {self.dims}"
                )
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed(texts, self.document_prefix)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], self.query_prefix)[0]
