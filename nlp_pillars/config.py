"""
Configuration management for NLP Learning Workflow.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import find_dotenv, load_dotenv


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys - Optional as CLI commands decide what's required
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    supabase_url: Optional[str] = Field(None, env="SUPABASE_URL")
    supabase_key: Optional[str] = Field(None, env="SUPABASE_KEY")
    qdrant_url: Optional[str] = Field(None, env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")
    # Optional second projection. Empty means the cloud target stays unconfigured.
    qdrant_cloud_url: Optional[str] = Field(None, env="QDRANT_CLOUD_URL")
    qdrant_cloud_api_key: Optional[str] = Field(None, env="QDRANT_CLOUD_API_KEY")
    searxng_url: Optional[str] = Field(None, env="SEARXNG_URL")

    # Semantic Scholar API (optional, for higher rate limits)
    semantic_scholar_api_key: Optional[str] = Field(None, env="SEMANTIC_SCHOLAR_API_KEY")

    # Anthropic API for podcast synthesis (call 5)
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")

    # DeepSeek API for Ground Pack extraction (calls 1–4)
    deepseek_api_key: Optional[str] = Field(None, env="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        "https://api.deepseek.com",
        env="DEEPSEEK_BASE_URL",
    )

    # Generation routing — see nlp_pillars/llm_routes.py, which holds the
    # defaults and the fallback chain. Every one of these is None on purpose:
    # the route's own default applies unless an env var is set, so a new
    # stage's model can be changed in one place (the registry) rather than in
    # two (here and the registry).
    llm_summarize_model: Optional[str] = Field(None, env="LLM_SUMMARIZE_MODEL")
    llm_synthesize_model: Optional[str] = Field(None, env="LLM_SYNTHESIZE_MODEL")
    llm_quiz_model: Optional[str] = Field(None, env="LLM_QUIZ_MODEL")
    llm_discover_model: Optional[str] = Field(None, env="LLM_DISCOVER_MODEL")
    llm_concepts_model: Optional[str] = Field(None, env="LLM_CONCEPTS_MODEL")
    llm_podcast_extraction_model: Optional[str] = Field(
        None, env="LLM_PODCAST_EXTRACTION_MODEL"
    )
    llm_podcast_synthesis_model: Optional[str] = Field(
        None, env="LLM_PODCAST_SYNTHESIS_MODEL"
    )
    llm_factcheck_extract_model: Optional[str] = Field(
        None, env="LLM_FACTCHECK_EXTRACT_MODEL"
    )
    llm_factcheck_screen_model: Optional[str] = Field(
        None, env="LLM_FACTCHECK_SCREEN_MODEL"
    )
    llm_factcheck_rejudge_model: Optional[str] = Field(
        None, env="LLM_FACTCHECK_REJUDGE_MODEL"
    )
    llm_table_repair_model: Optional[str] = Field(None, env="LLM_TABLE_REPAIR_MODEL")
    # Evidence-writer ledger builder (nlp_pillars/evidence/ledger.py).
    llm_podcast_ledger_model: Optional[str] = Field(None, env="LLM_PODCAST_LEDGER_MODEL")
    # How the podcast script is written: "free" (Ground Pack + paper, the
    # original writer) or "evidence" (a code-verified evidence ledger, every
    # line cites it). Stays "free" until the evidence writer passes its gate;
    # see docs/PODCAST_EVIDENCE_WRITER.md. Read through get_writer_mode(), which
    # refuses a value that is neither.
    podcast_writer_mode: str = Field("free", env="PODCAST_WRITER_MODE")
    # Page-image table repair at ingestion (nlp_pillars/table_repair.py).
    table_repair_enabled: bool = Field(True, env="TABLE_REPAIR_ENABLED")
    llm_playground_model: Optional[str] = Field(None, env="LLM_PLAYGROUND_MODEL")
    llm_playground_align_model: Optional[str] = Field(None, env="LLM_PLAYGROUND_ALIGN_MODEL")
    llm_playground_diff_model: Optional[str] = Field(
        None, env="LLM_PLAYGROUND_DIFF_MODEL"
    )
    # Paper-map knowledge layer (nlp_pillars/map_enrich.py,
    # nlp_pillars/concept_anchoring.py). MAP_ENRICHMENT_ENABLED is the switch
    # for ALL of its paid calls: community names, bridge sentences and the
    # model's pick among vocabulary candidates. Off, the map and the anchoring
    # still run on templates and exact matches, and nothing is billed.
    map_enrichment_enabled: bool = Field(True, env="MAP_ENRICHMENT_ENABLED")
    llm_map_names_model: Optional[str] = Field(None, env="LLM_MAP_NAMES_MODEL")
    llm_map_bridge_model: Optional[str] = Field(None, env="LLM_MAP_BRIDGE_MODEL")
    llm_concept_anchor_model: Optional[str] = Field(None, env="LLM_CONCEPT_ANCHOR_MODEL")
    # Extra vocabulary files beside the bundled NLP-KG one, comma separated
    # (nlp_pillars/vocabulary.py), e.g. a local Papers with Code conversion.
    vocabulary_files: Optional[str] = Field(None, env="VOCABULARY_FILES")

    # Legacy routing names, still honoured and consulted second. They used to
    # carry the defaults themselves; the registry does now.
    podcast_extraction_model: Optional[str] = Field(
        None, env="PODCAST_EXTRACTION_MODEL",
    )
    podcast_synthesis_model: Optional[str] = Field(
        None, env="PODCAST_SYNTHESIS_MODEL",
    )
    # "anthropic" (the default) or "deepseek". Left unset, a DeepSeek model id
    # in PODCAST_EXTRACTION_MODEL selects the DeepSeek path, and anything else
    # keeps it on Claude.
    podcast_extraction_provider: Optional[str] = Field(
        None, env="PODCAST_EXTRACTION_PROVIDER",
    )

    # Fact-check routes. See nlp_pillars/llm_routes.py. Haiku screens every
    # claim; Sonnet re-judges only what the screen flagged. No temperature.
    factcheck_extract_model: Optional[str] = Field(
        None, env="FACTCHECK_EXTRACT_MODEL",
    )
    factcheck_screen_model: Optional[str] = Field(
        None, env="FACTCHECK_SCREEN_MODEL",
    )
    factcheck_rejudge_model: Optional[str] = Field(
        None, env="FACTCHECK_REJUDGE_MODEL",
    )
    # Start a fact-check run after a podcast script is saved. A failed check
    # does not fail the script run and does not change the script. Set false
    # to leave checks to POST /api/podcast/{id}/factcheck and the CLI.
    factcheck_auto_dispatch: bool = Field(True, env="FACTCHECK_AUTO_DISPATCH")

    # Discovery settings
    discovery_candidate_count: int = Field(10, env="DISCOVERY_CANDIDATE_COUNT")
    vector_search_top_k: int = Field(5, env="VECTOR_SEARCH_TOP_K")

    # Application Settings with defaults
    #
    # DEFAULT_MODEL is retired. It named the OpenAI model the summarizer,
    # lesson and quiz agents used, and that account ran out of credit on
    # 2026-10-25. Generation routing lives in nlp_pillars/llm_routes.py now;
    # this value is kept only so a stale .env line does not break startup,
    # and nothing generation-related reads it.
    default_model: str = Field("gpt-4o", env="DEFAULT_MODEL")
    # Postgres chunk embeddings, and the vectors the projector copies into
    # Qdrant. Default is the pinned local bge model. EMBEDDING_PROVIDER=openai
    # stores OpenAI vectors and projects those. A text-embedding-* value of
    # EMBEDDING_MODEL with the local provider does not switch the store.
    embedding_provider: str = Field("local", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field("BAAI/bge-base-en-v1.5", env="EMBEDDING_MODEL")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Daily-run scheduler. Consumed by nlp_pillars/scheduler.py, which runs as the
    # `scheduler` service in docker-compose.yml. These three keys have been in .env
    # since before the scheduler existed; until it did, `extra = "ignore"` below
    # silently dropped them.
    schedule_enabled: bool = Field(False, env="SCHEDULE_ENABLED")
    schedule_time: str = Field("08:00", env="SCHEDULE_TIME")  # "HH:MM", 24-hour
    schedule_timezone: str = Field("UTC", env="SCHEDULE_TIMEZONE")  # IANA name
    # Papers per pillar per scheduled run. Default 1 matches what the (now
    # schedule-disabled) GitHub Action passed as `--papers 1`. The per-pillar
    # `papers_per_day` column in the database is deliberately not consulted, so
    # one knob controls the whole run's API spend.
    papers_per_day: int = Field(1, env="PAPERS_PER_DAY", ge=1)

    # Weekly paper-map refresh (nlp_pillars/map_refresh.py). It is registered by
    # the same scheduler process, so it only ever runs when SCHEDULE_ENABLED is
    # true; this flag lets a scheduler that is on skip it. Day is an APScheduler
    # day_of_week ("sun", "mon", "mon-fri"); time is HH:MM in SCHEDULE_TIMEZONE.
    weekly_map_refresh_enabled: bool = Field(True, env="WEEKLY_MAP_REFRESH_ENABLED")
    weekly_map_refresh_day: str = Field("sun", env="WEEKLY_MAP_REFRESH_DAY")
    weekly_map_refresh_time: str = Field("03:00", env="WEEKLY_MAP_REFRESH_TIME")

    # Obsidian vault. Off unless VAULT_EXPORT_ENABLED is true: the scheduler
    # then rewrites VAULT_DIR after the daily run. The CLI writes that
    # directory on demand either way. Blank VAULT_DIR means ~/NLP-Vault.
    # Compose sets VAULT_DIR=/vault and bind-mounts the host folder there.
    vault_export_enabled: bool = Field(False, env="VAULT_EXPORT_ENABLED")
    vault_dir: str = Field("~/NLP-Vault", env="VAULT_DIR")

    # IndexTTS on the host GPU — see nlp_pillars/tts/indextts_client.py
    indextts_url: str = Field(
        "http://host.docker.internal:7860",
        env="INDEXTTS_URL",
    )
    indextts_start_command: str = Field(
        "cd /home/gabo/index-tts && uv run webui.py --host 0.0.0.0 --port 7860",
        env="INDEXTTS_START_COMMAND",
    )
    voices_dir: str = Field("/voices", env="VOICES_DIR")

    # Locked-down Python runner for paper playgrounds (docs/PLAYGROUND_RUNNER.md).
    # Reachable only over compose's internal `playground` network.
    playground_runner_url: str = Field(
        "http://playground-runner:8765",
        env="PLAYGROUND_RUNNER_URL",
    )
    # Cloud reader publishing (nlp_pillars/cloud/publish.py). Off unless
    # CLOUD_PUBLISH_ENABLED is true; the three below must then all be set. The
    # client id and secret are the Access service token that may reach /ingest
    # and nothing else. The bundle directory keeps the transcoded audio between
    # runs, so it belongs on the uploads volume.
    cloud_publish_enabled: bool = Field(False, env="CLOUD_PUBLISH_ENABLED")
    cloud_publish_url: Optional[str] = Field(None, env="CLOUD_PUBLISH_URL")
    cf_access_client_id: Optional[str] = Field(None, env="CF_ACCESS_CLIENT_ID")
    cf_access_client_secret: Optional[str] = Field(None, env="CF_ACCESS_CLIENT_SECRET")
    cloud_publish_max_bytes: int = Field(5_000_000_000, env="CLOUD_PUBLISH_MAX_BYTES", gt=0)
    cloud_bundle_dir: str = Field("/app/data/cloud-bundle", env="CLOUD_BUNDLE_DIR")
    # The reader's read-only copy of each playground (cloud/export.py): bytes of
    # base64 figure data, and of every other saved output as JSON, kept per
    # playground. What does not fit is left out and marked where it was.
    cloud_playground_max_image_bytes: int = Field(3_000_000, env="CLOUD_PLAYGROUND_MAX_IMAGE_BYTES", ge=0)
    cloud_playground_max_output_bytes: int = Field(1_500_000, env="CLOUD_PLAYGROUND_MAX_OUTPUT_BYTES", ge=0)
    podcast_audio_dir: str = Field("/app/data/podcast_audio", env="PODCAST_AUDIO_DIR")
    tts_download_dir: str = Field("/app/data/tts-downloads", env="TTS_DOWNLOAD_DIR")
    tts_preview_dir: str = Field("/app/data/tts-previews", env="TTS_PREVIEW_DIR")

    # Paper Playground (docs/PLAYGROUND.md). The data directory is the
    # playground_data volume; each playground caches its datasets under
    # <dir>/<playground_id>/ with a manifest.json the sandbox reads.
    playground_data_dir: str = Field("/app/playground-data", env="PLAYGROUND_DATA_DIR")
    # Total bytes of downloaded datasets per playground (default 1 GB).
    playground_data_max_bytes: int = Field(
        1_000_000_000, env="PLAYGROUND_DATA_MAX_BYTES", gt=0,
    )
    # Largest .ipynb accepted for import (default 25 MB).
    playground_notebook_max_bytes: int = Field(
        25_000_000, env="PLAYGROUND_NOTEBOOK_MAX_BYTES", gt=0,
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env file


# Default pillar configurations, used by get_pillar_config() below only when the
# database lookup fails. Keys are slugs matching database IDs.
#
# These MUST stay in step with the pillars `create_pillars.py` seeds — the eight
# below are the authoritative set. When this dict disagreed with the database it
# produced wrong pillar names and focus areas under database failure rather than
# a clean error, which is harder to notice than an outage.
PILLAR_CONFIGS = {
    "formal-linguistics-nlp": {
        "name": "Formal Linguistics for NLP",
        "goal": "Master the linguistic theories that underpin modern NLP systems, from syntax to semantics to pragmatics",
        "abbreviation": "FormLin",
        "focus_areas": [
            "Generative syntax and constituency parsing",
            "Formal semantics and compositional meaning",
            "Pragmatics and discourse analysis",
            "Morphological theory and word formation",
            "Phonology for speech processing",
            "Type-theoretic approaches to meaning"
        ]
    },
    "neural-architectures-language": {
        "name": "Neural Architectures for Language",
        "goal": "Understand the mathematical and architectural foundations of neural networks used in NLP",
        "abbreviation": "NeurArc",
        "focus_areas": [
            "Transformer architecture deep dive",
            "Attention mechanisms and variants",
            "Recurrent and convolutional approaches",
            "Pretraining objectives (MLM, CLM, T5-style)",
            "Efficient transformers and long-context models",
            "State space models (Mamba, RWKV)"
        ]
    },
    "llm-theory-practice": {
        "name": "LLM Theory & Practice",
        "goal": "Develop deep expertise in how large language models work, their capabilities, and limitations",
        "abbreviation": "LLMThe",
        "focus_areas": [
            "Scaling laws and emergent abilities",
            "In-context learning mechanisms",
            "RLHF and preference optimization (DPO, PPO)",
            "Prompt engineering and chain-of-thought",
            "Instruction tuning and alignment",
            "Hallucination and factuality"
        ]
    },
    "computational-semantics": {
        "name": "Computational Semantics & Meaning",
        "goal": "Bridge formal semantics with distributional and neural approaches to meaning representation",
        "abbreviation": "CompSem",
        "focus_areas": [
            "Distributional semantics and word embeddings",
            "Compositional distributional models",
            "Knowledge graphs and symbolic reasoning",
            "Semantic role labeling and AMR parsing",
            "Textual entailment and inference",
            "Grounding language in perception"
        ]
    },
    "model-interpretability": {
        "name": "Model Interpretability & Probing",
        "goal": "Learn to analyze what neural models learn and develop tools for understanding model behavior",
        "abbreviation": "ModeInt",
        "focus_areas": [
            "Probing classifiers and linguistic analysis",
            "Attention visualization and interpretation",
            "Mechanistic interpretability",
            "Behavioral testing and challenge sets",
            "Bias detection and measurement",
            "Robustness and adversarial analysis"
        ]
    },
    "ai-agents-tool-use": {
        "name": "AI Agents & Autonomous Systems",
        "goal": "Master the emerging field of LLM-powered agents that can reason, plan, and use tools",
        "abbreviation": "AIAge",
        "focus_areas": [
            "ReAct and chain-of-thought reasoning",
            "Tool use and function calling",
            "Multi-agent systems and collaboration",
            "Memory and retrieval augmented generation",
            "Planning and task decomposition",
            "Agent evaluation and benchmarks"
        ]
    },
    "ml-systems-production": {
        "name": "ML Systems & Production AI",
        "goal": "Understand how to build, deploy, and scale machine learning systems in production",
        "abbreviation": "MLSys",
        "focus_areas": [
            "Distributed training (DeepSpeed, FSDP)",
            "Model serving and inference optimization",
            "Quantization and model compression",
            "MLOps and experiment tracking",
            "GPU programming and CUDA basics",
            "Cloud ML platforms (AWS, GCP, Azure)"
        ]
    },
    "ai-safety-alignment": {
        "name": "AI Safety & Responsible AI",
        "goal": "Understand the challenges of building safe, aligned, and beneficial AI systems",
        "abbreviation": "AISaf",
        "focus_areas": [
            "Constitutional AI and RLHF",
            "Jailbreaking and red teaming",
            "Truthfulness and calibration",
            "Value alignment approaches",
            "Governance and AI policy",
            "Societal impact and ethics"
        ]
    }
}

# Legacy ID mapping for backward compatibility.
#
# STALE, and deliberately left in place rather than removed or re-pointed: these
# P1-P5 IDs predate the slug migration and their targets are not among the eight
# pillars above, so get_pillar_config("P1") now raises "Pillar not found" instead
# of silently returning a retired pillar. Re-pointing them at five of the current
# eight would be an invention, not a migration.
# The same mapping is duplicated in pillar_utils.LEGACY_PILLAR_MAPPING.
LEGACY_TO_SLUG = {
    "P1": "linguistic-cognitive-foundations",
    "P2": "models-architectures",
    "P3": "data-training-methodologies",
    "P4": "evaluation-interpretability",
    "P5": "ethics-applications",
}


def get_pillar_config(pillar_id: str) -> dict:
    """
    Get pillar configuration, preferring database over static config.

    Args:
        pillar_id: Pillar slug or legacy ID (P1-P5)

    Returns:
        Pillar configuration dict

    Raises:
        ValueError: If pillar not found
    """
    # Normalize legacy IDs
    if pillar_id in LEGACY_TO_SLUG:
        pillar_id = LEGACY_TO_SLUG[pillar_id]

    # Try database first
    try:
        from .db import get_pillar_by_id
        pillar = get_pillar_by_id(pillar_id)
        if pillar:
            return {
                'id': pillar.id,
                'name': pillar.name,
                'goal': pillar.goal,
                'focus_areas': pillar.focus_areas,
                'papers_per_day': pillar.papers_per_day,
                'abbreviation': getattr(pillar, 'abbreviation', ''),
            }
    except Exception:
        pass  # Fall through to static config

    # Fallback to static config
    if pillar_id in PILLAR_CONFIGS:
        config = PILLAR_CONFIGS[pillar_id].copy()
        config['id'] = pillar_id
        config['papers_per_day'] = 2
        return config

    raise ValueError(f"Pillar not found: {pillar_id}")


# Module-level cache for settings
_settings: Optional[Settings] = None
_env_path: Optional[Path] = None


def env_loaded_path() -> Optional[Path]:
    """Return the path to the loaded .env file, if any."""
    return _env_path


def get_settings() -> Settings:
    """Get application settings singleton."""
    global _settings, _env_path

    if _settings is not None:
        return _settings

    # Find and load .env file if it exists
    env_file = find_dotenv()
    if env_file:
        load_dotenv(env_file, override=True)
        _env_path = Path(env_file)

    # Create settings instance
    _settings = Settings()

    return _settings
