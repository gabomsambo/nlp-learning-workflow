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

    # Podcast model routing — see nlp_pillars/podcast_models.py
    podcast_extraction_model: str = Field(
        "deepseek-v4-flash",
        env="PODCAST_EXTRACTION_MODEL",
    )
    podcast_synthesis_model: str = Field(
        "claude-sonnet-4-5-20250929",
        env="PODCAST_SYNTHESIS_MODEL",
    )

    # Discovery settings
    discovery_candidate_count: int = Field(10, env="DISCOVERY_CANDIDATE_COUNT")
    vector_search_top_k: int = Field(5, env="VECTOR_SEARCH_TOP_K")

    # Application Settings with defaults
    default_model: str = Field("gpt-4o", env="DEFAULT_MODEL")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
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

    # IndexTTS on the host GPU — see nlp_pillars/tts/indextts_client.py
    indextts_url: str = Field(
        "http://host.docker.internal:7861",
        env="INDEXTTS_URL",
    )
    indextts_start_command: str = Field(
        "cd /home/gabo/index-tts && uv run webui.py --host 0.0.0.0 --port 7861",
        env="INDEXTTS_START_COMMAND",
    )
    voices_dir: str = Field("/voices", env="VOICES_DIR")
    podcast_audio_dir: str = Field("/app/data/podcast_audio", env="PODCAST_AUDIO_DIR")
    tts_download_dir: str = Field("/app/data/tts-downloads", env="TTS_DOWNLOAD_DIR")
    tts_preview_dir: str = Field("/app/data/tts-previews", env="TTS_PREVIEW_DIR")

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
