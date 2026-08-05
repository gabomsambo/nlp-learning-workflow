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

    # Anthropic API for podcast generation
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")

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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env file


# Default pillar configurations (fallback for fresh installs)
# Keys are now slugs matching database IDs
PILLAR_CONFIGS = {
    "linguistic-cognitive-foundations": {
        "name": "Linguistic & Cognitive Foundations",
        "goal": "Master core linguistic theory and cognitive alignment between humans and AI",
        "abbreviation": "LingCog",
        "focus_areas": [
            "Morphology and Syntax",
            "Semantics and Pragmatics",
            "Psycholinguistics",
            "Cognitive alignment with LLMs",
            "Formal language theory"
        ]
    },
    "models-architectures": {
        "name": "Models & Architectures",
        "goal": "Understand cutting-edge model architectures and emerging paradigms",
        "abbreviation": "ModArch",
        "focus_areas": [
            "Transformer variants",
            "Long-context models",
            "Multimodal architectures",
            "Neurosymbolic AI",
            "Emergent communication",
            "State space models"
        ]
    },
    "data-training-methodologies": {
        "name": "Data, Training & Methodologies",
        "goal": "Master data curation, training techniques, and multilingual challenges",
        "abbreviation": "DataTrn",
        "focus_areas": [
            "Data curation and annotation",
            "Low-resource languages",
            "RLHF and DPO",
            "Instruction tuning",
            "Synthetic data generation",
            "Linguistic typology"
        ]
    },
    "evaluation-interpretability": {
        "name": "Evaluation & Interpretability",
        "goal": "Develop expertise in model evaluation, analysis, and interpretability",
        "abbreviation": "EvalInt",
        "focus_areas": [
            "Metrics and benchmarks",
            "Robustness testing",
            "Explainable AI",
            "Probing techniques",
            "Error analysis",
            "Out-of-distribution generalization"
        ]
    },
    "ethics-applications": {
        "name": "Ethics & Applications",
        "goal": "Understand ethical implications and real-world applications",
        "abbreviation": "EthApp",
        "focus_areas": [
            "Bias detection and mitigation",
            "Healthcare applications",
            "Legal and policy frameworks",
            "Educational technology",
            "Cultural preservation",
            "Linguistic justice"
        ]
    }
}

# Legacy ID mapping for backward compatibility
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
