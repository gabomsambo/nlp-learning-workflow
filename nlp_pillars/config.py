"""
Configuration management for NLP Learning Workflow.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import find_dotenv, load_dotenv


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys - Optional as CLI commands decide what's required
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    SUPABASE_URL: Optional[str] = Field(None, env="SUPABASE_URL")
    SUPABASE_KEY: Optional[str] = Field(None, env="SUPABASE_KEY")
    QDRANT_URL: Optional[str] = Field(None, env="QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = Field(None, env="QDRANT_API_KEY")
    SEARXNG_URL: Optional[str] = Field(None, env="SEARXNG_URL")
    
    # Application Settings with defaults
    EMBEDDING_MODEL: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env file


# Pillar configurations
PILLAR_CONFIGS = {
    "P1": {
        "name": "Linguistic & Cognitive Foundations",
        "goal": "Master core linguistic theory and cognitive alignment between humans and AI",
        "focus_areas": [
            "Morphology and Syntax",
            "Semantics and Pragmatics",
            "Psycholinguistics",
            "Cognitive alignment with LLMs",
            "Formal language theory"
        ]
    },
    "P2": {
        "name": "Models & Architectures",
        "goal": "Understand cutting-edge model architectures and emerging paradigms",
        "focus_areas": [
            "Transformer variants",
            "Long-context models",
            "Multimodal architectures",
            "Neurosymbolic AI",
            "Emergent communication",
            "State space models"
        ]
    },
    "P3": {
        "name": "Data, Training & Methodologies",
        "goal": "Master data curation, training techniques, and multilingual challenges",
        "focus_areas": [
            "Data curation and annotation",
            "Low-resource languages",
            "RLHF and DPO",
            "Instruction tuning",
            "Synthetic data generation",
            "Linguistic typology"
        ]
    },
    "P4": {
        "name": "Evaluation & Interpretability",
        "goal": "Develop expertise in model evaluation, analysis, and interpretability",
        "focus_areas": [
            "Metrics and benchmarks",
            "Robustness testing",
            "Explainable AI",
            "Probing techniques",
            "Error analysis",
            "Out-of-distribution generalization"
        ]
    },
    "P5": {
        "name": "Ethics & Applications",
        "goal": "Understand ethical implications and real-world applications",
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
        load_dotenv(env_file)
        _env_path = Path(env_file)
    
    # Create settings instance
    _settings = Settings()
    
    return _settings
