"""
Utility functions for dynamic pillar operations.
Provides helpers for pillar config retrieval, abbreviation generation, and validation.
"""

import logging
from typing import Dict, Any, Optional

from .db import get_pillar_by_id, pillar_exists, get_pillars

logger = logging.getLogger(__name__)


def get_pillar_config(pillar_id: str) -> Dict[str, Any]:
    """
    Get pillar configuration from database.

    Args:
        pillar_id: The pillar slug (e.g., 'models-architectures')

    Returns:
        Dictionary with pillar configuration

    Raises:
        ValueError: If pillar not found
    """
    pillar = get_pillar_by_id(pillar_id)
    if not pillar:
        raise ValueError(f"Pillar not found: {pillar_id}")

    return {
        'id': pillar.id,
        'name': pillar.name,
        'goal': pillar.goal,
        'focus_areas': pillar.focus_areas,
        'papers_per_day': pillar.papers_per_day,
        'abbreviation': getattr(pillar, 'abbreviation', generate_abbreviation(pillar.name)),
        'created_at': pillar.created_at,
        'updated_at': pillar.updated_at,
        'last_active': pillar.last_active
    }


def generate_abbreviation(name: str, max_len: int = 7) -> str:
    """
    Generate a smart abbreviation from pillar name.

    Examples:
        "Linguistic & Cognitive Foundations" -> "LingCog"
        "Models & Architectures" -> "ModArch"
        "Data, Training & Methodologies" -> "DataTrn"
        "Machine Learning Basics" -> "MLBasic"

    Args:
        name: Full pillar name
        max_len: Maximum length of abbreviation (default 7)

    Returns:
        Abbreviated name suitable for compact display
    """
    if not name:
        return ""

    # Remove common words and punctuation
    stopwords = {'&', 'and', 'the', 'of', 'for', 'to', 'in', 'a', 'an'}
    words = [w for w in name.replace('&', '').replace(',', '').split()
             if w.lower() not in stopwords and len(w) > 1]

    if not words:
        return name[:max_len]

    if len(words) == 1:
        # Single word - take first max_len chars capitalized
        return words[0][:max_len].title()

    if len(words) == 2:
        # Two words - take first 4 chars of first, 3 of second
        abbr = words[0][:4].title() + words[1][:3].title()
        return abbr[:max_len]

    # Multiple words - take first 3-4 chars of first two significant words
    abbr = words[0][:4].title() + words[1][:3].title()
    return abbr[:max_len]


def validate_pillar_id(pillar_id: str) -> bool:
    """
    Check if a pillar ID is valid (exists in database).

    Args:
        pillar_id: The pillar slug to validate

    Returns:
        True if pillar exists, False otherwise
    """
    if not pillar_id:
        return False
    return pillar_exists(pillar_id)


def get_all_pillar_ids() -> list[str]:
    """
    Get all pillar IDs from the database.

    Returns:
        List of pillar ID strings
    """
    pillars = get_pillars(limit=100)
    return [p.id for p in pillars]


def get_pillar_choices() -> list[tuple[str, str]]:
    """
    Get pillar choices for dropdown menus.

    Returns:
        List of (id, name) tuples for use in select elements
    """
    pillars = get_pillars(limit=100)
    return [(p.id, p.name) for p in pillars]


def get_pillar_display_name(pillar_id: str, use_abbreviation: bool = False) -> str:
    """
    Get display name for a pillar.

    Args:
        pillar_id: The pillar slug
        use_abbreviation: If True, return abbreviated name

    Returns:
        Display name or abbreviation
    """
    pillar = get_pillar_by_id(pillar_id)
    if not pillar:
        return pillar_id  # Fallback to ID

    if use_abbreviation:
        abbr = getattr(pillar, 'abbreviation', None)
        if abbr:
            return abbr
        return generate_abbreviation(pillar.name)

    return pillar.name


# Legacy pillar ID mapping for backward compatibility
LEGACY_PILLAR_MAPPING = {
    'P1': 'linguistic-cognitive-foundations',
    'P2': 'models-architectures',
    'P3': 'data-training-methodologies',
    'P4': 'evaluation-interpretability',
    'P5': 'ethics-applications',
}

REVERSE_LEGACY_MAPPING = {v: k for k, v in LEGACY_PILLAR_MAPPING.items()}


def normalize_pillar_id(pillar_id: str) -> str:
    """
    Normalize a pillar ID, converting legacy P1-P5 to slugs if needed.

    Args:
        pillar_id: Either legacy (P1, P2, etc.) or new slug format

    Returns:
        Normalized pillar slug
    """
    if pillar_id in LEGACY_PILLAR_MAPPING:
        return LEGACY_PILLAR_MAPPING[pillar_id]
    return pillar_id


def get_legacy_id(pillar_id: str) -> Optional[str]:
    """
    Get legacy P1-P5 ID for a pillar slug (if applicable).

    Args:
        pillar_id: The pillar slug

    Returns:
        Legacy ID (P1, P2, etc.) or None if not a legacy pillar
    """
    return REVERSE_LEGACY_MAPPING.get(pillar_id)
