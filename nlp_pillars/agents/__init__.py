"""
Agents module for NLP Learning Workflow.
"""

# Expose only the lightweight summarizer that does not require external LLM deps
from .summarizer_agent import SummarizerAgent

__all__ = ["SummarizerAgent"]
