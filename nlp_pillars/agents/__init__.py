"""
Agents module for NLP Learning Workflow.
"""

# Expose only the summarizer. It is a lazy proxy: importing this package pulls in
# instructor/openai but constructs no client and needs no API key - the OpenAI
# client is built on the first .run() call. See summarizer_agent._LazySummarizerAgent.
from .summarizer_agent import SummarizerAgent

__all__ = ["SummarizerAgent"]
