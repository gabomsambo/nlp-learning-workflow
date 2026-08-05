#!/usr/bin/env python3
"""Script to create pillars in Supabase database."""

import os
import sys
from pathlib import Path

# Load .env from the project root
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import httpx

# Resolve the REST endpoint the same way the application does
# (webui/services/postgrest_client.py), so this script follows the database
# wherever the app is pointed: self-hosted PostgREST first, hosted Supabase
# only as a fallback.
SUPABASE_URL = os.getenv("POSTGREST_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: POSTGREST_URL (or SUPABASE_URL) and SUPABASE_KEY must be set in .env")
    sys.exit(1)

print(f"REST URL: {SUPABASE_URL}")
print(f"Key present: {bool(SUPABASE_KEY)}")

# Define the 8 pillars
pillars = [
    {
        "id": "formal-linguistics-nlp",
        "name": "Formal Linguistics for NLP",
        "goal": "Master the linguistic theories that underpin modern NLP systems, from syntax to semantics to pragmatics",
        "papers_per_day": 2,
        "focus_areas": [
            "Generative syntax and constituency parsing",
            "Formal semantics and compositional meaning",
            "Pragmatics and discourse analysis",
            "Morphological theory and word formation",
            "Phonology for speech processing",
            "Type-theoretic approaches to meaning"
        ]
    },
    {
        "id": "neural-architectures-language",
        "name": "Neural Architectures for Language",
        "goal": "Understand the mathematical and architectural foundations of neural networks used in NLP",
        "papers_per_day": 2,
        "focus_areas": [
            "Transformer architecture deep dive",
            "Attention mechanisms and variants",
            "Recurrent and convolutional approaches",
            "Pretraining objectives (MLM, CLM, T5-style)",
            "Efficient transformers and long-context models",
            "State space models (Mamba, RWKV)"
        ]
    },
    {
        "id": "llm-theory-practice",
        "name": "LLM Theory & Practice",
        "goal": "Develop deep expertise in how large language models work, their capabilities, and limitations",
        "papers_per_day": 2,
        "focus_areas": [
            "Scaling laws and emergent abilities",
            "In-context learning mechanisms",
            "RLHF and preference optimization (DPO, PPO)",
            "Prompt engineering and chain-of-thought",
            "Instruction tuning and alignment",
            "Hallucination and factuality"
        ]
    },
    {
        "id": "computational-semantics",
        "name": "Computational Semantics & Meaning",
        "goal": "Bridge formal semantics with distributional and neural approaches to meaning representation",
        "papers_per_day": 2,
        "focus_areas": [
            "Distributional semantics and word embeddings",
            "Compositional distributional models",
            "Knowledge graphs and symbolic reasoning",
            "Semantic role labeling and AMR parsing",
            "Textual entailment and inference",
            "Grounding language in perception"
        ]
    },
    {
        "id": "model-interpretability",
        "name": "Model Interpretability & Probing",
        "goal": "Learn to analyze what neural models learn and develop tools for understanding model behavior",
        "papers_per_day": 2,
        "focus_areas": [
            "Probing classifiers and linguistic analysis",
            "Attention visualization and interpretation",
            "Mechanistic interpretability",
            "Behavioral testing and challenge sets",
            "Bias detection and measurement",
            "Robustness and adversarial analysis"
        ]
    },
    {
        "id": "ai-agents-tool-use",
        "name": "AI Agents & Autonomous Systems",
        "goal": "Master the emerging field of LLM-powered agents that can reason, plan, and use tools",
        "papers_per_day": 2,
        "focus_areas": [
            "ReAct and chain-of-thought reasoning",
            "Tool use and function calling",
            "Multi-agent systems and collaboration",
            "Memory and retrieval augmented generation",
            "Planning and task decomposition",
            "Agent evaluation and benchmarks"
        ]
    },
    {
        "id": "ml-systems-production",
        "name": "ML Systems & Production AI",
        "goal": "Understand how to build, deploy, and scale machine learning systems in production",
        "papers_per_day": 2,
        "focus_areas": [
            "Distributed training (DeepSpeed, FSDP)",
            "Model serving and inference optimization",
            "Quantization and model compression",
            "MLOps and experiment tracking",
            "GPU programming and CUDA basics",
            "Cloud ML platforms (AWS, GCP, Azure)"
        ]
    },
    {
        "id": "ai-safety-alignment",
        "name": "AI Safety & Responsible AI",
        "goal": "Understand the challenges of building safe, aligned, and beneficial AI systems",
        "papers_per_day": 2,
        "focus_areas": [
            "Constitutional AI and RLHF",
            "Jailbreaking and red teaming",
            "Truthfulness and calibration",
            "Value alignment approaches",
            "Governance and AI policy",
            "Societal impact and ethics"
        ]
    }
]

# Insert pillars using Supabase REST API
headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

base_url = SUPABASE_URL.rstrip("/")

# Hosted Supabase serves PostgREST under /rest/v1; a self-hosted PostgREST
# serves it at the root. Same conditional as nlp_pillars/db.py:28 and
# webui/services/postgrest_client.py, so the path is no longer hardcoded.
if "supabase.co" in base_url and "/rest/v1" not in base_url:
    base_url = f"{base_url}/rest/v1"

# First, let's delete old pillars to start fresh
print("\nClearing old pillars...")
resp = httpx.delete(f"{base_url}/pillars?id=neq.NONEXISTENT", headers=headers)
print(f"Delete response: {resp.status_code}")

# Insert new pillars
print("\nInserting new pillars...")
for pillar in pillars:
    resp = httpx.post(f"{base_url}/pillars", json=pillar, headers=headers)
    if resp.status_code in [200, 201]:
        print(f"✅ Created: {pillar['name']}")
    else:
        print(f"❌ Failed: {pillar['name']} - {resp.status_code} - {resp.text[:200]}")

print("\n✅ All pillars created! Check your WebUI at http://localhost:8000/pillars/")

