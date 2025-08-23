# NLP Learning Workflow - Product Requirements Document

## Executive Summary
A self-updating, AI-powered learning system that automatically discovers, ingests, and synthesizes the latest NLP research across 5 independent learning pillars. The system generates daily lessons, quizzes, and optional podcast scripts while maintaining separate progress for each pillar.

## Problem Statement
Keeping up with NLP research is overwhelming:
- 100+ papers published daily on arXiv alone
- Information scattered across conferences, journals, and preprints
- No systematic way to build comprehensive understanding across subfields
- Difficult to maintain learning momentum across multiple topics
- Traditional reading groups don't scale to individual learning pace

## Solution Overview
An automated learning companion that:
1. **Discovers** relevant papers daily based on your learning goals
2. **Synthesizes** complex research into digestible lessons
3. **Tracks** progress independently across 5 core NLP pillars
4. **Adapts** to your learning pace and interests
5. **Generates** multiple learning formats (summaries, quizzes, podcasts)

## User Stories

### Core User Stories
1. **As a researcher**, I want to stay current with NLP advances without spending hours searching for papers
2. **As a practitioner**, I want to understand new techniques I can apply to real problems
3. **As a student**, I want structured learning paths through complex NLP topics
4. **As a busy professional**, I want to switch between learning topics without losing progress

### Detailed Scenarios

#### Scenario 1: Morning Learning Routine
- User wakes up to find today's lesson ready (automated overnight)
- Reviews 5-minute summary of latest paper in Pillar 2 (Models & Architectures)
- Takes quick quiz to reinforce yesterday's learning
- Saves interesting paper to "deep dive" queue for weekend

#### Scenario 2: Topic Switching
- Monday-Tuesday: Focus on Pillar 1 (Linguistic Foundations)
- Wednesday: Switch to Pillar 4 (Evaluation) for work project
- Thursday-Friday: Back to Pillar 1, system remembers exact progress
- Each pillar maintains its own context and learning history

#### Scenario 3: Knowledge Synthesis
- System identifies connections between papers across pillars
- Generates "bridge lessons" showing how concepts relate
- Creates knowledge graph visualization of learned concepts

## Core Features

### 1. Intelligent Paper Discovery
- **Multi-source search**: arXiv, Semantic Scholar, OpenAlex
- **Relevance ranking**: Based on citations, recency, and user goals
- **Deduplication**: By DOI/arXiv ID to avoid redundancy
- **Smart filtering**: Remove surveys/workshops when seeking novel research

### 2. Structured Knowledge Extraction
- **Consistent schemas**: Problem, Method, Findings, Limitations
- **Key term extraction**: Technical vocabulary with definitions
- **Citation network**: Related papers for deeper exploration
- **Code/dataset links**: Extracted from paper content

### 3. Multi-Format Learning Generation
- **Executive summaries**: 500-word overviews
- **Technical deep-dives**: Detailed methodology explanations
- **Practical takeaways**: "How to apply this" sections
- **Quiz generation**: Multiple difficulty levels
- **Podcast scripts**: Conversational explanations

### 4. Progress Tracking & Spaced Repetition
- **Per-pillar progress**: Papers read, concepts mastered
- **Spaced repetition**: SM-2 algorithm for quiz scheduling
- **Learning streaks**: Motivation through consistency tracking
- **Knowledge gaps**: Identifies areas needing reinforcement

### 5. Pillar Management System
- **Independent queues**: Each pillar has its own paper pipeline
- **Context isolation**: Switching pillars doesn't pollute context
- **Goal setting**: Define learning objectives per pillar
- **Progress dashboard**: Visual overview of all pillars

## Technical Requirements

### Architecture Components
1. **Agents** (Atomic Agents v2.0)
   - DiscoveryAgent: Query generation and search
   - IngestAgent: PDF parsing and text extraction
   - SummarizerAgent: Structured note creation
   - SynthesisAgent: Lesson and takeaway generation
   - QuizAgent: Question/answer creation
   - Orchestrator: Pipeline coordination

2. **Data Storage**
   - **Supabase**: Structured data (papers, notes, progress)
   - **Qdrant**: Vector embeddings (namespaced by pillar)
   - **File System**: PDF cache, generated content

3. **External Services**
   - **LLM Providers**: OpenAI, Anthropic, Groq (configurable)
   - **Search APIs**: arXiv, Semantic Scholar, OpenAlex
   - **Optional**: NotebookLM, ElevenLabs (TTS)

### Performance Requirements
- Daily processing: 1-3 papers per pillar (configurable)
- Response time: <30s per paper for full pipeline
- Storage: ~10MB per paper (PDF + embeddings + notes)
- API calls: ~50-100 per day (optimized with caching)

### Security & Privacy
- API keys in environment variables
- No storage of personal reading habits in external services
- Local option for sensitive research areas
- Rate limiting on all external API calls

## Success Metrics

### Quantitative Metrics
- Papers processed per week: Target 15-20
- Quiz retention rate: >80% correct on spaced repetition
- Learning streak: Average 5+ days/week
- Pillar coverage: All 5 pillars active monthly

### Qualitative Metrics
- Understanding depth: Can explain papers to others
- Practical application: Ideas implemented in projects
- Knowledge connections: Cross-pillar insights identified
- Research confidence: Comfortable reading latest papers

## Constraints & Assumptions

### Constraints
- API rate limits (OpenAI: 500 req/min, arXiv: 1 req/3s)
- Storage costs (Supabase free tier: 500MB)
- Processing costs (~$0.10-0.50 per paper with GPT-4o-mini)
- PDF parsing accuracy (some papers have complex layouts)

### Assumptions
- User has basic NLP knowledge (graduate level)
- Papers are in English
- PDFs are publicly accessible
- User reviews content at least 3x/week

## MVP Scope (Week 1)

### Must Have
- [ ] 5 pillars with independent progress tracking
- [ ] Basic paper discovery (arXiv only)
- [ ] PDF ingestion and summarization
- [ ] Simple CLI interface
- [ ] Local SQLite for testing (Supabase later)

### Nice to Have
- [ ] Quiz generation
- [ ] Spaced repetition
- [ ] Multiple LLM providers
- [ ] Web dashboard

### Out of Scope (Future)
- [ ] Mobile app
- [ ] Collaborative features
- [ ] Custom pillar creation
- [ ] Paper recommendation ML model

## Implementation Phases

### Phase 1: Foundation (Days 1-3)
- Set up project structure
- Implement basic agents
- Create Pydantic schemas
- Test with single pillar

### Phase 2: Multi-Pillar (Days 4-5)
- Add pillar management
- Implement context isolation
- Create orchestrator
- Add CLI commands

### Phase 3: Persistence (Days 6-7)
- Integrate Supabase
- Add Qdrant vector store
- Implement progress tracking
- Create status dashboard

### Phase 4: Enhancement (Week 2)
- Add quiz generation
- Implement spaced repetition
- Create podcast scripts
- Add multiple search sources

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| API costs exceed budget | High | Implement caching, use smaller models |
| Poor paper quality | Medium | Add quality filters, citation thresholds |
| Information overload | Medium | Adjustable pace, summary depth levels |
| Pillar context confusion | High | Strict namespace isolation, clear UI |
| Dependency on external APIs | High | Local fallbacks, multiple providers |

## Appendix

### The 5 Pillars (Detailed)

1. **Linguistic & Cognitive Foundations**
   - Morphology, Syntax, Semantics, Pragmatics
   - Psycholinguistics and language acquisition
   - Cognitive alignment with human processing
   - Formal language theory

2. **Models & Architectures**
   - Transformers and attention mechanisms
   - Long-context and multimodal models
   - Neurosymbolic AI
   - Emergent communication

3. **Data, Training & Methodologies**
   - Data curation and annotation
   - Low-resource languages
   - RLHF, DPO, instruction tuning
   - Synthetic data generation

4. **Evaluation & Interpretability**
   - Metrics and benchmarks
   - Robustness testing
   - Explainable AI techniques
   - Error analysis frameworks

5. **Ethics & Applications**
   - Bias detection and mitigation
   - Real-world deployments
   - Policy and governance
   - Cultural preservation

### Example Daily Output

```
📚 Today's Lesson - Pillar 2: Models & Architectures
Paper: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

🎯 TL;DR: 
Mamba introduces selective state space models that match Transformer quality 
while scaling linearly with sequence length, not quadratically.

💡 Key Takeaways:
1. Hardware-aware parallel algorithm in recurrent mode
2. 5x faster inference than Transformers on long sequences
3. Strong performance on language, audio, and genomics

🛠 Try This:
- Compare Mamba vs GPT on a 10K token document
- Implement the selective scan algorithm
- Test on time-series prediction

🔗 Related Papers:
- "Hungry Hungry Hippos: Towards Language Modeling with State Space Models"
- "Efficiently Modeling Long Sequences with Structured State Spaces"

📝 Quiz: [3 questions available]
🎙 Podcast: [12-minute episode ready]
```
