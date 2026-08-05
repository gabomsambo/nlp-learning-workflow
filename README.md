# 🧠 NLP Learning Workflow

An intelligent, self-updating learning system that automatically discovers, processes, and synthesizes the latest NLP research across 8 independent learning pillars. Built with Atomic Agents v2.0 for maximum modularity and maintainability.

## 🎯 What It Does

Every day, this system:
1. **Discovers** relevant papers based on your learning goals
2. **Summarizes** complex research into structured notes
3. **Synthesizes** digestible lessons with practical takeaways
4. **Generates** quizzes for spaced repetition learning
5. **Tracks** your progress independently across 8 NLP pillars

## 🏛️ The 8 Pillars

Each pillar maintains its own queue, memory, and progress. These are the pillars
`create_pillars.py` seeds, and they are the authoritative set — the slugs below are
the `pillar_id` values used throughout the database and API:

1. **Formal Linguistics for NLP** (`formal-linguistics-nlp`) - Syntax, formal semantics, pragmatics, morphology
2. **Neural Architectures for Language** (`neural-architectures-language`) - Transformers, attention, state space models
3. **LLM Theory & Practice** (`llm-theory-practice`) - Scaling laws, in-context learning, RLHF, alignment
4. **Computational Semantics & Meaning** (`computational-semantics`) - Embeddings, knowledge graphs, entailment, grounding
5. **Model Interpretability & Probing** (`model-interpretability`) - Probing classifiers, mechanistic interpretability, bias measurement
6. **AI Agents & Autonomous Systems** (`ai-agents-tool-use`) - ReAct, tool use, multi-agent systems, RAG, planning
7. **ML Systems & Production AI** (`ml-systems-production`) - Distributed training, serving, quantization, MLOps
8. **AI Safety & Responsible AI** (`ai-safety-alignment`) - Constitutional AI, red teaming, value alignment, governance

The static fallback in `nlp_pillars/config.py` (`PILLAR_CONFIGS`, used only when the
database lookup fails) mirrors this list. Change both together.

## 🚀 Quick Start (Docker — recommended)

`docker compose up -d --build` is the supported way to run this project. It brings up
the whole stack, with the database schema applied automatically:

| Service | Container | Host port | What it is |
|---|---|---|---|
| `db` | `nlp_postgres` | 127.0.0.1:5432 | Postgres — the application database |
| `postgrest` | `nlp_postgrest` | 127.0.0.1:3000 | PostgREST, the REST layer the app talks to |
| `searxng` | `nlp_searxng` | 8080 | Metasearch, used for paper discovery |
| `qdrant` | `nlp_qdrant` | 6333/6334 | Local vector store — **offline option only**, see below |
| `webui` | `nlp_webui` | 8000 | The FastAPI web UI |

```bash
cp .env.example .env      # then edit — OPENAI_API_KEY and the Qdrant Cloud settings
docker compose up -d --build

# Seed the 8 pillars (the script is not in the image; copy it in)
docker compose cp create_pillars.py webui:/tmp/create_pillars.py
docker compose exec -e PYTHONPATH=/app webui python /tmp/create_pillars.py

open http://localhost:8000
```

Notes that save time:

- **The database is self-hosted here, not Supabase.** The hosted Supabase project is gone.
  `docker-compose.yml` overrides `POSTGREST_URL`, `SUPABASE_URL` and `SUPABASE_KEY` on the
  `webui` service to point at the local PostgREST; the comments there explain why all three
  are needed. `SUPABASE_URL`/`SUPABASE_KEY` in `.env` are ignored for the containerised app.
- **The schema is applied on first start only**, from `docker-entrypoint-initdb.d`
  (`db/init/01-roles.sh`, `schema.sql`, `db/init/03-grants.sql`). Editing `schema.sql` does
  nothing to an existing `nlp_pg_data` volume — migrate by hand or recreate the volume.
- **Never `docker compose down -v` casually.** The `nlp_pg_data` and `nlp_uploads` volumes
  hold real data; for a file-uploaded paper, `nlp_uploads` holds the only copy of the PDF.
- **Vector search uses Qdrant Cloud**, read from `QDRANT_URL` in `.env`. The local `qdrant`
  container is deliberately *not* wired to the app and is kept only as an offline option.

## 🛠️ Manual Setup (without Docker)

### Prerequisites
- Python 3.12+ (atomic-agents requires >= 3.12)
- OpenAI API key (required)
- A Postgres database with PostgREST in front of it (see the `db` / `postgrest` services in
  `docker-compose.yml` for a working reference configuration)
- Qdrant instance (Qdrant Cloud, or a local one)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/NLPWorkflow.git
cd NLPWorkflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Environment Setup

Create a `.env` file with:
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Database — PostgREST in front of Postgres.
# Under Docker these are overridden in docker-compose.yml and can be left alone.
POSTGREST_URL=http://localhost:3000
SUPABASE_URL=http://localhost:3000   # nlp_pillars/db.py reads this one, not POSTGREST_URL
SUPABASE_KEY=eyJ...                  # sent as the PostgREST Bearer token; must be non-empty

# Vector Store (Qdrant Cloud is what the stack actually uses)
QDRANT_URL=https://xxx.cloud.qdrant.io:6333   # http://localhost:6333 for the local container
QDRANT_API_KEY=...  # required for cloud

# Metasearch (Docker sets this to http://searxng:8080)
SEARXNG_URL=http://localhost:8080

# Optional providers
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...
```

### Database Setup

Apply `schema.sql` to your Postgres database, then the role/grant scripts in `db/init/`
(`01-roles.sh`, `03-grants.sql`). Under Docker all three run automatically on first start.

### First Run

```bash
# Initialize pillars
python -m nlp_pillars.cli pillars init

# Run learning session for one pillar (use the slug, not a P1-P5 legacy ID)
python -m nlp_pillars.cli run --pillar llm-theory-practice --papers 1

# Check status
python -m nlp_pillars.cli status --pillar llm-theory-practice

# Review quiz cards
python -m nlp_pillars.cli review --pillar llm-theory-practice
```

## 📖 Usage

### Daily Learning Flow

```bash
# Morning: Process new papers for chosen pillar
python -m nlp_pillars.cli run --pillar neural-architectures-language --papers 2

# Review: Check what you learned
python -m nlp_pillars.cli show-lesson --pillar neural-architectures-language --latest

# Quiz: Test your knowledge
python -m nlp_pillars.cli quiz --pillar neural-architectures-language

# Switch pillars anytime (progress is saved)
python -m nlp_pillars.cli run --pillar model-interpretability --papers 1
```

### Automated Daily Runs

Set up a cron job or GitHub Action:
```bash
# crontab -e
0 8 * * * cd /path/to/NLPWorkflow && python -m nlp_pillars.cli run --pillar auto --papers 1
```

### CLI Commands

| Command | Description |
|---------|------------|
| `run --pillar llm-theory-practice --papers N` | Process N papers for pillar |
| `status --pillar llm-theory-practice` | Show pillar progress and queue |
| `review --pillar llm-theory-practice` | Review spaced repetition cards |
| `quiz --pillar llm-theory-practice` | Take interactive quiz |
| `pillars list` | List all pillars and goals |
| `pillars set-goal llm-theory-practice "..."` | Update pillar learning goal |
| `export --pillar llm-theory-practice --format md` | Export notes as markdown |

## 🏗️ Project Structure

```
NLPWorkflow/
├── nlp_pillars/
│   ├── agents/          # Atomic Agents (discovery, summarizer, etc.)
│   ├── tools/           # Search tools (arxiv, semantic scholar)
│   ├── context/         # Context providers for agents
│   ├── schemas.py       # Pydantic data models
│   ├── db.py           # Database operations
│   ├── vectors.py      # Qdrant vector operations
│   ├── orchestrator.py # Main pipeline coordinator
│   └── cli.py          # CLI interface
├── tests/              # Unit and integration tests
├── docs/               # Documentation
├── .env.example        # Environment variables template
└── requirements.txt    # Python dependencies
```

## 🔧 Configuration

### Adjusting Learning Pace

Edit `config.yaml`:
```yaml
pillars:
  llm-theory-practice:
    papers_per_day: 2
    summary_depth: detailed  # or: concise, comprehensive
    quiz_difficulty: medium  # easy, medium, hard
  neural-architectures-language:
    papers_per_day: 1
    focus: "transformer alternatives"
```

### Adding Custom Search Sources

Create a new tool in `nlp_pillars/tools/`:
```python
from atomic_agents.tools import BaseTool

class CustomSearchTool(BaseTool):
    def search(self, query: str) -> List[PaperRef]:
        # Your implementation
        pass
```

## 📊 Example Output

```
📚 Today's Lesson - Neural Architectures for Language
═══════════════════════════════════════════════════

Paper: "Mamba: Linear-Time Sequence Modeling"

🎯 TL;DR: 
Mamba introduces selective state space models that match
Transformer quality while scaling linearly with sequence length.

💡 Key Takeaways:
• Hardware-aware parallel algorithm in recurrent mode
• 5x faster inference than Transformers on long sequences
• Strong performance on language, audio, and genomics

🛠️ Practice Ideas:
• Compare Mamba vs GPT on 10K token documents
• Implement the selective scan algorithm
• Test on time-series prediction tasks

📝 Quiz Available: 5 questions ready for review
🎙️ Podcast Script: 12-minute episode generated
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_agents.py

# Run with coverage
pytest --cov=nlp_pillars tests/
```

## 🐛 Troubleshooting

### Common Issues

1. **"API key not found"**
   - Ensure `.env` file exists and contains valid keys
   - Check environment variable names match exactly

2. **"Cannot connect to the database" / PostgREST errors**
   - Under Docker: `docker compose ps` — check `db` is `healthy` and `postgrest` is up
   - `PGRST300 "Server lacks JWT secret"` means `SUPABASE_KEY` and PostgREST's
     `PGRST_JWT_SECRET` are mismatched; they are a matched pair in `docker-compose.yml`
   - Outside Docker: `SUPABASE_URL` must be set as well as `POSTGREST_URL` —
     `nlp_pillars/db.py` reads only the former

3. **Paper discovery returns nothing from SearXNG**
   - SearXNG serves JSON only when `formats: [html, json]` is set in
     `searxng_config/settings.yml`; without it the API answers 403
   - Check with `curl -s -o /dev/null -w '%{http_code}' "http://localhost:8080/search?q=test&format=json"`

4. **"PDF extraction failed"**
   - Some papers have complex layouts
   - System will retry with different extractors
   - Check logs in `logs/` directory

5. **"Rate limit exceeded"**
   - ArXiv: Wait 3 seconds between requests
   - OpenAI: Reduce parallel processing

## 📈 Monitoring Progress

View your learning analytics:
```bash
# Overall statistics
python -m nlp_pillars.cli stats

# Detailed pillar report
python -m nlp_pillars.cli report --pillar llm-theory-practice --format html
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- Built with [Atomic Agents](https://github.com/BrainBlend-AI/atomic-agents)
- Powered by OpenAI, Anthropic, and Groq APIs
- Database: Postgres + PostgREST (self-hosted, see docker-compose.yml)
- Vector search by Qdrant

## 📬 Support

- Issues: [GitHub Issues](https://github.com/yourusername/NLPWorkflow/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/NLPWorkflow/discussions)
- Email: your.email@example.com

---

**Ready to accelerate your NLP learning? Start with:**
```bash
python -m nlp_pillars.cli run --pillar llm-theory-practice --papers 1
```

## 🌐 Web UI (FastAPI)

Under Docker the web UI is already running — `docker compose up -d` serves it on
`http://localhost:8000`, and nothing below is needed.

To run it directly against an existing PostgREST instead, point `POSTGREST_URL`,
`SUPABASE_URL` and `SUPABASE_KEY` at it (see Environment Setup above) and then:
```bash
pip install -r requirements.txt -r requirements.lock.txt
uvicorn webui.app:app --reload
```

Open `http://localhost:8000`:
- Dashboard: counts and recent activity
- Papers: browse and filter by pillar
- Lessons: view generated lessons
- Pipeline: trigger runs and view results inline
- Analytics: placeholder (to be expanded)
