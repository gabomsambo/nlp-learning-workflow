# 🧠 NLP Learning Workflow

An intelligent, self-updating learning system that automatically discovers, processes, and synthesizes the latest NLP research across 5 independent learning pillars. Built with Atomic Agents v2.0 for maximum modularity and maintainability.

## 🎯 What It Does

Every day, this system:
1. **Discovers** relevant papers based on your learning goals
2. **Summarizes** complex research into structured notes
3. **Synthesizes** digestible lessons with practical takeaways
4. **Generates** quizzes for spaced repetition learning
5. **Tracks** your progress independently across 5 NLP pillars

## 🏛️ The 5 Pillars

Each pillar maintains its own queue, memory, and progress:

1. **Linguistic & Cognitive Foundations** - Core linguistics, psycholinguistics, cognitive alignment
2. **Models & Architectures** - Transformers, multimodal AI, emerging paradigms
3. **Data, Training & Methodologies** - RLHF, low-resource languages, data curation
4. **Evaluation & Interpretability** - Metrics, XAI, robustness testing
5. **Ethics & Applications** - Bias mitigation, real-world deployment, policy

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (required)
- Supabase account (free tier works)
- Qdrant instance (optional, can use local)

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

# Database (Supabase)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...

# Vector Store (Qdrant)
QDRANT_URL=http://localhost:6333  # or cloud URL
QDRANT_API_KEY=...  # if using cloud

# Optional providers
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...
```

### Database Setup

Run the SQL migrations in Supabase:
```sql
-- Run the SQL from docs/migrations/001_initial_schema.sql
```

### First Run

```bash
# Initialize pillars
python -m nlp_pillars.cli pillars init

# Run learning session for Pillar 1
python -m nlp_pillars.cli run --pillar P1 --papers 1

# Check status
python -m nlp_pillars.cli status --pillar P1

# Review quiz cards
python -m nlp_pillars.cli review --pillar P1
```

## 📖 Usage

### Daily Learning Flow

```bash
# Morning: Process new papers for chosen pillar
python -m nlp_pillars.cli run --pillar P2 --papers 2

# Review: Check what you learned
python -m nlp_pillars.cli show-lesson --pillar P2 --latest

# Quiz: Test your knowledge
python -m nlp_pillars.cli quiz --pillar P2

# Switch pillars anytime (progress is saved)
python -m nlp_pillars.cli run --pillar P4 --papers 1
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
| `run --pillar P1 --papers N` | Process N papers for pillar |
| `status --pillar P1` | Show pillar progress and queue |
| `review --pillar P1` | Review spaced repetition cards |
| `quiz --pillar P1` | Take interactive quiz |
| `pillars list` | List all pillars and goals |
| `pillars set-goal P1 "..."` | Update pillar learning goal |
| `export --pillar P1 --format md` | Export notes as markdown |

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
  P1:
    papers_per_day: 2
    summary_depth: detailed  # or: concise, comprehensive
    quiz_difficulty: medium  # easy, medium, hard
  P2:
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
📚 Today's Lesson - Pillar 2: Models & Architectures
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

2. **"Cannot connect to Supabase"**
   - Verify SUPABASE_URL includes https://
   - Check SUPABASE_KEY is the anon/public key

3. **"PDF extraction failed"**
   - Some papers have complex layouts
   - System will retry with different extractors
   - Check logs in `logs/` directory

4. **"Rate limit exceeded"**
   - ArXiv: Wait 3 seconds between requests
   - OpenAI: Reduce parallel processing

## 📈 Monitoring Progress

View your learning analytics:
```bash
# Overall statistics
python -m nlp_pillars.cli stats

# Detailed pillar report
python -m nlp_pillars.cli report --pillar P1 --format html
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
- Database by Supabase
- Vector search by Qdrant

## 📬 Support

- Issues: [GitHub Issues](https://github.com/yourusername/NLPWorkflow/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/NLPWorkflow/discussions)
- Email: your.email@example.com

---

**Ready to accelerate your NLP learning? Start with:**
```bash
python -m nlp_pillars.cli run --pillar P1 --papers 1
```
