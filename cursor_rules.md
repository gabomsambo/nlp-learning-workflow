# NLP Learning Workflow - Cursor AI Rules

## Project Context
This is a modular, self-updating NLP learning system using Atomic Agents v2.0. The system manages 5 independent learning pillars, each with its own memory, progress tracking, and content pipeline.

## Architecture Overview
- Framework: Atomic Agents v2.0 (modular, typed, composable agents)
- Database: Supabase (PostgreSQL)
- Vector Store: Qdrant (namespaced by pillar)
- Models: Multi-provider support (OpenAI, Anthropic, Groq, Gemini)
- Scheduling: APScheduler / GitHub Actions

## Core Design Principles
1. **Pillar Independence**: Each of the 5 pillars operates in isolation with separate:
   - Database tables (prefixed/namespaced)
   - Vector store namespaces
   - Progress tracking
   - Memory/context
   
2. **Atomic Modularity**: Every component is:
   - Single-purpose (one clear responsibility)
   - Strongly typed (Pydantic schemas)
   - Composable (input/output alignment)
   - Testable in isolation

3. **Type Safety First**: 
   - Always use Pydantic BaseModel for schemas
   - Define clear input/output types for every agent
   - Use TypedDict for complex nested structures
   - Leverage instructor for structured LLM outputs

## The 5 Pillars
1. **Linguistic & Cognitive Foundations**: Core linguistics, psycholinguistics, cognitive alignment
2. **Models & Architectures**: Transformers, emerging paradigms, multimodal systems
3. **Data, Training & Methodologies**: Data curation, RLHF, low-resource languages
4. **Evaluation & Interpretability**: Metrics, benchmarks, XAI, robustness testing
5. **Ethics & Applications**: Fairness, real-world deployment, policy, cultural preservation

## Code Style Guidelines
- Use descriptive variable names for business logic
- Use concise names (i, j, el) for loops and DOM manipulation
- Prefer composition over inheritance
- Keep functions small and focused (<50 lines)
- Document complex logic with inline comments

## Atomic Agents Patterns
```python
# Standard agent structure
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from pydantic import Field

class InputSchema(BaseIOSchema):
    field: str = Field(..., description="Clear description")

class OutputSchema(BaseIOSchema):
    result: str = Field(..., description="Clear description")

agent = AtomicAgent[InputSchema, OutputSchema](
    config=AgentConfig(
        client=instructor.from_openai(client),
        model="gpt-4o-mini",
        system_prompt_generator=system_prompt,
        history=ChatHistory(),
    )
)
```

## Database Schema Pattern
Tables follow this naming: `{pillar_id}_{entity}` or include `pillar_id` column
- pillars(id, name, goal, created_at)
- papers(pillar_id, paper_id, title, authors, venue, year, url_pdf)
- notes(pillar_id, paper_id, problem, method, findings, limitations)
- lessons(pillar_id, paper_id, tl_dr, takeaways, practice_ideas)
- quiz_cards(pillar_id, card_id, question, answer, difficulty, due_at)

## Testing Strategy
- Unit tests for each agent (mock LLM responses)
- Integration tests for tool chains
- Schema validation tests
- Database migration tests

## Error Handling
- Use structured exceptions
- Log errors with context
- Implement retry logic for API calls
- Graceful degradation for missing tools

## Common Commands
```bash
# Install dependencies
uv venv && source .venv/bin/activate
pip install -r requirements.txt

# Run daily learning session
python -m nlp_pillars.cli run --pillar P1 --papers 2

# Check pillar status
python -m nlp_pillars.cli status --pillar P1

# Review quiz cards
python -m nlp_pillars.cli review --pillar P1

# Run tests
pytest tests/ -v

# Format code
black nlp_pillars/
ruff check nlp_pillars/
```

## Environment Variables
Required:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY (optional)
- GROQ_API_KEY (optional)
- SUPABASE_URL
- SUPABASE_KEY
- QDRANT_URL
- QDRANT_API_KEY

## File Naming Conventions
- Agents: `{purpose}_agent.py` (e.g., discovery_agent.py)
- Tools: `{service}_tool.py` (e.g., arxiv_tool.py)
- Schemas: `{domain}_schemas.py` (e.g., paper_schemas.py)
- Tests: `test_{module}.py`

## When Implementing New Features
1. Define Pydantic schemas first
2. Create single-purpose agent
3. Write tests
4. Integrate with orchestrator
5. Update CLI commands
6. Document in README

## Performance Considerations
- Batch database operations
- Use async where possible
- Cache frequently accessed data
- Implement pagination for large datasets
- Use streaming for long responses

## Security Notes
- Never commit .env files
- Use environment variables for secrets
- Validate all user inputs
- Sanitize database queries
- Implement rate limiting for API calls
