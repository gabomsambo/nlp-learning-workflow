# NLP Learning Workflow - Development Prompts

## 🎯 Master Prompts for Cursor AI

### Initial Project Setup Prompt
```
I'm building an NLP learning workflow using Atomic Agents v2.0. The system has 5 independent learning pillars (Linguistic Foundations, Models & Architectures, Data & Training, Evaluation, Ethics & Applications). Each pillar needs isolated progress tracking, its own paper queue, and separate vector namespaces in Qdrant.

Key requirements:
1. Use Atomic Agents v2.0 patterns with typed Pydantic schemas
2. Each agent should be single-purpose and composable
3. Implement a daily pipeline: Discovery -> Search -> Ingest -> Summarize -> Synthesize -> Quiz
4. Use Supabase for structured data, Qdrant for vectors
5. Support multiple LLM providers (OpenAI, Anthropic, Groq)

Please help me implement [SPECIFIC COMPONENT] following these patterns.
```

### Agent Creation Prompt
```
Create a new Atomic Agent v2.0 for [PURPOSE]. 

Requirements:
- Input schema: [DESCRIBE INPUTS]
- Output schema: [DESCRIBE OUTPUTS]  
- Use SystemPromptGenerator with clear background, steps, and output instructions
- Include relevant context providers (RecentProgressProvider, GoalProvider)
- Follow the pattern from the existing SummarizerAgent
- Make it composable with other agents in the pipeline
- Add proper error handling and logging

The agent should be single-purpose and strongly typed.
```

### Schema Definition Prompt
```
Define Pydantic schemas for [DOMAIN] following Atomic Agents BaseIOSchema patterns.

Requirements:
- Inherit from BaseIOSchema
- Include clear Field descriptions
- Use appropriate types (str, List, Optional, etc.)
- Consider schema evolution and backwards compatibility
- Make schemas composable (output of one agent = input of next)

Example domains: PaperNote, Lesson, QuizCard, PodcastScript
```

### Tool Implementation Prompt
```
Create a tool for [SERVICE] (e.g., ArXiv, Semantic Scholar) that:

1. Implements proper rate limiting
2. Returns data in PaperRef schema format
3. Handles errors gracefully with retries
4. Supports both sync and async operations
5. Includes comprehensive logging
6. Can be easily swapped with other search tools

Follow the Atomic Agents tool patterns and make it compatible with the Discovery Agent's output schema.
```

### Database Operation Prompt
```
Implement database operations for [ENTITY] with Supabase:

Requirements:
- Maintain pillar isolation (use pillar_id)
- Implement CRUD operations
- Add proper indexes for performance
- Include transaction support where needed
- Handle conflicts (upsert by DOI/arxiv ID)
- Add migration scripts

Entities: papers, notes, lessons, quiz_cards, progress
```

### CLI Command Prompt
```
Add a new CLI command using Typer for [ACTION]:

Requirements:
- Use descriptive option names with help text
- Support both interactive and scriptable modes
- Include progress indicators for long operations
- Handle errors gracefully with helpful messages
- Follow existing CLI patterns in cli.py

Example actions: export notes, bulk import papers, reset pillar, analyze trends
```

### Testing Prompt
```
Write comprehensive tests for [COMPONENT]:

Requirements:
- Unit tests with mocked dependencies
- Use pytest fixtures for common setup
- Test both success and failure cases
- Include edge cases and boundary conditions
- Mock external API calls
- Verify schema validation
- Test async operations if applicable

Components: agents, tools, database operations, CLI commands
```

### Bug Fix Prompt
```
Debug and fix [ISSUE]:

Context: [DESCRIBE PROBLEM]
Error message: [PASTE ERROR]
Expected behavior: [WHAT SHOULD HAPPEN]

Please:
1. Identify root cause
2. Propose fix following existing patterns
3. Add tests to prevent regression
4. Update documentation if needed
```

## 🔄 Refactoring Prompts

### Performance Optimization
```
Optimize [COMPONENT] for better performance:

Current issues:
- [LIST PERFORMANCE PROBLEMS]

Requirements:
- Implement caching where appropriate
- Use batch operations for database/API calls
- Add async support if beneficial
- Maintain code readability
- Add performance logging

Measure and document the improvements.
```

### Code Quality Improvement
```
Refactor [MODULE] to improve code quality:

Goals:
- Reduce complexity (target McCabe complexity < 10)
- Improve type hints coverage to 100%
- Extract common patterns into utilities
- Add comprehensive docstrings
- Follow PEP 8 and project conventions
- Increase test coverage to >80%
```

## 📚 Context Snippets

### When working on Agents
```
Remember: Each agent in this system uses Atomic Agents v2.0 patterns:
- AtomicAgent[InputSchema, OutputSchema] with generic types
- AgentConfig with instructor client
- SystemPromptGenerator for prompts
- Context providers for dynamic information
- Structured outputs via Pydantic schemas
```

### When working on Database
```
Database context:
- Using Supabase (PostgreSQL)
- Each pillar has isolated data (pillar_id foreign key)
- Papers identified by DOI or arXiv ID (deduplication)
- JSONB columns for flexible array storage
- Spaced repetition fields in quiz_cards table
```

### When working on Search Tools
```
Search tool requirements:
- Respect rate limits (ArXiv: 1/3s, Semantic Scholar: 100/5min)
- Return PaperRef schema for compatibility
- Implement exponential backoff for retries
- Cache results to minimize API calls
- Support query refinement based on results
```

## 🎨 UI/Visualization Prompts

### Progress Dashboard
```
Create a simple text-based dashboard showing:
- Papers processed per pillar
- Current learning streaks
- Quiz performance metrics
- Queue lengths
- Recent lessons summary

Use rich/textual for terminal UI or generate static HTML.
```

### Knowledge Graph
```
Generate a knowledge graph visualization showing:
- Paper relationships (citations)
- Concept connections across pillars
- Learning path through topics
- Temporal progression of research

Output as interactive HTML using D3.js or PyVis.
```

## 🚀 Advanced Feature Prompts

### Multi-User Support
```
Extend the system for multiple users:
- Add user authentication
- Isolate data per user
- Share papers but not progress
- Add collaborative features (shared notes)
- Implement role-based access
```

### Custom Pillar Creation
```
Allow users to define custom pillars beyond the default 5:
- Dynamic pillar configuration
- Custom goal and search parameters
- Pillar templates for common domains
- Import/export pillar definitions
```

### Advanced Analytics
```
Implement analytics to track:
- Learning velocity trends
- Knowledge retention rates
- Topic difficulty analysis
- Optimal learning times
- Cross-pillar correlation insights
```

## 💡 Quick Fixes

### "Make it work first" prompt
```
I need a quick working implementation of [FEATURE]. Don't worry about perfection, just make it functional so I can test the concept. We can refactor later. Use simple solutions and existing libraries where possible.
```

### "Fix this error" prompt
```
I'm getting this error: [PASTE ERROR]
Code context: [PASTE RELEVANT CODE]
What I tried: [LIST ATTEMPTS]

Give me the minimal fix to make this work, then explain why it happened.
```

### "Simplify this code" prompt
```
This code works but is too complex: [PASTE CODE]

Simplify it while maintaining functionality. Use Python idioms, remove unnecessary abstractions, and make it more readable.
```

## 🏗️ Architecture Decision Prompts

### Technology Choice
```
I need to choose between [OPTION A] and [OPTION B] for [PURPOSE].

Context:
- Current stack: [LIST TECHNOLOGIES]
- Requirements: [LIST REQUIREMENTS]
- Constraints: [LIST CONSTRAINTS]

Analyze trade-offs and recommend the best option with justification.
```

### Design Pattern Selection
```
I need to implement [FEATURE]. Should I use:
- Pattern A: [DESCRIBE]
- Pattern B: [DESCRIBE]

Consider maintainability, testability, and alignment with Atomic Agents patterns.
```

## 📋 Checklist Prompts

### Pre-Commit Checklist
```
Review my changes and ensure:
- [ ] All tests pass
- [ ] Type hints are complete
- [ ] Docstrings are updated
- [ ] No hardcoded values
- [ ] Error handling is comprehensive
- [ ] Logging is appropriate
- [ ] Code follows project style
- [ ] Documentation is updated
```

### Feature Complete Checklist
```
Verify that [FEATURE] implementation includes:
- [ ] Core functionality
- [ ] Error handling
- [ ] Tests (unit + integration)
- [ ] Documentation
- [ ] CLI command
- [ ] Configuration options
- [ ] Performance considerations
- [ ] Security review
```

## 🎯 Specific Component Prompts

### Discovery Agent
```
The Discovery Agent needs to generate intelligent search queries based on:
- Pillar learning goals
- Recent papers to avoid duplication
- Current research trends
- User-specified priorities

Output 3-5 diverse queries that will find relevant, recent papers.
```

### Summarizer Agent
```
The Summarizer Agent must extract structured information:
- Problem: What specific problem does this paper address?
- Method: How do they solve it? (architecture, algorithm, approach)
- Findings: What are the key results? (metrics, improvements)
- Limitations: What doesn't it do? What are the weaknesses?
- Future Work: What questions remain open?
- Key Terms: Technical vocabulary with brief definitions
```

### Quiz Generator
```
Generate quiz questions that:
- Test different cognitive levels (recall, understanding, application)
- Include varied formats (multiple choice, short answer, true/false)
- Have clear, unambiguous answers
- Are appropriately difficult for the content
- Support spaced repetition scheduling
```

### Podcast Script Generator
```
Create a podcast dialogue between two hosts discussing the paper:
- Host 1: CS perspective (technical implementation)
- Host 2: Linguistics perspective (theoretical implications)
- Natural, conversational tone
- Explain complex concepts simply
- Include specific examples
- 10-15 minute episode length
```

## 🔍 Debugging Prompts

### Trace Execution
```
Add detailed logging to trace execution flow through [COMPONENT]:
- Entry/exit points
- Parameter values
- Intermediate results
- Time taken for each step
- Memory usage if relevant
```

### Memory Leak Investigation
```
Investigate potential memory leak in [COMPONENT]:
- Profile memory usage
- Identify objects not being garbage collected
- Check for circular references
- Verify context managers are used correctly
- Add memory monitoring
```

### Performance Profiling
```
Profile [COMPONENT] to identify bottlenecks:
- Use cProfile/line_profiler
- Measure database query times
- Check API call frequency
- Identify N+1 queries
- Find unnecessary computations
```

## 🌟 Best Practices Reminders

### When stuck, remember:
1. Start simple, iterate towards complexity
2. Write tests first for unclear requirements
3. Use type hints to catch errors early
4. Small, focused commits with clear messages
5. Document "why" not just "what"
6. Profile before optimizing
7. Handle errors at appropriate levels
8. Log liberally during development, refine for production

### Code Review Self-Prompt
Before requesting review, ask yourself:
- Would I understand this code in 6 months?
- Are edge cases handled?
- Is it testable in isolation?
- Does it follow project patterns?
- Are there any security concerns?
- Is the complexity justified?

## 🚦 Getting Started Prompts

### Day 1: "Help me set up the basic project structure with all necessary files and folders"

### Day 2: "Let's implement the core schemas and basic Discovery Agent"

### Day 3: "Now add the Search tools and Ingest Agent with PDF processing"

### Day 4: "Implement the Summarizer and Synthesis Agents with proper context"

### Day 5: "Add the Quiz Agent and wire everything together in the Orchestrator"

### Day 6: "Set up Supabase integration and implement persistence"

### Day 7: "Create the CLI interface and add automated scheduling"

## 📝 Notes Section
Use this section to keep track of:
- Decisions made and why
- Known issues to address
- Future improvements
- Performance benchmarks
- Useful code snippets
- External resources

---

Remember: The goal is to build a working system iteratively. Start with the happy path, then add error handling, then optimize. Each component should be testable and replaceable.
