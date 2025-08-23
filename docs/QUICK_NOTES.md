Atomic Agents v2.0 - Quick Notes for Developers
This document is a lightweight "cheat sheet" summarizing the essential developer patterns for implementing agents with the Atomic Agents v2.0 framework. It is intended for quick reference during development.

1. Core Concepts
The framework is built around a few key components that work together.

AtomicAgent: The core class for creating an agent. It orchestrates the interaction between the user input, system prompt, chat history, and the LLM. In v2.0, it uses generics for type safety: AtomicAgent[InputSchema, OutputSchema].

AgentConfig: A Pydantic model used to configure an AtomicAgent. It holds the instructor client, model name, ChatHistory, and SystemPromptGenerator.

BaseIOSchema: The base class for all input and output schemas. It inherits from Pydantic's BaseModel and enforces that a docstring is present to describe the schema's purpose to the LLM.

SystemPromptGenerator: Constructs the system prompt from static parts (background, steps, output instructions) and dynamic parts (context_providers).

ChatHistory: Manages the conversation history. It stores messages with roles (user, assistant) and content, handling serialization and turn management automatically.

Basic Agent Initialization
import instructor
import openai
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory

# 1. Configure the client
client = instructor.from_openai(openai.OpenAI())

# 2. Set up the agent configuration
config = AgentConfig(
    client=client,
    model="gpt-4o-mini",
    history=ChatHistory()
)

# 3. Instantiate the agent with I/O schemas
agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

# 4. Run the agent
response = agent.run(BasicChatInputSchema(chat_message="Hello, world!"))
print(response.chat_message)

2. I/O Schemas: Pydantic + Instructor
Structured input and output are central to Atomic Agents. You define the structure using Pydantic models that inherit from BaseIOSchema. The instructor library handles the magic of getting the LLM to return valid JSON matching your output schema.

Docstrings are critical: The docstring of your schema is used as its description in the LLM prompt.

Field descriptions: Use pydantic.Field to provide descriptions for each attribute, guiding the LLM on what to generate.

Custom Schema Example
from pydantic import Field
from typing import List
from atomic_agents import BaseIOSchema

class ResearchQueryInput(BaseIOSchema):
    """Input schema for generating research queries."""
    topic: str = Field(..., description="The central topic to research.")

class ResearchQueryOutput(BaseIOSchema):
    """Output schema with a list of generated search queries."""
    queries: List[str] = Field(..., description="A list of 3-5 optimized search engine queries.")

3. Context Providers
Context providers dynamically inject runtime information into the system prompt. This is useful for providing data like search results, the current date, or the state of another component.

Create a class that inherits from BaseDynamicContextProvider.

Implement the get_info() method to return a formatted string.

Register an instance of it with your agent.

Context Provider Example
from atomic_agents.context import BaseDynamicContextProvider
from datetime import datetime

class CurrentDateProvider(BaseDynamicContextProvider):
    def get_info(self) -> str:
        return f"The current date is {datetime.now().strftime('%Y-%m-%d')}."

# Register with an agent instance
date_provider = CurrentDateProvider(title="Current Date")
agent.register_context_provider("current_date", date_provider)

4. Chaining Agents
To chain agents, align the output_schema of the first agent with the input_schema of the second agent. This creates a predictable, type-safe pipeline.

Chaining Pattern
# Agent 1 generates search queries
query_agent = AtomicAgent[UserInputSchema, SearchQueryOutputSchema](config1)

# Agent 2 takes search queries and returns a summary
summary_agent = AtomicAgent[SearchQueryOutputSchema, SummaryOutputSchema](config2)

# Chain the execution
user_input = UserInputSchema(topic="Quantum Computing")
search_queries = query_agent.run(user_input)
summary = summary_agent.run(search_queries) # Output of agent 1 is input for agent 2

5. Orchestrator Pattern
An orchestrator is a primary agent that decides which tool or subsequent agent to call next. This is achieved by using a Union in its output schema to define the possible choices.

Orchestrator Output Schema Example
from typing import Union
from orchestration_agent.tools.calculator import CalculatorToolInputSchema
from orchestration_agent.tools.searxng_search import SearXNGSearchToolInputSchema

class OrchestratorOutputSchema(BaseIOSchema):
    """Combined output schema for the Orchestrator Agent. Contains the tool parameters."""
    tool_parameters: Union[SearXNGSearchToolInputSchema, CalculatorToolInputSchema] = Field(
        ..., description="The parameters for the selected tool (either search or calculator)."
    )

The orchestrator's logic then inspects the type of tool_parameters to determine which tool to execute.

6. Atomic Forge Tools
The framework includes a CLI, atomic, to download pre-built, standalone tools from the Atomic Forge. These tools are added directly to your project for full control and transparency.

Using a Forge Tool (e.g., SearXNG Search)
Download the tool:

atomic # Follow the interactive prompts to download searxng_search

Use it in your code:

from deep_research.tools.searxng_search import SearXNGSearchTool, SearXNGSearchToolConfig, SearXNGSearchToolInputSchema

# Configure and instantiate the tool
search_config = SearXNGSearchToolConfig(base_url="http://localhost:8080")
search_tool = SearXNGSearchTool(search_config)

# Run the tool with its specific input schema
search_input = SearXNGSearchToolInputSchema(queries=["what is atomic agents v2.0"])
results = search_tool.run(search_input)

print(results)

7. What's New in v2.0
Cleaner Imports: Core components are now available at the top level.

# v2.0 import style
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema

Generics for Type Safety: AtomicAgent is now a generic class, improving type hinting and reducing errors.

# Define agent with specific input/output types
agent = AtomicAgent[MyInputSchema, MyOutputSchema](config)

Streaming Support: For interactive, real-time responses, use the run_stream (sync) or run_async_stream (async) methods. They yield partial responses as they are generated by the LLM.

# Synchronous streaming
for partial_response in agent.run_stream(user_input):
    print(partial_response.chat_message, end="", flush=True)

# Asynchronous streaming
async for partial_response in agent.run_async_stream(user_input):
    print(partial_response.chat_message, end="", flush=True)
