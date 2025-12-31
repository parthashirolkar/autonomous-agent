# Development Guidelines for Autonomous Agents

## Build/Lint/Test Commands

### Code Quality
```bash
ruff check .          # Lint all Python files
ruff format .         # Format all Python files
```

### Running Tests
```bash
# No test framework currently configured
# Tests should be added using pytest: uv run pytest tests/
```

### Running the Application
```bash
# Main agent system
uv run python agents.py

# Setup database from CSV files
uv run python setup_database.py

# Generate agent graph visualization
uv run python generate_graph.py
```

### Dependency Management
```bash
# Install dependencies (preferred method)
uv pip install -r requirements.txt

# Alternative: pip install
pip install -r requirements.txt
```

## Code Style Guidelines

### Imports
- Group imports: standard library → third-party → local
- Use `from typing_extensions` for TypedDict, Annotated, List, Literal, Dict, Any
- LangChain imports: `from langchain_ollama`, `from langgraph.graph`, etc.
- Example order:
  ```python
  import asyncio
  from dotenv import load_dotenv
  from langchain_ollama import ChatOllama
  from langgraph.graph import START, StateGraph, END
  from tools import web_search, execute_code
  ```

### Formatting
- Use **ruff** for consistent formatting (no configuration file needed)
- Line length: default ruff settings
- Use f-strings for string formatting
- Triple quotes for module and function docstrings

### Type Hints
- All functions should use type hints
- Use `TypedDict` from `typing_extensions` for complex state objects
- Use `Annotated` for fields with special annotations (e.g., `add_messages`)
- Use `Literal` for string union types (e.g., `"main_data_agent" | "sql_expert_agent"`)
- Example:
  ```python
  from typing_extensions import TypedDict, Annotated, List, Literal, Dict, Any

  class AgentState(TypedDict):
      messages: Annotated[List, add_messages]
      original_query: str
      complexity_level: Literal["simple", "moderate", "complex"]
  ```

### Naming Conventions
- **Functions/variables**: `snake_case` (e.g., `task_classifier`, `get_database_schema`)
- **Classes**: `PascalCase` (e.g., `AgentState`, `StreamingCallback`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DB_PATH`, `main_tools`)
- **Agent nodes**: `lowercase_with_underscores` (e.g., `main_data_agent`, `sql_expert_agent`)
- **Private functions**: `snake_case` (Python convention, no prefix needed)

### Error Handling
- Always wrap database operations and tool calls in try/except
- Return descriptive error messages from async tools
- Use Rich console for error formatting: `console.print(f"[red]❌ Error: {msg}[/red]")`
- Log errors appropriately with logging module
- Example:
  ```python
  try:
      result = await operation()
      return result
  except Exception as e:
      error_msg = f"Operation failed: {str(e)}"
      console.print(f"[red]❌ {error_msg}[/red]")
      return error_msg
  ```

### Async/Await
- Use `async def` for all LangGraph agent functions
- Use `await` for LLM invocations and async tool calls
- Use `asyncio.run()` in `__main__` blocks
- Main entry point pattern:
  ```python
  async def main():
      # async code here
      pass

  if __name__ == "__main__":
      asyncio.run(main())
  ```

### Tool Implementation
- Use `@tool` decorator from `langchain_core.tools`
- Tools must be async functions returning `str`
- Include comprehensive docstrings for AI understanding
- Display tool execution with Rich panels
- Return formatted results, not print statements
- Example:
  ```python
  @tool
  async def my_tool(param: str) -> str:
      """Tool description for the AI agent."""
      try:
          console.print(f"[yellow]Executing: {param}[/yellow]")
          result = perform_operation(param)
          console.print("[green]✅ Success[/green]")
          return f"Result: {result}"
      except Exception as e:
          return f"Error: {str(e)}"
  ```

### LangGraph Patterns
- Use `StateGraph` with custom `TypedDict` for state management
- Define nodes as async functions that accept and return state
- Use conditional edges for routing: `add_conditional_edges()`
- Router functions should return string literals for node names
- Pattern:
  ```python
  agent = StateGraph(AgentState)
  agent.add_node("node_name", node_function)
  agent.add_conditional_edges("source_node", router_func, {"route": "target_node"})
  agent.add_edge(START, "first_node")
  agent.add_edge("last_node", END)
  compiled_agent = agent.compile()
  ```

### Database Operations
- Use sqlite3 connections with context managers when possible
- Always use parameterized queries to prevent SQL injection
- Validate SQL queries before execution (block DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, TRUNCATE)
- Use LIMIT clauses for large tables
- Close connections explicitly: `conn.close()`

### Console Output with Rich
- Use `console.print()` with rich markup: `[color]text[/color]`
- Use `Panel.fit()` for important messages
- Use `Tree()` for hierarchical data display
- Use `Table()` for tabular data
- Icons: 🎯 🧠 🗃️ 🐍 📊 📝 ✅ ❌

### File Organization
- **agents.py**: Main agent system, state definitions, graph construction
- **tools.py**: Tool implementations with @tool decorators
- **database.py**: Database utilities and schema extraction
- **setup_database.py**: Database initialization scripts
- All Python files use shebang: `#!/usr/bin/env python3`

### Security Best Practices
- Block dangerous SQL keywords in query tools
- Use controlled builtins in code execution (no file operations)
- Never commit .env files or API keys
- Database is read-only for agent operations
- Validate all external inputs

### Environment Setup
- Required env var: `TAVILY_API_KEY`
- Load with: `load_dotenv()` at module level
- Use `.gitignore` to exclude `.env`, `*.db`, `csv-data/`, `.venv`
