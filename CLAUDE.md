# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This is an autonomous multi-agent system built with LangGraph that can intelligently route tasks based on complexity and autonomously call tools to complete user queries. The system uses local Ollama models (qwen2.5:latest for main agent, gemma3:1b for supporting tasks) and integrates with external APIs.

## Core Architecture

### Agent Flow Structure
The system implements a dynamic routing architecture:

```
User Query → Task Classifier → [Simple: Direct Tool Agent | Complex: Planning → Tool Agent]
                                      ↓
Tool Agent ⟷ ToolNode (web_search, http_request, execute_code, get_time_info)
     ↓
Continue Processing → Progress Check → [More Tasks | Summarizer] → End
```

### Key Components

**agents.py** - Single file containing the complete system:
- **AgentState**: TypedDict defining shared state across all agents including messages, todo_list, tool_results, complexity_level
- **Task Classifier**: Analyzes query complexity (simple/moderate/complex) and determines routing
- **Planning Agent**: Breaks complex queries into structured todo lists with tool requirements
- **Tool-Calling Agent**: Autonomously decides which tools to call and when, maintains conversation context
- **ToolNode**: Executes the 4 approved tools (web search via Tavily, HTTP requests, code execution, time/date)
- **Routing Functions**: `complexity_router()`, `should_continue()`, `progress_checker()` manage flow control

### Tool Integration
- **Tavily Search**: Real web search with advanced depth, requires TAVILY_API_KEY in .env
- **HTTP Request**: GET/POST requests to external APIs with timeout handling
- **Code Execution**: Safe Python code execution with restricted built-ins namespace
- **Time Info**: UTC time and timezone information

## Development Commands

### Running the System
```bash
python agents.py
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt  # or use uv for faster installs

# Required environment variables in .env:
TAVILY_API_KEY=your_tavily_api_key_here
```

### Code Quality
```bash
ruff check .    # Linting
ruff format .   # Formatting
```

## Configuration Notes

### Model Configuration
- Main agent uses `qwen2.5:latest` with 10k context (requires sufficient GPU memory)
- Todo agent uses `gemma3:1b` for efficiency
- Models are bound to tools via `main_llm.bind_tools(tools)` for autonomous tool selection


### State Management
- **Asynchronous throughout**: All agents and tools use async/await patterns
- **Message-based context**: Tool results integrate into conversation history via state["messages"]
- **Progress tracking**: Todo lists track task completion with status updates

## Extending the System

### Adding New Tools
1. Define tool with `@tool` decorator
2. Add to `tools` list
3. ToolNode automatically integrates new tools
4. Update tool descriptions in agent prompts

### Modifying Agent Behavior
- **Task complexity logic**: Modify `complexity_router()` routing conditions
- **Tool calling prompts**: Update system prompts in `tool_calling_agent()`
- **Planning behavior**: Adjust `planning_agent()` task breakdown logic

### Architecture Patterns
- All routing uses conditional edges with explicit return type hints
- Tools maintain consistent error handling with try/catch blocks
- Rich console provides colored output with panels and progress indicators
- State mutations happen in-place and return modified state objects