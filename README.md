# E-Commerce Data Analysis Agent

A specialized data analysis helper agent system for e-commerce profitability analysis. Uses a coordinator agent with specialized subagents to process complex data queries and provide business insights.

## Features

- **Coordinator Agent** - Plans tasks, delegates to subagents, and synthesizes results
- **SQL Expert Subagent** - Database queries across 7 e-commerce tables
- **Data Analyst Subagent** - Python code execution, pandas operations, visualizations
- **Web Researcher Subagent** - Market research and external data integration via Tavily API
- **Multi-table Database** - 128k+ transaction records across Amazon sales, inventory, pricing, and B2B data
- **Dynamic Schema Awareness** - Automatically adapts to database changes without manual updates
- **Business Intelligence** - Profitability analysis, trend identification, and actionable recommendations
- **Built-in Task Planning** - Automatic task breakdown via `write_todos`
- **Context Management** - Large result offloading to filesystem (`.deepagents/` directory)

## Available Tools

- **SQL Query Tool** - Safe database queries with injection protection across e-commerce tables
- **Python Code Execution** - Full pandas/numpy/matplotlib environment for data analysis and visualization
- **Web Search** - Real-time market research and external data via Tavily API

## Quick Start

### Prerequisites
- Python 3.12+
- Ollama installed with model: `gpt-oss:20b-cloud`
- Tavily API key (get from [tavily.com](https://tavily.com))
- E-commerce dataset from Kaggle: [Unlock Profits with E-Commerce Sales Data](https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data)

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd autonomous-agent
```

2. Install dependencies
```bash
uv sync
```

3. Download and setup database
```bash
# Download the Kaggle dataset and extract CSV files to csv-data/ folder
# https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data

# Initialize SQLite database from CSV files
python setup_database.py
```

4. Set up environment variables
```bash
echo "TAVILY_API_KEY=your_api_key_here" > .env
```

5. Run the system
```bash
uv run python agents.py
```

## How It Works

The system uses a coordinator-subagent architecture powered by the [deepagents](https://pypi.org/project/deepagents/) framework:

![Agent Graph](images/agent_graph.png)

**Flow Overview:**
- **Coordinator** - Analyzes queries, creates task plans, delegates to subagents, synthesizes results
- **SQL Expert** (`sql-expert`) - Database queries and e-commerce data retrieval
- **Data Analyst** (`data-analyst`) - Python code execution, pandas operations, visualizations
- **Web Researcher** (`web-researcher`) - Market research and external data gathering

### Example Queries

**Simple Data Query:**
```
"Show me the top 10 selling products by revenue"
```
Coordinator delegates to sql-expert → Executes database query → Returns formatted results

**Complex Analysis:**
```
"Analyze profitability across all marketplaces, identify trends, and recommend optimization strategies"
```
Coordinator creates task plan → Delegates to sql-expert (data) + data-analyst (analysis) + web-researcher (market context) → Synthesizes business insights

## Architecture

- **agents.py** - Coordinator and subagent definitions using `create_deep_agent()`
- **tools.py** - Modular tool implementations (SQL queries, Python execution, web search)
- **database.py** - Database utilities and dynamic schema extraction
- **setup_database.py** - Database initialization from CSV files
- **generate_graph.py** - Agent graph visualization generator
- **deepagents** - Framework for multi-agent orchestration with built-in planning and context management
- **SQLite Database** - 7 tables with 128k+ e-commerce transaction records
- **Rich Console** - Enhanced transparency with syntax highlighting and execution details

## Configuration

### Model Requirements
- `gpt-oss:20b-cloud` - Coordinator and subagent model

### Environment Variables
```bash
TAVILY_API_KEY=your_tavily_api_key
```

## Development

### Code Quality
```bash
ruff check .    # Linting
ruff format .   # Formatting
```

### Extending the System
Add new data sources by:
1. Adding CSV files to `csv-data/` folder
2. Running `python setup_database.py` to regenerate database
3. Database schemas auto-update with dynamic injection

Add new subagents by:
1. Define subagent dict with required keys: `name`, `description`, `system_prompt`, `tools`
2. Add to the `subagents` list in `create_ecommerce_agent()`
3. Add corresponding entry in the coordinator's system prompt under "Available Subagents"

Add new tools by:
1. Define in `tools.py` with `@tool` decorator
2. Assign to the appropriate subagent's `tools` list

## License

MIT License
