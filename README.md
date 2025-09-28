# E-Commerce Data Analysis Agent

A specialized data analysis helper agent system built with LangGraph for e-commerce profitability analysis. Features a main data analysis coordinator and an SQL expert subagent that work together to process complex data queries and provide business insights.

## Features

- **SQL Expert Agent** - Specialized subagent for complex database queries across 7 e-commerce tables
- **Data Analysis Coordinator** - Main agent for calculations, visualizations, and business insights
- **Multi-table Database** - 128k+ transaction records across Amazon sales, inventory, pricing, and B2B data
- **Dynamic Schema Awareness** - Automatically adapts to database changes without manual updates
- **Business Intelligence** - Profitability analysis, trend identification, and actionable recommendations
- **Real Web Search** - Market research and external data integration via Tavily API
- **Enhanced Visualizations** - Matplotlib/seaborn integration for charts and data insights

## Available Tools

- **SQL Query Tool** - Safe database queries with injection protection across e-commerce tables
- **Python Code Execution** - Full pandas/numpy/matplotlib environment for data analysis and visualization
- **Web Search** - Real-time market research and external data via Tavily API

## Quick Start

### Prerequisites
- Python 3.12+
- Ollama installed with models: `qwen2.5:latest` and `llama3.2:1b`
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
pip install -r requirements.txt
# or using uv (recommended for speed)
uv pip install -r requirements.txt
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
# Create .env file
echo "TAVILY_API_KEY=your_api_key_here" > .env
```

5. Run the system
```bash
python agents.py
```

## How It Works

The system uses a data-focused routing architecture:

![Agent Graph](images/agent_graph.png)

**Flow Overview:**
- **Data Task Classifier** - Analyzes queries for SQL needs, web search requirements, and complexity
- **Planning Agent** - Creates data analysis task breakdowns with agent assignments
- **SQL Expert Agent** - Specialized for database queries with e-commerce domain knowledge
- **Main Data Agent** - Handles calculations, visualizations, and web research
- **Agent Router** - Routes between SQL expert and main agent based on task requirements
- **Data Summarizer** - Creates business-focused analysis summaries with actionable insights

### Example Queries

**Simple Data Query:**
```
"Show me the top 10 selling products by revenue"
```
Routes to SQL Expert -> Executes database query -> Returns formatted results

**Complex Analysis:**
```
"Analyze profitability across all marketplaces, identify trends, and recommend optimization strategies"
```
Routes through planning -> SQL Expert (data retrieval) -> Main Agent (analysis/visualization) -> Summarizer (business insights)

## Architecture

- **agents.py** - Complete data analysis system with SQL expert and main data agents
- **tools.py** - Modular tool implementations (SQL queries, Python execution, web search)
- **database.py** - Database utilities and dynamic schema extraction
- **setup_database.py** - Database initialization from CSV files
- **AgentState** - Enhanced state with SQL results and data analysis workflow
- **LangGraph** - Orchestrates dual-agent workflow and intelligent routing
- **SQLite Database** - 7 tables with 128k+ e-commerce transaction records
- **Rich Console** - Enhanced transparency with syntax highlighting and execution details

## Configuration

### Model Requirements
- `qwen2.5:latest` - SQL Expert and Main Data agents (10k/8k context)
- `llama3.2:1b` - Data Summarizer agent (lightweight, efficient)

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

Add new tools by:
1. Defining in `tools.py` with `@tool` decorator
2. Binding to appropriate agent (SQL expert or main agent)
3. System automatically integrates via enhanced ToolNode

## License

MIT License