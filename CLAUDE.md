# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This is a specialized data analysis helper agent system built with LangGraph for e-commerce profitability analysis. The system features a main data analysis coordinator and an SQL expert subagent that work together to process complex data queries. Uses local Ollama models (qwen2.5:latest for main agents, llama3.2:1b for summarization) and integrates with e-commerce databases.

## Core Architecture

### Agent Flow Structure
The system implements a data-focused routing architecture:

```
User Query → Data Task Classifier → [Simple: Main Agent | Complex: Planning → Agent Router]
                                                    ↓
Agent Router → [SQL Expert: Database Queries | Main Agent: Analysis & Visualization]
                                ↓
Continue Processing → Progress Check → [More Tasks | Data Summarizer] → End
```

### Key Components

**agents.py** - Complete data analysis system:
- **AgentState**: Enhanced with `sql_results`, `data_analysis_results`, `needs_sql_expert` for data workflow
- **Data Task Classifier**: Analyzes queries for SQL needs, web search requirements, and complexity
- **Planning Agent**: Creates data analysis task breakdowns with agent assignments (sql_expert vs main)
- **SQL Expert Agent**: Specialized for database queries with e-commerce domain knowledge
- **Main Data Agent**: Handles calculations, visualizations, and web research
- **Data Summarizer**: Creates business-focused analysis summaries with actionable insights

### Database Integration
- **SQLite Database**: Auto-generated from CSV files containing e-commerce transaction data
- **7 Data Tables**: Amazon sales, inventory, pricing, B2B transactions, P&L, expenses, logistics
- **SQL Query Tool**: Safe SELECT-only queries with injection protection and result formatting
- **Schema Awareness**: SQL agent has complete understanding of all table structures

### Tool Integration
- **SQL Query Tool**: Execute database queries with safety checks and formatted results
- **Enhanced Code Execution**: Full pandas/numpy/matplotlib access for data analysis and visualization
- **Tavily Search**: Web search for market trends and external data research

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

### Database Setup
```bash
# Download e-commerce dataset from Kaggle first:
# https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data
# Extract CSV files to csv-data/ folder

# Initialize database from CSV files
python setup_database.py
```

### Model Configuration
- **Main agents**: `qwen2.5:latest` with 10k context (SQL) and 8k context (main)
- **Summarizer**: `llama3.2:1b` for efficiency in final analysis generation
- **Tool binding**: Agents bound to specific tools (`main_agent_with_tools`, `sql_agent_with_tools`)

### Data Analysis Features
- **Multi-table queries**: Complex joins across sales, inventory, and pricing data
- **Profitability analysis**: Cross-marketplace revenue and cost analysis
- **Visualization support**: Matplotlib/seaborn integration for charts and insights
- **Business intelligence**: Actionable recommendations and trend analysis

## Extending the System

### Adding New Data Sources
1. Add CSV files to `csv-data/` folder
2. Run `setup_database.py` to regenerate database
3. Update SQL expert agent system prompt with new table descriptions
4. Database schemas auto-generated in `database_schemas.sql`

### Modifying Analysis Behavior
- **Data complexity logic**: Modify `complexity_router()` for different analysis types
- **SQL expert prompts**: Update domain knowledge in `sql_expert_agent()`
- **Analysis focus**: Adjust `summarizer_agent()` for different business metrics

### Architecture Patterns
- **Dual agent system**: SQL expert for data retrieval, main agent for processing
- **Safety-first**: SQL injection protection, read-only database access
- **Business-focused**: All outputs oriented toward actionable business insights
- **Scalable**: Easy to add new data sources and analysis types
- Use `uv` to run any/all python files and scripts.