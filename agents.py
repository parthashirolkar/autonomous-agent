#!/usr/bin/env python3
"""
Deep Agents Runner for E-commerce Data Analysis
Migrated from LangGraph StateGraph to deepagents framework
"""

import asyncio
import os
from dotenv import load_dotenv

from deepagents import create_deep_agent
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Import existing tools
from tools import web_search, execute_code, sql_query
from database import get_database_schema

load_dotenv()
console = Console()

# Models - using local Ollama models as in original agents.py
coordinator_model = ChatOllama(model="gpt-oss:20b-cloud", temperature=0.2)


def get_coordinator_system_prompt(enable_a2a: bool = False) -> str:
    """Build the coordinator system prompt with database schema."""
    db_schema = get_database_schema()

    a2a_section = ""
    if enable_a2a:
        a2a_section = """
## A2A Protocol Communication
You also have access to the `a2a_call` tool which lets you communicate with
remote A2A-compatible agents. Use this when:
- A task is better handled by an external specialized agent
- You need to query or collaborate with another agent system
- Cross-organizational agent cooperation is required

Provide the remote agent's URL and a clear message describing the task.
"""

    return f"""You are an expert e-commerce data analysis coordinator specializing in profitability analysis.

## Your Role
You coordinate complex data analysis tasks by:
1. Breaking down complex queries using the write_todos tool
2. Delegating specialized work to subagents
3. Synthesizing results into actionable business insights

## Available Subagents

### sql-expert
- **Use for**: Database queries, sales analysis, inventory checks, profitability calculations
- **Delegate when**: The task requires SQL queries against the e-commerce database
- **Example tasks**: "Get top 10 products by revenue", "Calculate profit margins by category"

### data-analyst
- **Use for**: Data processing, calculations, visualizations, statistical analysis
- **Delegate when**: The task requires Python code execution, pandas operations, or matplotlib charts
- **Example tasks**: "Create a revenue trend chart", "Calculate statistical summaries"

### web-researcher
- **Use for**: Market trends, competitor analysis, external data gathering
- **Delegate when**: The task requires current market information not in the database
- **Example tasks**: "Research e-commerce trends for 2025", "Find competitor pricing strategies"

## Database Context
{db_schema}

## Workflow Guidelines

1. **For simple queries** (single data retrieval):
   - Delegate directly to the appropriate subagent
   - Return the result with brief business context

2. **For complex queries** (multi-step analysis):
   - Use `write_todos` to create a task breakdown
   - Assign each task to the appropriate subagent (sql-expert, data-analyst, or web-researcher)
   - Execute tasks in logical order
   - Synthesize findings into a comprehensive business summary

3. **Context Management**:
   - Store large intermediate results in files using filesystem tools
   - Keep your context clean by having subagents return summaries, not raw data
   - Reference files when needed rather than keeping large data in memory

4. **Final Output**:
   - Always provide business-focused summaries
   - Include actionable recommendations
   - Use markdown formatting for clarity
   - Highlight key metrics and insights

## Important Notes
- The database contains real e-commerce transaction data (128k+ rows in amazon_sale_report)
- Always use LIMIT clauses for large queries
- Focus on profitability insights and actionable recommendations
{a2a_section}
"""


# Define subagents
sql_expert_subagent = {
    "name": "sql-expert",
    "description": "Specialized agent for database queries and e-commerce data retrieval. Use for SQL queries, sales analysis, inventory checks, and profitability calculations from the e-commerce database.",
    "system_prompt": f"""You are an SQL Expert specializing in e-commerce profitability analysis.

{get_database_schema()}

## Your Capabilities
- Execute SELECT queries against the e-commerce database using the sql_query tool
- Analyze sales data, inventory levels, and pricing information
- Calculate profitability metrics and identify trends

## Guidelines
1. Always use LIMIT clauses for large tables (amazon_sale_report has 128k+ rows)
2. Convert TEXT amounts to REAL: CAST(amount AS REAL)
3. Filter by status='Shipped - Delivered to Buyer' for confirmed sales
4. Provide business insights along with query results

## Output Format
Return a concise summary of your findings including:
- Key metrics discovered
- Notable patterns or anomalies
- Relevant data points (keep to essential rows only)

Do NOT return raw query results with hundreds of rows. Summarize the data.""",
    "tools": [sql_query],
}

data_analyst_subagent = {
    "name": "data-analyst",
    "description": "Specialized agent for data processing, calculations, and visualizations. Use for Python code execution, pandas operations, statistical analysis, and creating charts with matplotlib/seaborn.",
    "system_prompt": """You are a Data Analysis Expert specializing in e-commerce profitability analysis.

## Your Capabilities
- Execute Python code for data analysis using the execute_code tool
- Use pandas for data manipulation and aggregation
- Create visualizations with matplotlib and seaborn
- Perform statistical analysis and calculations
- Access the SQLite database via sqlite3 and DB_PATH

## Guidelines
1. Write clean, efficient Python code
2. Use pandas for data manipulation
3. Create clear, informative visualizations when helpful
4. Store a 'result' variable to return your findings
5. Handle errors gracefully

## Output Format
Return a concise summary including:
- Key findings from your analysis
- Any visualizations created (mention filename if saved)
- Statistical insights
- Recommendations based on the data

Keep your response focused and actionable.""",
    "tools": [execute_code],
}

web_researcher_subagent = {
    "name": "web-researcher",
    "description": "Specialized agent for gathering external market information. Use for researching market trends, competitor analysis, industry benchmarks, and current e-commerce best practices.",
    "system_prompt": """You are a Web Research Expert specializing in e-commerce market intelligence.

## Your Capabilities
- Search the web for current market information using the web_search tool
- Research competitor strategies and pricing
- Find industry benchmarks and trends
- Gather external data to complement internal analysis

## Guidelines
1. Focus searches on actionable business intelligence
2. Verify information from multiple sources when possible
3. Distinguish between factual data and opinions
4. Note the recency of information found

## Output Format
Return a concise research summary including:
- Key findings with source attribution
- Relevant trends or benchmarks
- Actionable recommendations
- Any limitations or caveats about the data

Keep your response focused on what's relevant to e-commerce profitability.""",
    "tools": [web_search],
}


def create_ecommerce_agent(enable_a2a: bool = False) -> tuple:
    """Create and configure the deep agent for e-commerce analysis.

    Args:
        enable_a2a: If True, register the a2a_call tool so the coordinator
            can communicate with remote A2A-compatible agents.

    Returns:
        Tuple of (agent, a2a_tool) where a2a_tool is the A2AClientTool
        if enable_a2a is True, otherwise None.
    """
    a2a_tool = None
    coordinator_tools: list = []

    if enable_a2a:
        from a2a_bridge.client_tool import create_a2a_call_tool

        a2a_tool = create_a2a_call_tool()
        coordinator_tools.append(a2a_tool)

    return (
        create_deep_agent(
            model=coordinator_model,
            tools=coordinator_tools,
            system_prompt=get_coordinator_system_prompt(enable_a2a=enable_a2a),
            subagents=[
                sql_expert_subagent,
                data_analyst_subagent,
                web_researcher_subagent,
            ],
        ),
        a2a_tool,
    )


async def main():
    """Main entry point for the deep agents runner."""
    enable_a2a = os.getenv("ENABLE_A2A", "false").lower() in ("true", "1", "yes")

    console.print(
        Panel.fit(
            "[bold magenta]🏪 E-commerce Data Analysis Helper (DeepAgents)[/bold magenta]\n"
            "[dim]Specialized multi-agent system for profitability analysis[/dim]\n"
            "[dim cyan]Powered by deepagents framework[/dim cyan]"
            + ("\n[dim green]A2A protocol: enabled[/dim green]" if enable_a2a else ""),
            border_style="magenta",
        )
    )

    # Create the agent
    agent, _ = create_ecommerce_agent(enable_a2a=enable_a2a)

    # Initialize conversation state
    state = {"messages": []}

    console.print(
        "\n[bold cyan]Enter your queries below. Type 'exit' or 'quit' to end.[/bold cyan]\n"
    )

    while True:
        try:
            # Get user input
            query = console.input("[bold]Query:[/bold] ").strip()

            # Check for exit commands
            if query.lower() in ["exit", "quit", "q"]:
                console.print("\n[bold yellow]👋 Goodbye![/bold yellow]")
                break

            # Skip empty queries
            if not query:
                continue

            console.print()

            # Add user message to state
            state["messages"].append({"role": "user", "content": query})

            console.print("[bold green]🚀 Processing query...[/bold green]")
            console.print()

            # Run the agent
            result = await agent.ainvoke(
                state,
                {"recursion_limit": 100},
            )

            # Update state with new messages
            state = result

            # Display final result
            console.print()
            console.print(
                Panel.fit(
                    "[bold yellow]📊 Analysis Result[/bold yellow]",
                    border_style="yellow",
                )
            )
            console.print()

            # Get the final message content
            final_message = result["messages"][-1]
            if hasattr(final_message, "content"):
                markdown_content = Markdown(final_message.content)
                console.print(markdown_content)
            else:
                console.print(str(final_message))

            console.print()
            console.print("[bold green]✅ Query completed![/bold green]\n")

        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]👋 Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]❌ Error: {str(e)}[/red]\n")


if __name__ == "__main__":
    asyncio.run(main())
