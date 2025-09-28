#!/usr/bin/env python3
"""
Data Analysis Helper Agent System
Specialized for e-commerce profitability analysis with SQL expert subagent
"""

from typing_extensions import TypedDict, Annotated, List, Literal, Dict, Any
import asyncio
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph.message import add_messages
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.markdown import Markdown

# Import tools and database utilities
from tools import web_search, execute_code, sql_query
from database import get_database_schema

load_dotenv()
console = Console()

# Main agent tools (only web search and python execution)
main_tools = [web_search, execute_code]
main_tool_node = ToolNode(main_tools)

# SQL expert tools (only SQL query)
sql_tools = [sql_query]
sql_tool_node = ToolNode(sql_tools)


class StreamingCallback(BaseCallbackHandler):
    def __init__(self, agent_name: str, task_description: str = ""):
        self.agent_name = agent_name
        self.task_description = task_description
        self.tokens = []

    def on_llm_new_token(self, token: str, **_kwargs) -> None:
        self.tokens.append(token)
        console.print(f"[dim cyan]{token}[/dim cyan]", end="")


# Agent models
todo_agent = ChatOllama(model="llama3.2:1b", temperature=0.1, num_ctx=2000)
main_llm = ChatOllama(model="qwen2.5:latest", temperature=0.4, num_ctx=10000)
sql_llm = ChatOllama(model="qwen2.5:latest", temperature=0.2, num_ctx=8000)

# Bind tools to agents
main_agent_with_tools = main_llm.bind_tools(main_tools)
main_agent = main_llm | JsonOutputParser()
sql_agent_with_tools = sql_llm.bind_tools(sql_tools)


class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    original_query: str
    todo_list: List[Dict[str, Any]]
    current_task_index: int
    research_results: List[str]
    final_summary: str
    tool_results: List[Dict[str, Any]]
    complexity_level: str
    required_tools: List[str]
    tool_call_history: List[Dict[str, Any]]
    sql_results: List[Dict[str, Any]]
    data_analysis_results: List[str]
    needs_sql_expert: bool


async def task_classifier(state: AgentState) -> AgentState:
    """Analyzes data analysis query complexity and determines if SQL database access is needed."""
    console.print(
        Panel.fit(
            "[bold cyan]🎯 Data Analysis Task Classifier[/bold cyan]\n[dim]Analyzing query for data analysis requirements...[/dim]",
            border_style="cyan",
        )
    )

    system_prompt = SystemMessage(
        content=(
            "You are a data analysis task classifier for an e-commerce profitability analysis system. "
            "Analyze the user query and return ONLY a JSON response with this format: "
            '{"complexity_level": "simple|moderate|complex", "needs_sql_expert": true|false, "requires_web_search": true|false, "analysis_type": "category", "reasoning": "brief explanation"} '
            "\n\nAvailable data: E-commerce sales (Amazon, B2B), inventory, pricing across platforms, P&L data "
            "\n\nSQL Expert needed for: Database queries, sales analysis, inventory checks, profitability calculations "
            "\n\nWeb search needed for: Market trends, competitor analysis, external data "
            "\n\nComplexity levels: "
            "- simple: Single data query or calculation "
            "- moderate: Multi-table analysis, basic insights "
            "- complex: Advanced analytics, forecasting, comprehensive reports"
        )
    )

    human_message = HumanMessage(content=f"Query: {state['original_query']}")

    callback = StreamingCallback("Task Classifier", "Analyzing data requirements")
    console.print("[cyan]Data Task Classifier:[/cyan] ", end="")

    response = await main_agent.ainvoke(
        [system_prompt, human_message], config={"callbacks": [callback]}
    )
    console.print()

    if isinstance(response, dict):
        state["complexity_level"] = response.get("complexity_level", "moderate")
        state["needs_sql_expert"] = response.get("needs_sql_expert", True)
        state["required_tools"] = []
        if response.get("requires_web_search", False):
            state["required_tools"].append("web_search")

        console.print(
            f"[cyan]📊 Complexity: {state['complexity_level']} | SQL Expert: {state['needs_sql_expert']} | Web Search: {response.get('requires_web_search', False)}[/cyan]"
        )
    else:
        state["complexity_level"] = "moderate"
        state["needs_sql_expert"] = True
        state["required_tools"] = []

    state["tool_results"] = []
    state["tool_call_history"] = []
    state["sql_results"] = []
    state["data_analysis_results"] = []
    console.print()

    return state


async def planning_agent(state: AgentState) -> AgentState:
    console.print(
        Panel.fit(
            "[bold blue]🧠 Data Analysis Planning Agent[/bold blue]\n[dim]Breaking down the data analysis query into actionable tasks...[/dim]",
            border_style="blue",
        )
    )

    system_prompt = SystemMessage(
        content=(
            "You are a data analysis planning assistant for e-commerce profitability analysis. "
            "Break down complex data queries into specific, actionable tasks. "
            f"SQL Expert available: {state.get('needs_sql_expert', False)} "
            f"Web search needed: {'web_search' in state.get('required_tools', [])} "
            "Return ONLY a JSON response with this format: "
            '{"tasks": [{"task": "specific task description", "status": "pending", "agent_needed": "main|sql_expert", "tools_needed": ["tool1"]}, ...]} '
            "\n\nFor data analysis tasks: "
            "- Use 'sql_expert' agent for database queries, sales analysis, inventory checks "
            "- Use 'main' agent for calculations, visualizations, web search, data processing "
            "Each task should be atomic and specify which agent should handle it."
        )
    )

    human_message = HumanMessage(content=f"Query: {state['original_query']}")

    callback = StreamingCallback("Planning Agent", "Creating data analysis breakdown")
    console.print("[blue]Data Planning Agent:[/blue] ", end="")

    response = await main_agent.ainvoke(
        [system_prompt, human_message], config={"callbacks": [callback]}
    )
    console.print()

    if isinstance(response, dict) and "tasks" in response:
        state["todo_list"] = response["tasks"]
        console.print(
            f"[green]✅ Created {len(response['tasks'])} data analysis tasks[/green]"
        )

        tree = Tree("📋 Data Analysis Task Breakdown")
        for i, task in enumerate(response["tasks"], 1):
            agent_icon = "🗃️" if task.get("agent_needed") == "sql_expert" else "🐍"
            tree.add(f"[dim]{i}.[/dim] {agent_icon} {task['task']}")
        console.print(tree)
    else:
        state["todo_list"] = [
            {
                "task": "Analyze the data query",
                "status": "pending",
                "agent_needed": "main",
            }
        ]
        console.print("[yellow]⚠️ Fallback: Created single analysis task[/yellow]")

    state["current_task_index"] = 0
    state["research_results"] = []
    console.print()

    return state


async def sql_expert_agent(state: AgentState) -> AgentState:
    """SQL Expert agent specialized for database queries and e-commerce analysis."""
    current_task = state["todo_list"][state["current_task_index"]]
    task_num = state["current_task_index"] + 1
    total_tasks = len(state["todo_list"])

    console.print(
        Panel.fit(
            f"[bold yellow]🗃️ SQL Expert Agent[/bold yellow] [{task_num}/{total_tasks}]\n[dim]Working on: {current_task['task']}[/dim]",
            border_style="yellow",
        )
    )

    # Get dynamic database schema
    db_schema = get_database_schema()

    system_prompt = SystemMessage(
        content=(
            "You are an SQL Expert specializing in e-commerce profitability analysis. "
            "You have access to a comprehensive e-commerce database. "
            f"\n\n{db_schema}\n\n"
            "Use the sql_query tool to execute SELECT queries for data analysis. "
            "Focus on profitability, sales trends, inventory analysis, and marketplace performance. "
            "Always reference the exact column names and data types shown in the schema above. "
            "Provide business insights along with your query results."
        )
    )

    human_message = HumanMessage(content=f"Task: {current_task['task']}")

    # Build message history for context
    messages = [system_prompt] + state["messages"] + [human_message]

    callback = StreamingCallback("SQL Expert", current_task["task"])
    console.print(
        f"[yellow]SQL Expert Agent ({task_num}/{total_tasks}):[/yellow] ", end=""
    )

    # SQL Expert will decide to call SQL queries
    response = await sql_agent_with_tools.ainvoke(
        messages, config={"callbacks": [callback]}
    )
    console.print()

    # Add the response to state messages
    state["messages"].append(response)

    return state


async def main_data_agent(state: AgentState) -> AgentState:
    """Main data analysis agent that handles calculations, visualizations, and web search."""
    current_task = (
        state["todo_list"][state["current_task_index"]]
        if state["todo_list"]
        else {"task": state["original_query"]}
    )
    task_num = state["current_task_index"] + 1 if state["todo_list"] else 1
    total_tasks = len(state["todo_list"]) if state["todo_list"] else 1

    console.print(
        Panel.fit(
            f"[bold green]🐍 Main Data Analysis Agent[/bold green] [{task_num}/{total_tasks}]\n[dim]Working on: {current_task['task']}[/dim]",
            border_style="green",
        )
    )

    system_prompt = SystemMessage(
        content=(
            "You are a data analysis expert specializing in e-commerce profitability analysis. "
            "You have access to: web_search for market data and execute_code for data processing. "
            "For data processing tasks, you can: "
            "- Use pandas to analyze data and create calculations "
            "- Use matplotlib/seaborn for visualizations "
            "- Access SQLite database via sqlite3 and DB_PATH for additional queries "
            "- Perform statistical analysis and data transformations "
            "Always provide insights and interpretations with your analysis."
        )
    )

    human_message = HumanMessage(content=f"Task: {current_task['task']}")

    # Build message history for context
    messages = [system_prompt] + state["messages"] + [human_message]

    callback = StreamingCallback("Main Data Agent", current_task["task"])
    console.print(
        f"[green]Main Data Agent ({task_num}/{total_tasks}):[/green] ", end=""
    )

    # Agent will decide to call tools or respond directly
    response = await main_agent_with_tools.ainvoke(
        messages, config={"callbacks": [callback]}
    )
    console.print()

    # Add the response to state messages
    state["messages"].append(response)

    return state


def should_continue_main(state: AgentState) -> Literal["main_tools", "continue"]:
    """Determine if main agent should call tools or continue."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "main_tools"
    return "continue"


def should_continue_sql(state: AgentState) -> Literal["sql_tools", "continue"]:
    """Determine if SQL agent should call tools or continue."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "sql_tools"
    return "continue"


async def continue_after_tools(state: AgentState) -> AgentState:
    """Continue processing after tools have been called."""
    if state["todo_list"]:
        # Working through a todo list
        current_task = state["todo_list"][state["current_task_index"]]

        # Extract final answer from the conversation
        last_message = state["messages"][-1]
        result_content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        state["research_results"].append(
            f"Task: {current_task['task']}\nResult: {result_content}"
        )
        state["todo_list"][state["current_task_index"]]["status"] = "completed"
        state["current_task_index"] += 1

        console.print(f"[green]✅ Task {state['current_task_index']} completed[/green]")
    else:
        # Simple query completed
        last_message = state["messages"][-1]
        state["final_summary"] = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )
        console.print("[green]✅ Simple query completed[/green]")

    console.print()
    return state


def complexity_router(state: AgentState) -> Literal["main_data_agent", "planning"]:
    """Route based on task complexity."""
    complexity = state.get("complexity_level", "moderate")
    if complexity == "simple":
        return "main_data_agent"  # Direct execution
    else:
        return "planning"  # Complex tasks need planning


async def agent_router(state: AgentState) -> AgentState:
    """Route to appropriate agent based on current task."""
    # This is a pass-through node that just returns the state
    # The actual routing is handled by the conditional edge
    return state


def determine_agent(
    state: AgentState,
) -> Literal["main_data_agent", "sql_expert_agent"]:
    """Determine which agent should handle the current task."""
    if not state["todo_list"]:
        return "main_data_agent"

    current_task = state["todo_list"][state["current_task_index"]]
    agent_needed = current_task.get("agent_needed", "main")

    if agent_needed == "sql_expert":
        return "sql_expert_agent"
    else:
        return "main_data_agent"


def progress_checker(state: AgentState) -> Literal["agent_router", "summarizer"]:
    if not state["todo_list"]:
        return "summarizer"  # Simple query, go to summary
    return (
        "agent_router"
        if state["current_task_index"] < len(state["todo_list"])
        else "summarizer"
    )


async def summarizer_agent(state: AgentState) -> AgentState:
    console.print(
        Panel.fit(
            "[bold purple]📝 Data Analysis Summary[/bold purple]\n[dim]Compiling final comprehensive analysis...[/dim]",
            border_style="purple",
        )
    )

    system_prompt = SystemMessage(
        content=(
            "You are a data analysis summarization expert for e-commerce profitability analysis. "
            "Create a comprehensive final analysis summary in markdown format. "
            "Focus on business insights, actionable recommendations, and data-driven conclusions. "
            "Use proper markdown formatting with: "
            "- # for main headings (Analysis Summary, Key Findings, Recommendations) "
            "- ## for subheadings (Sales Trends, Profitability Analysis, etc.) "
            "- **bold** for key metrics and important insights "
            "- Bullet points (-) for actionable items "
            "- Tables for key data comparisons "
            "Make the analysis business-focused and actionable for e-commerce decision makers."
        )
    )

    research_summary = "\n\n".join(state["research_results"])
    human_message = HumanMessage(
        content=(
            f"Original Query: {state['original_query']}\n\n"
            f"Analysis Results:\n{research_summary}\n\n"
            f"Please provide a comprehensive business analysis summary with actionable insights."
        )
    )

    callback = StreamingCallback("Data Summarizer", "Creating final analysis")
    console.print("[purple]Data Analysis Summarizer:[/purple] ", end="")

    response = await todo_agent.ainvoke(
        [system_prompt, human_message], config={"callbacks": [callback]}
    )
    console.print()

    state["final_summary"] = response.content

    console.print("[purple]✅ Final analysis completed[/purple]")
    console.print()

    return state


# Create the data analysis state graph
agent = StateGraph(AgentState)

# Add nodes
agent.add_node("task_classifier", task_classifier)
agent.add_node("planning", planning_agent)
agent.add_node("agent_router", agent_router)
agent.add_node("main_data_agent", main_data_agent)
agent.add_node("sql_expert_agent", sql_expert_agent)
agent.add_node("main_tools", main_tool_node)
agent.add_node("sql_tools", sql_tool_node)
agent.add_node("continue", continue_after_tools)
agent.add_node("summarizer", summarizer_agent)

# Add edges
agent.add_edge(START, "task_classifier")

# Complexity routing
agent.add_conditional_edges(
    "task_classifier",
    complexity_router,
    {"main_data_agent": "main_data_agent", "planning": "planning"},
)

agent.add_edge("planning", "agent_router")

# Agent routing for task execution
agent.add_conditional_edges(
    "agent_router",
    determine_agent,
    {"main_data_agent": "main_data_agent", "sql_expert_agent": "sql_expert_agent"},
)

# Main agent tool flow
agent.add_conditional_edges(
    "main_data_agent",
    should_continue_main,
    {"main_tools": "main_tools", "continue": "continue"},
)
agent.add_edge("main_tools", "continue")

# SQL agent tool flow
agent.add_conditional_edges(
    "sql_expert_agent",
    should_continue_sql,
    {"sql_tools": "sql_tools", "continue": "continue"},
)
agent.add_edge("sql_tools", "continue")

# Progress checking
agent.add_conditional_edges(
    "continue",
    progress_checker,
    {"agent_router": "agent_router", "summarizer": "summarizer"},
)

agent.add_edge("summarizer", END)


async def main():
    console.print(
        Panel.fit(
            "[bold magenta]🏪 E-commerce Data Analysis Helper[/bold magenta]\n[dim]Specialized multi-agent system for profitability analysis[/dim]",
            border_style="magenta",
        )
    )

    query = """Give me an executive actionable summary of the Amazon Sale report table. First pull the data with an SQL query, then do some analysis on it."""
    console.print(f"[bold]Query:[/bold] {query}")
    console.print()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "original_query": query,
        "todo_list": [],
        "current_task_index": 0,
        "research_results": [],
        "final_summary": "",
        "tool_results": [],
        "complexity_level": "moderate",
        "required_tools": [],
        "tool_call_history": [],
        "sql_results": [],
        "data_analysis_results": [],
        "needs_sql_expert": True,
    }

    with console.status("[bold green]Initializing data analysis system...") as status:
        compiled_agent = agent.compile()
        status.update("[bold blue]Starting data analysis...[/bold blue]")

    console.print("[bold green]🚀 Starting e-commerce data analysis...[/bold green]")
    console.print()

    final_state = await compiled_agent.ainvoke(initial_state)

    console.print(
        Panel.fit(
            "[bold yellow]📊 Data Analysis Summary[/bold yellow]", border_style="yellow"
        )
    )
    console.print()

    markdown_content = Markdown(final_state["final_summary"])
    console.print(markdown_content)

    console.print(
        f"[bold green]✨ Data analysis completed! Processed {len(final_state['todo_list'])} tasks.[/bold green]"
    )


if __name__ == "__main__":
    asyncio.run(main())
