from typing_extensions import TypedDict, Annotated, List, Literal, Dict, Any
import asyncio
import json
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.markdown import Markdown

load_dotenv()
console = Console()


# Tool Definitions
# Create Tavily search tool
tavily_search = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
)


@tool
async def web_search(query: str) -> str:
    """Search the internet for current information using Tavily search API."""
    try:
        console.print(f"[yellow]🔍 Searching web for: {query}[/yellow]")
        result = await tavily_search.ainvoke({"query": query})
        return f"Web search results for '{query}':\n{result}"
    except Exception as e:
        return f"Web search failed: {str(e)}"


@tool
async def http_request(url: str, method: str = "GET", data: str = None) -> str:
    """Make HTTP requests to external APIs."""
    try:
        console.print(f"[yellow]🌐 Making {method} request to: {url}[/yellow]")

        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
        elif method.upper() == "POST":
            payload = json.loads(data) if data else {}
            response = requests.post(url, json=payload, timeout=10)
        else:
            return f"Unsupported HTTP method: {method}"

        return f"HTTP {response.status_code}: {response.text[:500]}..."
    except Exception as e:
        return f"HTTP request failed: {str(e)}"


@tool
async def execute_code(code: str) -> str:
    """Execute Python code snippets safely."""
    try:
        console.print("[yellow]🐍 Executing Python code[/yellow]")

        # Create a safe execution environment
        namespace = {"__builtins__": {}}

        # Add safe built-ins
        for item in [
            "len",
            "str",
            "int",
            "float",
            "list",
            "dict",
            "range",
            "sum",
            "min",
            "max",
        ]:
            namespace[item] = getattr(__builtins__, item)

        # Execute code
        exec(f"result = {code}", namespace)
        return str(namespace.get("result", "Code executed successfully"))
    except Exception as e:
        return f"Code execution failed: {str(e)}"


@tool
async def get_time_info(timezone_name: str = "UTC") -> str:
    """Get current time and date information."""
    try:
        console.print(f"[yellow]⏰ Getting time info for: {timezone_name}[/yellow]")

        if timezone_name.upper() == "UTC":
            current_time = datetime.now(timezone.utc)
        else:
            # For simplicity, just return UTC time
            current_time = datetime.now(timezone.utc)

        return f"Current time ({timezone_name}): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    except Exception as e:
        return f"Time info failed: {str(e)}"


# Create tool list
tools = [web_search, http_request, execute_code, get_time_info]
tool_node = ToolNode(tools)


class StreamingCallback(BaseCallbackHandler):
    def __init__(self, agent_name: str, task_description: str = ""):
        self.agent_name = agent_name
        self.task_description = task_description
        self.tokens = []

    def on_llm_new_token(self, token: str, **_kwargs) -> None:
        self.tokens.append(token)
        console.print(f"[dim cyan]{token}[/dim cyan]", end="")


todo_agent = ChatOllama(model="gemma3:1b", temperature=0.1, num_ctx=2000)

# Main agent that can handle both tool calling and JSON output
main_llm = ChatOllama(model="qwen2.5:latest", temperature=0.4, num_ctx=10000)
main_agent_with_tools = main_llm.bind_tools(tools)
main_agent = main_llm | JsonOutputParser()


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


async def task_classifier(state: AgentState) -> AgentState:
    """Analyzes query complexity and determines if tools are needed."""
    console.print(
        Panel.fit(
            "[bold cyan]🎯 Task Classifier[/bold cyan]\n[dim]Analyzing query complexity and tool requirements...[/dim]",
            border_style="cyan",
        )
    )

    system_prompt = SystemMessage(
        content=(
            "You are a task analysis assistant that determines query complexity and required tools. "
            "Analyze the user query and return ONLY a JSON response with this format: "
            '{"complexity_level": "simple|moderate|complex", "requires_tools": true|false, "suggested_tools": ["tool1", "tool2"], "reasoning": "brief explanation"} '
            "\n\nAvailable tools: web_search, http_request, execute_code, get_time_info "
            "\n\nComplexity levels: "
            "- simple: Single action, direct answer possible "
            "- moderate: 2-3 steps, some tool usage "
            "- complex: Multi-step workflow, planning needed"
        )
    )

    human_message = HumanMessage(content=f"Query: {state['original_query']}")

    callback = StreamingCallback("Task Classifier", "Analyzing complexity")
    console.print("[cyan]Task Classifier:[/cyan] ", end="")

    response = await main_agent.ainvoke(
        [system_prompt, human_message], config={"callbacks": [callback]}
    )
    console.print()

    if isinstance(response, dict):
        state["complexity_level"] = response.get("complexity_level", "moderate")
        state["required_tools"] = response.get("suggested_tools", [])
        console.print(
            f"[cyan]📊 Complexity: {state['complexity_level']} | Tools: {state['required_tools']}[/cyan]"
        )
    else:
        state["complexity_level"] = "moderate"
        state["required_tools"] = []

    state["tool_results"] = []
    state["tool_call_history"] = []
    console.print()

    return state


async def planning_agent(state: AgentState) -> AgentState:
    console.print(
        Panel.fit(
            "[bold blue]🧠 Planning Agent[/bold blue]\n[dim]Breaking down the query into actionable tasks...[/dim]",
            border_style="blue",
        )
    )

    system_prompt = SystemMessage(
        content=(
            "You are a planning assistant that breaks down complex queries into specific, actionable tasks. "
            "Analyze the user's query and create a structured list of tasks, incorporating tool usage when needed. "
            f"Available tools: {', '.join([tool.name for tool in tools])} "
            f"Required tools for this query: {state.get('required_tools', [])} "
            "Return ONLY a JSON response with this format: "
            '{"tasks": [{"task": "specific task description", "status": "pending", "tools_needed": ["tool1", "tool2"]}, ...]} '
            "Each task should be atomic and specific. Include tools_needed array for tasks requiring tool usage."
        )
    )

    human_message = HumanMessage(content=f"Query: {state['original_query']}")

    callback = StreamingCallback("Planning Agent", "Creating task breakdown")
    console.print("[blue]Planning Agent:[/blue] ", end="")

    response = await main_agent.ainvoke(
        [system_prompt, human_message], config={"callbacks": [callback]}
    )
    console.print()

    if isinstance(response, dict) and "tasks" in response:
        state["todo_list"] = response["tasks"]
        console.print(f"[green]✅ Created {len(response['tasks'])} tasks[/green]")

        tree = Tree("📋 Task Breakdown")
        for i, task in enumerate(response["tasks"], 1):
            tree.add(f"[dim]{i}.[/dim] {task['task']}")
        console.print(tree)
    else:
        state["todo_list"] = [
            {"task": "Address the original query", "status": "pending"}
        ]
        console.print("[yellow]⚠️ Fallback: Created single task[/yellow]")

    state["current_task_index"] = 0
    state["research_results"] = []
    console.print()

    return state


async def tool_calling_agent(state: AgentState) -> AgentState:
    """Agent that can call tools and decide next steps."""
    current_task = (
        state["todo_list"][state["current_task_index"]]
        if state["todo_list"]
        else {"task": state["original_query"]}
    )
    task_num = state["current_task_index"] + 1 if state["todo_list"] else 1
    total_tasks = len(state["todo_list"]) if state["todo_list"] else 1

    console.print(
        Panel.fit(
            f"[bold green]Tool-Calling Agent[/bold green] [{task_num}/{total_tasks}]\n[dim]Working on: {current_task['task']}[/dim]",
            border_style="green",
        )
    )

    system_prompt = SystemMessage(
        content=(
            "You are an autonomous research assistant with access to tools. "
            "Complete the given task using any tools you need. "
            "Available tools: web_search, http_request, execute_code, get_time_info. "
            "Call tools when needed to get accurate information. "
            "After calling tools, provide a concise final answer."
        )
    )

    human_message = HumanMessage(content=f"Task: {current_task['task']}")

    # Build message history for context
    messages = [system_prompt] + state["messages"] + [human_message]

    callback = StreamingCallback("Tool Agent", current_task["task"])
    console.print(
        f"[green]Tool-Calling Agent ({task_num}/{total_tasks}):[/green] ", end=""
    )

    # Agent will decide to call tools or respond directly
    response = await main_agent_with_tools.ainvoke(
        messages, config={"callbacks": [callback]}
    )
    console.print()

    # Add the response to state messages
    state["messages"].append(response)

    return state


def should_continue(state: AgentState) -> Literal["tools", "continue"]:
    """Determine if we should call tools or continue."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
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


async def task_executor(state: AgentState) -> AgentState:
    if state["current_task_index"] >= len(state["todo_list"]):
        return state

    current_task = state["todo_list"][state["current_task_index"]]
    task_num = state["current_task_index"] + 1
    total_tasks = len(state["todo_list"])

    console.print(
        Panel.fit(
            f"[bold green]⚡ Task Executor[/bold green] [{task_num}/{total_tasks}]\n[dim]Working on: {current_task['task']}[/dim]",
            border_style="green",
        )
    )

    system_prompt = SystemMessage(
        content=(
            "You are a concise research assistant. Complete the specific task with key facts only. "
            "CONSTRAINTS: "
            "- Maximum 3-4 sentences per response "
            "- Focus on essential information only "
            "- No repetition or filler text "
            "- Direct, factual answers "
            "- Bullet points if listing multiple items"
        )
    )

    human_message = HumanMessage(
        content=(
            f"Task: {current_task['task']}\n"
            f"Provide ONLY the essential facts. Be concise and direct."
        )
    )

    callback = StreamingCallback("Task Executor", current_task["task"])
    console.print(f"[green]Task Executor ({task_num}/{total_tasks}):[/green] ", end="")

    response = await todo_agent.ainvoke(
        [system_prompt, human_message], config={"callbacks": [callback]}
    )
    console.print()

    state["research_results"].append(
        f"Task: {current_task['task']}\nResult: {response.content}"
    )
    state["todo_list"][state["current_task_index"]]["status"] = "completed"
    state["current_task_index"] += 1

    console.print(f"[green]✅ Task {task_num} completed[/green]")
    console.print()

    return state


def complexity_router(state: AgentState) -> Literal["tool_calling_agent", "planning"]:
    """Route based on task complexity."""
    complexity = state.get("complexity_level", "moderate")
    if complexity == "simple":
        return "tool_calling_agent"  # Direct tool execution
    else:
        return "planning"  # Complex tasks need planning


def progress_checker(state: AgentState) -> Literal["tool_calling_agent", "end"]:
    if not state["todo_list"]:
        return "end"  # Simple query, already completed
    return (
        "tool_calling_agent"
        if state["current_task_index"] < len(state["todo_list"])
        else "end"
    )


async def summarizer_agent(state: AgentState) -> AgentState:
    console.print(
        Panel.fit(
            "[bold purple]📝 Summarizer Agent[/bold purple]\n[dim]Compiling final comprehensive summary...[/dim]",
            border_style="purple",
        )
    )

    system_prompt = SystemMessage(
        content=(
            "You are a summarization assistant that creates comprehensive final summaries in markdown format. "
            "You will be given research results from multiple completed tasks. "
            "Create a well-structured, comprehensive summary using proper markdown formatting: "
            "- Use # for main headings, ## for subheadings "
            "- Use **bold** for key terms and important points "
            "- Use bullet points (-) or numbered lists for multiple items "
            "- Use > for important quotes or highlights "
            "Organize the information logically and ensure all key points are covered. "
            "Make sure your answer is coherent and stands on its own. The goal is to abstract the research process from the user and provide a clear, well-formatted final answer."
        )
    )

    research_summary = "\n\n".join(state["research_results"])
    human_message = HumanMessage(
        content=(
            f"Original Query: {state['original_query']}\n\n"
            f"Research Results:\n{research_summary}\n\n"
            f"Please provide a comprehensive summary that fully addresses the original query."
        )
    )

    callback = StreamingCallback("Summarizer Agent", "Creating final summary")
    console.print("[purple]Summarizer Agent:[/purple] ", end="")

    response = await todo_agent.ainvoke(
        [system_prompt, human_message], config={"callbacks": [callback]}
    )
    console.print()

    state["final_summary"] = response.content

    console.print("[purple]✅ Final summary completed[/purple]")
    console.print()

    return state


agent = StateGraph(AgentState)

# Add nodes
agent.add_node("task_classifier", task_classifier)
agent.add_node("planning", planning_agent)
agent.add_node("tool_calling_agent", tool_calling_agent)
agent.add_node("tools", tool_node)
agent.add_node("continue", continue_after_tools)
agent.add_node("summarizer", summarizer_agent)

# Add edges
agent.add_edge(START, "task_classifier")
agent.add_conditional_edges(
    "task_classifier",
    complexity_router,
    {"tool_calling_agent": "tool_calling_agent", "planning": "planning"},
)
agent.add_edge("planning", "tool_calling_agent")

# Tool calling flow
agent.add_conditional_edges(
    "tool_calling_agent",
    should_continue,
    {"tools": "tools", "continue": "continue"},
)
agent.add_edge("tools", "continue")

# After tool execution, check progress
agent.add_conditional_edges(
    "continue",
    progress_checker,
    {"tool_calling_agent": "tool_calling_agent", "end": "summarizer"},
)
agent.add_edge("summarizer", END)


async def main():
    console.print(
        Panel.fit(
            "[bold magenta] Autonomous Multi-Agent System[/bold magenta]\n[dim]Breaking down complex tasks into manageable steps[/dim]",
            border_style="magenta",
        )
    )

    query = """Tell me a random fact about cats from the cat fact API: https://cat-fact.herokuapp.com/facts"""
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
    }

    with console.status(
        "[bold green]Initializing autonomous agent system..."
    ) as status:
        compiled_agent = agent.compile()
        status.update("[bold blue]Starting autonomous execution...[/bold blue]")

    console.print("[bold green]🚀 Starting autonomous execution...[/bold green]")
    console.print()

    final_state = await compiled_agent.ainvoke(initial_state)

    console.print(
        Panel.fit("[bold yellow]📋 Final Summary[/bold yellow]", border_style="yellow")
    )
    console.print()

    markdown_content = Markdown(final_state["final_summary"])
    console.print(markdown_content)

    console.print(
        f"[bold green]✨ Autonomous execution completed! Processed {len(final_state['todo_list'])} tasks.[/bold green]"
    )


if __name__ == "__main__":
    asyncio.run(main())
