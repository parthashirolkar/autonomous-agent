"""A2A server that wraps a deepagents CompiledStateGraph."""

from __future__ import annotations

import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.apps import A2AFastAPIApplication
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Part,
    TextPart,
)
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from a2a_bridge.adapters import langchain_to_a2a

logger = logging.getLogger(__name__)


class DeepAgentA2AExecutor(AgentExecutor):
    """Wraps a CompiledStateGraph as an A2A-compatible agent executor."""

    def __init__(self, graph: CompiledStateGraph) -> None:
        self.graph = graph

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_text = context.get_user_input()
        task_id = context.task_id or "unknown"
        context_id = context.context_id or "default"

        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()

        try:
            config = {"configurable": {"thread_id": context_id}}
            result = await self.graph.ainvoke(
                {"messages": [HumanMessage(content=user_text)]},
                config,
            )

            final_msg = result["messages"][-1]
            a2a_msg = langchain_to_a2a(final_msg)
            a2a_msg.task_id = task_id
            a2a_msg.context_id = context_id

            await updater.complete(a2a_msg)
        except Exception as e:
            logger.exception("Agent execution failed")
            error_msg = updater.new_agent_message(
                parts=[Part(root=TextPart(text=f"Agent error: {e}"))]
            )
            error_msg.task_id = task_id
            error_msg.context_id = context_id
            await updater.failed(error_msg)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or "unknown"
        context_id = context.context_id or "default"
        updater = TaskUpdater(event_queue, task_id, context_id)
        msg = updater.new_agent_message(
            parts=[Part(root=TextPart(text="Task cancelled"))]
        )
        msg.task_id = task_id
        msg.context_id = context_id
        await updater.cancel(msg)


def create_a2a_app(
    graph: CompiledStateGraph,
    name: str = "deepagent",
    description: str = "A deepagents-based multi-agent system",
    url: str = "http://localhost:8000",
    version: str = "0.1.0",
    skills: list[AgentSkill] | None = None,
) -> A2AFastAPIApplication:
    """Build an A2A FastAPI application from a CompiledStateGraph."""
    agent_card = AgentCard(
        name=name,
        description=description,
        url=url,
        version=version,
        capabilities=AgentCapabilities(
            streaming=False,
            push_notifications=False,
            state_transition_history=False,
        ),
        skills=skills
        or [
            AgentSkill(
                id="general",
                name="General Query",
                description="Handle general queries via the deepagents multi-agent system",
                tags=["multi-agent", "analysis"],
            )
        ],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )

    handler = DefaultRequestHandler(
        agent_executor=DeepAgentA2AExecutor(graph),
        task_store=InMemoryTaskStore(),
    )

    return A2AFastAPIApplication(agent_card, handler)
