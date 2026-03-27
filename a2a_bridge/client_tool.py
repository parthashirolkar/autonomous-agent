"""A2A client tool — lets the coordinator call remote A2A agents."""

from __future__ import annotations

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.types import TaskState
from a2a.utils.message import new_agent_text_message
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class A2ACallInput(BaseModel):
    agent_url: str = Field(
        ...,
        description="The URL of the remote A2A agent (e.g. http://localhost:8001)",
    )
    message: str = Field(
        ...,
        description="The message to send to the remote A2A agent",
    )


async def _call_a2a_agent(agent_url: str, message: str) -> str:
    """Connect to a remote A2A agent, send a message, and return the response."""
    httpx_client = httpx.AsyncClient(timeout=120.0)
    try:
        client = await ClientFactory.connect(
            agent_url,
            client_config=ClientConfig(httpx_client=httpx_client),
        )
        request = new_agent_text_message(text=message)
        result_parts: list[str] = []

        async for event in client.send_message(request):
            if isinstance(event, tuple):
                task, update = event
                if (
                    task.status.message
                    and task.status.message.parts
                    and task.status.state == TaskState.completed
                ):
                    for part in task.status.message.parts:
                        if hasattr(part.root, "text"):
                            result_parts.append(part.root.text)
                elif (
                    update
                    and hasattr(update, "message")
                    and update.message
                    and update.message.parts
                ):
                    for part in update.message.parts:
                        if hasattr(part.root, "text"):
                            result_parts.append(part.root.text)
            elif hasattr(event, "parts"):
                for part in event.parts:
                    if hasattr(part.root, "text"):
                        result_parts.append(part.root.text)

        return (
            "\n".join(result_parts)
            if result_parts
            else "No response from remote agent."
        )
    finally:
        await httpx_client.aclose()


def create_a2a_call_tool() -> StructuredTool:
    """Create a LangChain tool for calling remote A2A agents."""
    return StructuredTool.from_function(
        coroutine=_call_a2a_agent,
        name="a2a_call",
        description=(
            "Call a remote A2A-compatible agent. Provide the agent URL and a message. "
            "Use this when you need to communicate with an external agent that speaks "
            "the A2A protocol."
        ),
        args_schema=A2ACallInput,
    )
