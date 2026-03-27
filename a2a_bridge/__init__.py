"""A2A protocol integration for deepagents.

Expose your deepagents-based agent as an A2A server, or call remote
A2A agents as tools from within your coordinator.
"""

from a2a_bridge.adapters import (
    a2a_to_langchain,
    langchain_to_a2a,
    langchain_messages_to_a2a,
)
from a2a_bridge.client_tool import create_a2a_call_tool
from a2a_bridge.server import DeepAgentA2AExecutor, create_a2a_app

__all__ = [
    "DeepAgentA2AExecutor",
    "a2a_to_langchain",
    "create_a2a_app",
    "create_a2a_call_tool",
    "langchain_messages_to_a2a",
    "langchain_to_a2a",
]
