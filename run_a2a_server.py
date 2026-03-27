#!/usr/bin/env python3
"""Run the deepagents-based e-commerce agent as an A2A server."""

import logging
import os

import uvicorn
from a2a.types import AgentSkill
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

HOST = os.getenv("A2A_HOST", "0.0.0.0")
PORT = int(os.getenv("A2A_PORT", "8000"))


def create_app():
    from agents import create_ecommerce_agent
    from a2a_bridge.server import create_a2a_app

    graph, _ = create_ecommerce_agent()
    return create_a2a_app(
        graph=graph,
        name="ecommerce-analyst",
        description=(
            "E-commerce data analysis multi-agent system. "
            "Specializes in profitability analysis, sales trends, "
            "inventory optimization, and market research using "
            "SQL, Python, and web search capabilities."
        ),
        url=f"http://localhost:{PORT}",
        skills=[
            AgentSkill(
                id="sql-analysis",
                name="SQL Analysis",
                description="Query and analyze e-commerce database for sales, inventory, and profitability insights",
                tags=["sql", "database", "analytics", "profitability"],
            ),
            AgentSkill(
                id="data-visualization",
                name="Data Visualization",
                description="Create charts, statistical analyses, and data processing with Python/pandas/matplotlib",
                tags=["python", "pandas", "matplotlib", "visualization"],
            ),
            AgentSkill(
                id="web-research",
                name="Web Research",
                description="Research market trends, competitor strategies, and industry benchmarks",
                tags=["web", "research", "market", "trends"],
            ),
        ],
    ).build()


if __name__ == "__main__":
    uvicorn.run("run_a2a_server:app", host=HOST, port=PORT, reload=True)
else:
    app = create_app()
