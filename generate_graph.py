#!/usr/bin/env python3
"""
Generate agent graph visualization for current e-commerce data analysis system
"""

import asyncio
from agents import agent

async def generate_graph():
    """Generate and save the agent graph as PNG"""
    try:
        # Compile the graph first, then get visualization
        compiled_agent = agent.compile()
        graph_image = compiled_agent.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)

        # Save to images folder
        with open("images/agent_graph.png", "wb") as f:
            f.write(graph_image)

        print("✅ Successfully generated new agent_graph.png")

    except Exception as e:
        print(f"❌ Error generating graph: {e}")

if __name__ == "__main__":
    asyncio.run(generate_graph())