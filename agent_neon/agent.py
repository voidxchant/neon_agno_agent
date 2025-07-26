import asyncio
import os

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.mcp import MCPTools
from dotenv import load_dotenv
load_dotenv()
async def neon_agent(prompt: str):
    """
    Connect to Neon’s MCP server, create tables, insert rows, or answer queries.
    """
    # Neon MCP server – we pass the API key via env, not the CLI arg
    cmd = f"npx -y @neondatabase/mcp-server-neon start {os.getenv('NEON_API_KEY')}"

    async with MCPTools(command=cmd) as mcp_tools:
        agent = Agent(
            model=Gemini(id="gemini-2.0-flash-exp"),  # or any Gemini model you like
            tools=[mcp_tools],
            instructions=(
                "You are a helpful assistant with access to Neon Postgres. "
                "Use the Neon MCP tools to create tables, insert data, or answer questions. here is the prohect id weathered-leaf-81957026"
            ),
            show_tool_calls=True,
            markdown=True,
        )

        await agent.aprint_response(prompt, stream=True)

if __name__ == "__main__":
    # Example prompt – change at will
    asyncio.run(
        neon_agent(
            "Create a table `users(id serial primary key, name text)` "
            "and insert one row with name='Ada'. Then confirm with a SELECT."
        )
    )