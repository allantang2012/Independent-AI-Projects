import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_path = Path(__file__).parent / "mcp_server.py"

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover the server's tools
            tools = await session.list_tools()

            print("\nAvailable MCP tools:")
            for tool in tools.tools:
                print(f"- {tool.name}")

            # Call the PDF search tool via MCP
            result = await session.call_tool(
                "search_papers",
                arguments={
                    "query": "AutoGen multi-agent communication"
                }
            )

            print("\nMCP search results:")
            for content in result.content:
                if content.type == "text":
                    print(content.text)


if __name__ == "__main__":
    asyncio.run(main())
