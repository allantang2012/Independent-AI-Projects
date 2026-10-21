import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


async def main():
    server_path = Path(__file__).parent / "mcp_server.py"

    client = MultiServerMCPClient({
        "research_papers": {
            "command": sys.executable,
            "args": [str(server_path)],
            "transport": "stdio",
        }
    })

    tools = await client.get_tools()

    print("Available tools:", [tool.name for tool in tools])

    llm = ChatOpenAI(model="gpt-5-mini", max_retries=2)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a research assistant. "
            "Use the search_papers tool to find evidence "
            "before answering. "
            "Include source filenames and page numbers. "
            "Do not invent citations."
        )
    )

    result = await agent.ainvoke({
        "messages": [
            (
                "user",
                "Use search_papers to find how AutoGen "
                "supports multi-agent communication. "
                "Summarize the evidence with citations."
            )
        ]
    })

    print("\n--- RESEARCH RESULT ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
