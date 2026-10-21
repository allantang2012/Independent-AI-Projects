import asyncio
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from a2a.client import ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest

load_dotenv()


async def main():
    print("[RESEARCHER] Connecting to MCP server...")

    server_path = Path(__file__).parent / "mcp_server.py"

    mcp_client = MultiServerMCPClient({
        "research_papers": {
            "command": sys.executable,
            "args": [str(server_path)],
            "transport": "stdio",
        }
    })

    tools = await mcp_client.get_tools()

    search_tool = next(
        tool for tool in tools
        if tool.name == "search_papers"
    )

    print("[RESEARCHER] Searching PDF through MCP...")

    evidence = await search_tool.ainvoke({
        "query": "AutoGen multi-agent communication and collaboration"
    })

    llm = ChatOpenAI(
        model="gpt-5-mini",
        max_retries=2
    )

    print("[RESEARCHER] Summarizing evidence...")

    summary = await llm.ainvoke([
        (
            "system",
            "Summarize the retrieved research evidence. "
            "Preserve PDF filenames and page references. "
            "Do not invent facts or citations. "
            "Clearly indicate when evidence is insufficient."
        ),
        (
            "human",
            f"Retrieved PDF evidence:\n{evidence}"
        )
    ])

    research_text = summary.content

    print("\n--- RESEARCH FINDINGS ---")
    print(research_text)

    print("\n[A2A] Sending findings to Analyst...")

    async with httpx.AsyncClient(timeout=180.0) as http_client:
        client = await create_client(
            agent="http://127.0.0.1:9999",
            client_config=ClientConfig(
                streaming=False,
                httpx_client=http_client,
            ),
        )

        try:
            request = SendMessageRequest(
                message=new_text_message(
                    research_text,
                    role=Role.ROLE_USER
                )
            )

            async for response in client.send_message(request):
                print("\n--- A2A ANALYST RESPONSE ---")
                print(response)
        finally:
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())
