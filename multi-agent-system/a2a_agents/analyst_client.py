import asyncio
import httpx

from a2a.client import (
    ClientConfig,
    create_client,
)
from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest


async def main():
    http_client = httpx.AsyncClient(
       timeout=httpx.Timeout(180.0)
    )

    client = await create_client(
       agent="http://127.0.0.1:9999",
       client_config=ClientConfig(
           streaming=False,
           httpx_client=http_client,
       ),
    )

    print("[CLIENT] Connected to Analyst Agent")

    message = new_text_message(
        "Research findings: AI agents can automate tasks, "
        "but may make incorrect decisions and incur API costs. "
        "Please analyze these trade-offs.",
        role=Role.ROLE_USER,
    )

    request = SendMessageRequest(message=message)

    print("[CLIENT] Sending research to Analyst...")

    try:
        async for response in client.send_message(request):
            print("\n--- ANALYST RESPONSE ---")
            print(response)
    finally:
       await client.close()
       await http_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
