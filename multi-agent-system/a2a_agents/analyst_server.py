from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-5-mini", max_retries=2)

import uvicorn

from a2a.helpers import get_message_text, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)
from starlette.applications import Starlette


class AnalystExecutor(AgentExecutor):

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        research = get_message_text(context.message)

        print("\n[ANALYST] Research received via A2A...")
        print("[ANALYST] Generating analysis...")

        response = await llm.ainvoke([
            (
                "system",
                "You are a research analyst. "
                "Analyze the supplied findings, identify benefits, "
                "limitations, and trade-offs. "
                "Use only the supplied evidence. "
                "Do not invent facts or citations. "
                "Provide a concise, structured analysis."
            ),
            (
                "human",
                f"Research findings:\n{research}"
            )
        ])

        analysis_text = response.content

        await event_queue.enqueue_event(
            new_text_message(analysis_text)
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        raise Exception("Cancellation is not supported.")



agent_card = AgentCard(
    name="Research Analyst Agent",
    description="Analyzes research received from other AI agents.",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
    supported_interfaces=[
        AgentInterface(
            url="http://127.0.0.1:9999/",
            protocol_binding="JSONRPC",
            protocol_version="1.0",
        )
    ],
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    skills=[
        AgentSkill(
            id="analyze_research",
            name="Analyze Research",
            description="Analyze research findings and trade-offs.",
            tags=["research", "analysis"],
        )
    ],
)

handler = DefaultRequestHandler(
    agent_executor=AnalystExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=agent_card,
)

app = Starlette(
    routes=[
        *create_agent_card_routes(agent_card),
        *create_jsonrpc_routes(handler, "/"),
    ]
)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9999)
