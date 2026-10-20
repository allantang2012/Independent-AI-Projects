
from typing import TypedDict
from typing import Literal
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

# Initialize our language model
llm = ChatOpenAI(
    model="gpt-5-mini",
    max_retries=2
)


# Shared state passed between agents
class ResearchState(TypedDict):
    question: str
    evidence: str
    research: str
    analysis: str
    final_report: str
    next_agent: str


# Agent 1: Researcher
def researcher(state: ResearchState):
    print("\n[RESEARCHER] Organizing evidence...")

    response = llm.invoke(
        [
            (
                "system",
                "You are a research assistant. "
                "Organize only the supplied evidence into "
                "clear findings. Do not invent facts."
            ),
            (
                "human",
                f"Question: {state['question']}\n\n"
                f"Evidence:\n{state['evidence']}"
            )
        ]
    )

    return {"research": response.content}


# Agent 2: Analyst
def analyst(state: ResearchState):
    print("\n[ANALYST] Evaluating research...")

    response = llm.invoke(
        [
            (
                "system",
                "You are an analytical assistant. "
                "Analyze the research findings, identify "
                "trade-offs and limitations, and provide "
                "a concise conclusion. Do not invent evidence."
            ),
            (
                "human",
                f"Question: {state['question']}\n\n"
                f"Research findings:\n{state['research']}"
            )
        ]
    )

    return {"analysis": response.content}


# Agent 3: Writer
def writer(state: ResearchState):
    print("\n[WRITER] Preparing final report...")

    response = llm.invoke([
        (
            "system",
            "You are a professional research writer. "
            "Create a clear, structured report using only "
            "the research findings and analysis provided. "
            "Include an introduction, key findings, "
            "trade-offs, and conclusion. "
            "Do not invent facts or sources."
        ),
        (
            "human",
            f"Question: {state['question']}\n\n"
            f"Research:\n{state['research']}\n\n"
            f"Analysis:\n{state['analysis']}"
        )
    ])

    return {"final_report": response.content}


class RoutingDecision(BaseModel):
    next_agent: Literal["researcher", "analyst"] = Field(
        description="The next agent that should execute"
    )

coordinator_llm = llm.with_structured_output(RoutingDecision)

def coordinator(state: ResearchState):
    print("\n[COORDINATOR] Deciding next agent...")

    decision = coordinator_llm.invoke([
        (
            "system",
            "You coordinate a research workflow. "
            "Choose researcher if research findings are missing. "
            "Choose analyst if research findings already exist. "
            "Do not evaluate or rewrite the findings."
        ),
        (
            "human",
            f"Question: {state['question']}\n"
            f"Existing research: {state['research'] or 'None'}"
        )
    ])

    print(f"[COORDINATOR] Selected: {decision.next_agent}")
    return {"next_agent": decision.next_agent}


def route_from_coordinator(state: ResearchState):
    return state["next_agent"]


# Build the LangGraph workflow

graph = StateGraph(ResearchState)

graph.add_node("coordinator", coordinator)
graph.add_node("researcher", researcher)
graph.add_node("analyst", analyst)
graph.add_node("writer", writer)

graph.add_edge(START, "coordinator")

graph.add_conditional_edges(
    "coordinator",
    route_from_coordinator,
    {
        "researcher": "researcher",
        "analyst": "analyst"
    }
)

graph.add_edge("researcher", "analyst")
graph.add_edge("analyst", "writer")
graph.add_edge("writer", END)

app = graph.compile()


# Run the workflow
result = app.invoke({
    "question": "What are the benefits and limitations of AI agents?",
    "evidence": (
        "AI agents can automate multi-step tasks and use tools. "
        "They can reduce repetitive manual work. "
        "However, agents may make incorrect decisions, "
        "incur API costs, and require monitoring."
    ),
    "research": (
    "AI agents automate multi-step tasks and reduce manual work. "
    "However, they can make mistakes, incur API costs, "
    "and require human oversight."
    ),
    "analysis": "",
    "final_report": "",
    "next_agent": ""
})

print("\n--- RESEARCH FINDINGS ---")
print(result["research"])

print("\n--- FINAL ANALYSIS ---")
print(result["analysis"])

print("\n--- FINAL RESEARCH REPORT ---")
print(result["final_report"])
