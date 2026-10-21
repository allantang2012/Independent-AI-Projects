from langchain.agents import create_agent
from research_tools import search_knowledge_base

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


# Create a Researcher Agent with tool access
research_agent = create_agent(
    model=llm,
    tools=[search_knowledge_base],
    system_prompt=(
        "You are a research assistant with access to a "
        "knowledge-base search tool. "
        "Search the knowledge base before answering. "
        "Relevant topics are benefits, limitations, "
        "and reliability. "
        "Use retrieved information as your evidence. "
        "Clearly distinguish retrieved facts from "
        "your own interpretations. "
        "Do not invent sources."
    )
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
    print("\n[RESEARCHER] Retrieving information...")

    result = research_agent.invoke({
        "messages": [
            (
                "user",
                f"Research this question: {state['question']}\n"
                "Use the knowledge base to find relevant "
                "information about benefits, limitations, "
                "and reliability. Summarize the findings."
            )
        ]
    })

    # Extract the research agent's final answer
    research_text = result["messages"][-1].content

    return {"research": research_text}


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

    # Deterministic routing based on shared state
    existing_research = state.get("research", "").strip()

    if existing_research:
        next_agent = "analyst"
    else:
        next_agent = "researcher"

    print(f"[COORDINATOR] Selected: {next_agent}")

    return {"next_agent": next_agent}


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

if __name__ == "__main__":
    result = app.invoke({
        "question": (
            "What are the benefits, limitations, "
            "and reliability concerns of AI agents?"
        ),
        "evidence": "",
        "research": "",
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
