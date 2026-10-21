
from langchain_core.tools import tool

KNOWLEDGE_BASE = {
    "benefits": (
        "AI agents can automate multi-step tasks, "
        "use external tools, and reduce repetitive work."
    ),
    "limitations": (
        "AI agents can make incorrect decisions, "
        "incur API costs, and require monitoring."
    ),
    "reliability": (
        "Agent reliability can be evaluated using "
        "task success rates, tool-call accuracy, "
        "and error frequency."
    )
}


@tool
def search_knowledge_base(topic: str) -> str:
    """Look up information about AI agents by topic:
    benefits, limitations, or reliability.
    """
    print(f"[SEARCH TOOL] Searching for: {topic}")

    return KNOWLEDGE_BASE.get(
        topic.lower().strip(),
        "No information found for this topic."
    )
