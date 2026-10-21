
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


from document_retriever import search_documents


@tool
def search_research_papers(query: str) -> str:
    """Search local PDF research papers for passages
    relevant to a research question. Returns source
    filenames, page numbers, and retrieved text.
    """
    print(f"[PDF SEARCH] Searching for: {query}")

    results = search_documents(query, top_k=2)

    if not results:
        return "No relevant passages found."

    passages = []

    for result in results:
        passages.append(
            f"Source: {result['source']}\n"
            f"Page: {result['page']}\n"
            f"Similarity: {result['score']:.3f}\n"
            f"Passage: {result['text'][:1000]}"
        )

    return "\n\n---\n\n".join(passages)
