from mcp.server.fastmcp import FastMCP
from document_retriever import search_documents

mcp = FastMCP("Research Paper Server")


@mcp.tool()
def search_papers(query: str) -> str:
    """Search local PDF research papers for relevant evidence."""

    results = search_documents(query, top_k=2)

    if not results:
        return "No relevant passages found."

    passages = []

    for result in results:
        passages.append(
            f"Source: {result['source']}\n"
            f"Page: {result['page']}\n"
            f"Passage: {result['text'][:1000]}"
        )

    return "\n\n---\n\n".join(passages)

if __name__ == "__main__":
    mcp.run(transport="stdio")
