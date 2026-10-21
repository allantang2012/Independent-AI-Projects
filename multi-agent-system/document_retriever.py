from pathlib import Path

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCUMENT_DIR = Path(__file__).parent / "documents"


def load_documents():
    documents = []

    for path in sorted(DOCUMENT_DIR.glob("*.pdf")):
        reader = PdfReader(path)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text and text.strip():
                documents.append({
                    "source": path.name,
                    "page": page_number,
                    "text": text.strip()
                })

    return documents


def search_documents(query, top_k=3):
    documents = load_documents()

    if not documents:
        return []

    texts = [doc["text"] for doc in documents]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        matrix
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:
        if similarities[index] <= 0:
            continue

        document = documents[index].copy()
        document["score"] = float(
            similarities[index]
        )

        results.append(document)

    return results


if __name__ == "__main__":
    results = search_documents(
        "How can AI agents use tools reliably?"
    )

    for result in results:
        print(f"\nSource: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Score: {result['score']:.3f}")
        print(result["text"][:500])
