from app.rag.hybrid_retriever import HybridRetriever
from app.rag.generator import Generator


def main():

    print("=" * 70)
    print("InsightRAG - Local Document QA")
    print("=" * 70)
    print("Ask questions about the annual report.")
    print("Type 'exit' to quit.")

    retriever = HybridRetriever()
    generator = Generator()

    while True:

        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        print("\nSearching documents...")

        results = retriever.search(
            question,
            top_k=5
        )

        print("Generating answer...")

        answer = generator.generate(
            question,
            results
        )

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(answer)

        print("\n" + "=" * 70)
        print("SOURCES")
        print("=" * 70)

        for i, result in enumerate(results, start=1):
            print(
                f"{i}. {result['source']} — "
                f"Page {result['page']} — "
                f"Score: {result['score']:.4f}"
            )

    retriever.vector_retriever.vector_store.close()

    print("\nInsightRAG stopped.")


if __name__ == "__main__":
    main()
