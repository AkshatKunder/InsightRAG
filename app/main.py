from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import Reranker
from app.rag.generator import Generator


def main():

    print("=" * 70)
    print("InsightRAG - Hybrid RAG Assistant")
    print("=" * 70)

    # Initialize components
    retriever = HybridRetriever()
    reranker = Reranker()
    generator = Generator()

    print("\nSystem ready.")
    print("Ask questions about the annual report.")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        # --------------------------------
        # Step 1: Hybrid Retrieval
        # --------------------------------

        results = retriever.search(
            question,
            top_k=20,
            candidate_k=50
        )

        # --------------------------------
        # Step 2: Reranking
        # --------------------------------

        reranked_results = reranker.rerank(
            question,
            results,
            top_k=5
        )

        # --------------------------------
        # Step 3: Generate Answer
        # --------------------------------

        answer = generator.generate(
            question,
            reranked_results
        )

        # --------------------------------
        # Display Answer
        # --------------------------------

        print("\nAnswer:")
        print(answer)

        print("\nSources:")

        for i, result in enumerate(reranked_results, start=1):

            print(
                f"{i}. {result['source']} "
                f"— Page {result['page']} "
                f"— Reranker Score: "
                f"{result['reranker_score']:.4f}"
            )

        print("\n" + "-" * 70 + "\n")

    # Clean up Qdrant
    retriever.vector_retriever.vector_store.close()


if __name__ == "__main__":
    main()
