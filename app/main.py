from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import Reranker
from app.rag.generator import Generator
from app.rag.citation_selector import select_citations


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

        print("\nDEBUG — Pages passed to citation selector:")

        for result in reranked_results:
            print(
                f"Page {result['page']} | "
                f"Final score: {result.get('final_score', 'N/A')}"
            )

        citations = select_citations(
            question=question,
            answer=answer,
            retrieved_chunks=reranked_results,
            max_sources=3,
        )

        print("\nSelected Evidence:")

        if citations:
            for citation in citations:
                print(f"- {citation['source']} — Page {citation['page']}")
                print(f"  Evidence: {citation['text'][:250]}...")
        else:
            print("No relevant evidence candidates were selected.")

        print("\n" + "-" * 70 + "\n")

    # Clean up Qdrant
    retriever.vector_retriever.vector_store.close()


if __name__ == "__main__":
    main()
