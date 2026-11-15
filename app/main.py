from app.rag.pipeline import build_pipeline


def main():

    print("=" * 70)
    print("InsightRAG - Hybrid RAG Assistant")
    print("=" * 70)

    # Initialize components. The shared pipeline (app/rag/pipeline.py) runs
    # hybrid retrieval (multi-hop aware) -> reranking -> scope-aware generation
    # -> evidence selection -> scope-aware citation validation.
    pipeline = build_pipeline()

    print("\nSystem ready.")
    print("Ask questions about the annual report.")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        result = pipeline.answer(question)
        answer = result["answer"]
        citations = result["citations"]
        validation = result["validation"]

        # --------------------------------
        # Display Answer
        # --------------------------------

        print("\nAnswer:")
        print(answer)

        # --------------------------------
        # Citation Validation
        # --------------------------------

        print("\nCitation Validation:")
        print("Status:", validation["overall_status"])
        print("Reason:", validation["reason"])

        if validation["overall_status"] == "ABSTAINED":
            print("\n" + "-" * 70 + "\n")
            continue

        print("\nClaim Validation:")
        if validation["claims"]:
            for claim in validation["claims"]:
                location = ""
                if claim["source"]:
                    location = f" ({claim['source']} — Page {claim['page']})"
                print(f"- [{claim['status']}] {claim['claim_text']}{location}")
                print(f"  {claim['explanation']}")
        else:
            print("No factual claims were extracted from the answer.")

        print("\nSelected Evidence:")

        if citations:
            for citation in citations:
                print(f"- {citation['source']} — Page {citation['page']}")
                print(f"  Evidence: {citation['text'][:250]}...")
        else:
            print("No relevant evidence candidates were selected.")

        print("\n" + "-" * 70 + "\n")

    # Clean up Qdrant
    pipeline.retriever.vector_retriever.vector_store.close()


if __name__ == "__main__":
    main()
