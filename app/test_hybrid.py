from app.rag.hybrid_retriever import HybridRetriever


retriever = HybridRetriever()

query = "What was ArcelorMittal's sales in 2025?"

results = retriever.search(
    query,
    top_k=10,
    candidate_k=20
)

print("\nHYBRID RETRIEVAL RESULTS")
print("=" * 80)

for i, result in enumerate(results, start=1):

    print(f"\nRank {i}")
    print(f"Page: {result['page']}")
    print(f"RRF Score: {result['rrf_score']:.6f}")

    print(
        f"BM25 Rank: {result['bm25_rank']} | "
        f"BM25 Score: {result['bm25_score']}"
    )

    print(
        f"Vector Rank: {result['vector_rank']} | "
        f"Vector Score: {result['vector_score']}"
    )

    print(f"Text: {result['text'][:300]}...")

retriever.vector_retriever.vector_store.close()
