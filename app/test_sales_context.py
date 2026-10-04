from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import Reranker


query = "What was ArcelorMittal's sales in 2025?"

retriever = HybridRetriever()

results = retriever.search(
    query,
    top_k=20,
    candidate_k=50
)

reranker = Reranker()

reranked_results = reranker.rerank(
    query,
    results,
    top_k=5
)

print("\nTOP RERANKED CONTEXT")
print("=" * 100)

for i, result in enumerate(reranked_results, start=1):

    print(f"\n{'=' * 100}")
    print(f"RANK: {i}")
    print(f"PAGE: {result['page']}")
    print(f"RERANKER SCORE: {result['reranker_score']:.6f}")
    print(f"{'=' * 100}")

    print(result["text"])

retriever.vector_retriever.vector_store.close()
