from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import Reranker


query = "What was ArcelorMittal's sales in 2025?"


# ---------------------------------
# Hybrid Retrieval
# ---------------------------------

retriever = HybridRetriever()

results = retriever.search(
    query,
    top_k=20,
    candidate_k=50
)


print("\nBEFORE RERANKING")
print("=" * 80)

for i, result in enumerate(results, start=1):

    print(
        f"Rank {i} | "
        f"Page {result['page']} | "
        f"RRF: {result['rrf_score']:.6f}"
    )


# ---------------------------------
# Reranking
# ---------------------------------

reranker = Reranker()

reranked_results = reranker.rerank(
    query,
    results,
    top_k=5
)


print("\nAFTER RERANKING")
print("=" * 80)

for i, result in enumerate(reranked_results, start=1):

    print(
        f"Rank {i} | "
        f"Page {result['page']} | "
        f"Reranker Score: {result['reranker_score']:.6f}"
    )

    print(
        f"Text: {result['text'][:300]}..."
    )


retriever.vector_retriever.vector_store.close()
