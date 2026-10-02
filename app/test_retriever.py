from app.rag.retriever import Retriever


retriever = Retriever()

query = "How many employees did ArcelorMittal have in 2025?"

results = retriever.search(query, top_k=5)

print("\n" + "=" * 70)
print("RETRIEVAL RESULTS")
print("=" * 70)

for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")
    print(f"Score: {result['score']:.4f}")
    print(f"Source: {result['source']}")
    print(f"Page: {result['page']}")
    print(f"Text length: {len(result['text'])}")

    print("-" * 70)
    print(result["text"][:1000])