from app.rag.retriever import Retriever
from app.rag.generator import Generator


question = "How many employees did ArcelorMittal have in 2025?"


print("\nSearching documents...")

retriever = Retriever()

results = retriever.search(
    question,
    top_k=5
)


print(f"Retrieved {len(results)} relevant chunks.")


print("\nGenerating answer...")

generator = Generator()

answer = generator.generate(
    question,
    results
)


print("\n" + "=" * 70)
print("RAG ANSWER")
print("=" * 70)

print(answer)

print("\n" + "=" * 70)
print("RETRIEVED SOURCES")
print("=" * 70)

for i, result in enumerate(results, start=1):
    print(
        f"{i}. {result['source']} — "
        f"Page {result['page']} — "
        f"Score: {result['score']:.4f}"
    )

retriever.vector_store.close()
