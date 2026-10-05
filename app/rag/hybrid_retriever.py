from app.rag.loader import load_pdf
from app.rag.chunker import chunk_pages
from app.rag.bm25_retriever import BM25Retriever
from app.rag.retriever import Retriever
from app.rag.query_expander import expand_query


class HybridRetriever:

    def __init__(self, pdf_path="data/documents/annual_report.pdf"):

        pages = load_pdf(pdf_path)

        self.chunks = chunk_pages(pages)

        self.bm25 = BM25Retriever(self.chunks)

        self.vector_retriever = Retriever()

    def search(self, query, top_k=5, candidate_k=50):

        expanded_query = expand_query(query)

        # Retrieve more candidates from both systems
        bm25_results = self.bm25.search(
            expanded_query,
            top_k=candidate_k
        )

        vector_results = self.vector_retriever.search(
            expanded_query,
            top_k=candidate_k
        )

        combined = {}

        # -----------------------------------
        # BM25
        # -----------------------------------

        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            key = (
                result["source"],
                result["page"],
                result["text"]
            )

            if key not in combined:

                combined[key] = {
                    "text": result["text"],
                    "source": result["source"],
                    "page": result["page"],

                    "bm25_score": None,
                    "bm25_rank": None,

                    "vector_score": None,
                    "vector_rank": None,

                    "rrf_score": 0.0
                }

            combined[key]["bm25_score"] = result["score"]

            combined[key]["bm25_rank"] = rank

            combined[key]["rrf_score"] += (
                0.4 / (60 + rank)
            )

        # -----------------------------------
        # Vector Search
        # -----------------------------------

        for rank, result in enumerate(
            vector_results,
            start=1
        ):

            key = (
                result["source"],
                result["page"],
                result["text"]
            )

            if key not in combined:

                combined[key] = {
                    "text": result["text"],
                    "source": result["source"],
                    "page": result["page"],

                    "bm25_score": None,
                    "bm25_rank": None,

                    "vector_score": None,
                    "vector_rank": None,

                    "rrf_score": 0.0
                }

            combined[key]["vector_score"] = result["score"]

            combined[key]["vector_rank"] = rank

            combined[key]["rrf_score"] += (
                0.6 / (60 + rank)
            )

        # -----------------------------------
        # Sort
        # -----------------------------------

        ranked_results = sorted(
            combined.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        final_results = []
        seen_pages = set()

        for result in ranked_results:
            page_key = (
                result["source"],
                result["page"]
            )

            if page_key in seen_pages:
                continue

            final_results.append(result)
            seen_pages.add(page_key)

            if len(final_results) >= top_k:
                break

        # Main score = hybrid score
        for result in final_results:

            result["score"] = result["rrf_score"]

        return final_results
