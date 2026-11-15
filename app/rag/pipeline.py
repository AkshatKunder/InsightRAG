"""End-to-end question answering, shared by ``app/main.py`` and the evaluation scripts.

Phase 2 behaviour (``PipelineConfig.legacy()``): hybrid retrieval (top 20 from
50 candidates per retriever) -> cross-encoder rerank to 5 chunks -> generate ->
select 3 citations -> validate.

Phase 3 additions, each switchable for ablation and backward compatibility:

* ``scope_context`` (Task 1/2): retrieved chunks carry scope/section/chart
  metadata (``app.rag.document_context``) and the generator prompt shows it.
* ``multihop`` (Task 4): multi-fact questions are split into sub-queries; each
  is retrieved and reranked separately and the rankings are interleaved into
  the same 5-chunk context.
* ``question_aware_validation`` (Task 3): the validator sees the question, so
  segment figures are accepted only when the question names the segment.
* ``abstain_on_contradiction`` (Task 5): if every claim fails with a verified
  contradiction (scope/metric mismatch, arithmetic error, unsupported value),
  the answer is replaced by the standard abstention and nothing is cited.

Phase 3.1 additions (``PipelineConfig.phase3()`` turns them off):

* ``query_rewrite``: year-over-year change questions get an extra query for
  the consolidated "<Y2> <Y1> change" table (group-level chunks first);
  "primary business" questions get a company-overview query
  (``app.rag.query_rewriter``).
* ``calculate``: differences and changes are computed from evidence values
  (``app.rag.calculation``) instead of trusting LLM arithmetic.
* ``completeness_check``: list and definition answers are checked against
  the document's own enumeration/definition (``app.rag.completeness``);
  omissions trigger the single revision and are reported in
  ``validation["completeness"]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.rag.citation_selector import select_citations
from app.rag.citation_validator import extract_numbers, validate_citation
from app.rag.calculation import apply as apply_calculation
from app.rag.calculation import calculate
from app.rag.completeness import check as completeness_check
from app.rag.completeness import feedback as completeness_feedback
from app.rag.multihop import decompose, merge_rankings
from app.rag.query_rewriter import consolidated_first, rewrite_queries
from app.rag.scope_validation import is_contradiction

ABSTENTION = "I could not find this information in the provided document."
DEFAULT_PDF = "data/documents/annual_report.pdf"


@dataclass
class PipelineConfig:
    retrieve_top_k: int = 20
    candidate_k: int = 50
    context_size: int = 5
    max_citations: int = 3
    scope_context: bool = True
    multihop: bool = True
    question_aware_validation: bool = True
    revise_on_contradiction: bool = True
    abstain_on_contradiction: bool = True
    # Phase 3.1
    query_rewrite: bool = True
    calculate: bool = True
    completeness_check: bool = True

    @classmethod
    def legacy(cls) -> "PipelineConfig":
        return cls(scope_context=False, multihop=False, question_aware_validation=False,
                   revise_on_contradiction=False, abstain_on_contradiction=False,
                   query_rewrite=False, calculate=False, completeness_check=False)

    @classmethod
    def phase3(cls) -> "PipelineConfig":
        """The Phase 3 configuration (before the Phase 3.1 additions)."""
        return cls(query_rewrite=False, calculate=False, completeness_check=False)


def build_scope_index(retriever, pdf_path: str = DEFAULT_PDF):
    """Scope metadata for the retriever's chunks (pages re-read from the same PDF)."""

    from app.rag.chart_extractor import chart_values_by_page
    from app.rag.document_context import ScopeIndex
    from app.rag.loader import load_pdf

    try:
        charts = chart_values_by_page(pdf_path)
    except Exception:
        charts = {}
    return ScopeIndex(retriever.chunks, load_pdf(pdf_path), charts)


class RAGPipeline:
    def __init__(self, retriever, reranker, generator, config: PipelineConfig | None = None, scope_index=None):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.config = config or PipelineConfig()
        self.scope_index = scope_index
        if hasattr(self.generator, "scope_aware"):
            self.generator.scope_aware = self.config.scope_context

    def settings(self) -> dict:
        generator = self.generator.settings() if hasattr(self.generator, "settings") else {}
        return {"pipeline": asdict(self.config), "generation": generator}

    def retrieve(self, question: str) -> tuple[list[dict], list[str]]:
        cfg = self.config
        sub_queries = decompose(question) if cfg.multihop else [question]
        rewrites = []
        if cfg.query_rewrite:
            names = self.scope_index.segment_names if self.scope_index is not None else []
            rewrites = rewrite_queries(question, names)
        enrich = self.scope_index.enrich if cfg.scope_context and self.scope_index is not None else (lambda items: [dict(i) for i in items])
        rankings = []
        for query in sub_queries:
            candidates = self.retriever.search(query, top_k=cfg.retrieve_top_k, candidate_k=cfg.candidate_k)
            rankings.append(enrich(self.reranker.rerank(query, candidates, top_k=cfg.context_size)))
        for rewrite in rewrites:
            candidates = self.retriever.search(rewrite["query"], top_k=cfg.retrieve_top_k, candidate_k=cfg.candidate_k)
            ranking = enrich(self.reranker.rerank(rewrite["query"], candidates, top_k=cfg.context_size))
            rankings.append(consolidated_first(ranking) if rewrite["consolidated_first"] else ranking)
        queries = sub_queries + [r["query"] for r in rewrites]
        if len(rankings) == 1:
            context = [dict(item) for item in rankings[0]]
        else:
            context = merge_rankings(rankings, cfg.context_size, queries)
        return context, queries

    def answer(self, question: str) -> dict:
        cfg = self.config
        context, sub_queries = self.retrieve(question)
        answer = self.generator.generate(question, context)
        citations = self._citations(question, answer, context, sub_queries)
        validation = validate_citation(answer, citations, question=question if cfg.question_aware_validation else None)
        revision = None
        problems = _contradictions(validation)
        completeness = completeness_check(question, answer, context) if cfg.completeness_check else None
        if completeness and completeness["status"] == "incomplete":
            problems = problems + [completeness_feedback(completeness)]
        if cfg.revise_on_contradiction and problems:
            # One evidence-based revision: the validator's / completeness reasons are fed
            # back to the generator, which must use matching evidence or abstain.
            feedback = (
                f"{question}\n\n(A previous answer was rejected because: " + " ".join(problems[:3])
                + " Answer again from the excerpts: include every item or term the document gives, use only "
                "figures that match the question's metric, entity, year and group-level scope, with correct "
                "arithmetic, or state that the information could not be found.)"
            )
            revised = self.generator.generate(feedback, context)
            revised_citations = self._citations(question, revised, context, sub_queries)
            revised_validation = validate_citation(revised, revised_citations, question=question if cfg.question_aware_validation else None)
            revision = {"original_answer": answer, "original_validation": validation, "problems": problems}
            answer, citations, validation = revised, revised_citations, revised_validation
        calculation = None
        if cfg.calculate:
            # Differences and changes are computed from evidence values, never taken from the LLM.
            calculation = calculate(question, citations + [c for c in context if c not in citations])
            if calculation:
                answer, citations = apply_calculation(answer, validation, citations, calculation)
                validation = validate_citation(answer, citations, question=question if cfg.question_aware_validation else None)
        rejected = None
        if cfg.abstain_on_contradiction and _all_claims_contradicted(validation):
            rejected = {"answer": answer, "validation": validation}
            answer, citations = ABSTENTION, []
            validation = validate_citation(answer, citations)
            validation["reason"] = "The generated figures contradicted the cited evidence (scope, metric or arithmetic); the system abstained."
        if cfg.completeness_check and validation.get("overall_status") != "ABSTAINED":
            # Reported alongside the claim statuses (which are unchanged): omissions are visible
            # even when every stated number is supported.
            validation["completeness"] = completeness_check(question, answer, context)
        return {
            "calculation": _calc_summary(calculation),
            "answer": answer,
            "context": context,
            "citations": citations,
            "validation": validation,
            "sub_queries": sub_queries,
            "revision": revision,
            "rejected": rejected,
        }


    def _citations(self, question: str, answer: str, context: list[dict], sub_queries: list[str]) -> list[dict]:
        """Evidence selection; for multi-hop questions the best evidence of every sub-query is kept."""

        limit = self.config.max_citations
        if len(sub_queries) < 2:
            return select_citations(question=question, answer=answer, retrieved_chunks=context, max_sources=limit)
        selected, seen = [], set()
        answer_numbers = extract_numbers(answer)
        per_query = []
        for query in sub_queries:
            chunks = [c for c in context if c.get("sub_query") == query]
            ranking = select_citations(question=query, answer=answer, retrieved_chunks=chunks, max_sources=limit)
            if not ranking:
                # Below the selector's relevance threshold: fall back to the sub-query's
                # top-ranked chunk, but only if it contains a number the answer states.
                ranking = [c for c in chunks if extract_numbers(c.get("text", "")) & answer_numbers][:1]
            per_query.append(ranking)
        overall = select_citations(question=question, answer=answer, retrieved_chunks=context, max_sources=limit)
        # One citation per sub-query first, then the best remaining overall evidence.
        for candidate in [ranking[0] for ranking in per_query if ranking] + overall:
            key = (candidate.get("source"), candidate.get("page"))
            if key not in seen:
                seen.add(key)
                selected.append(candidate)
        return selected[:max(limit, len(sub_queries))]


def _calc_summary(calculation: dict | None) -> dict | None:
    if not calculation:
        return None
    return {
        "kind": calculation["kind"], "sentence": calculation["sentence"], "difference": calculation["difference"],
        "percent": calculation.get("percent"), "unit": calculation["unit"],
        "operands": [{k: o[k] for k in ("raw", "unit", "label", "page")} for o in calculation["operands"]],
    }


def _contradictions(validation: dict) -> list[str]:
    return [
        claim["explanation"] for claim in validation.get("claims") or []
        if claim.get("status") == "UNSUPPORTED" and is_contradiction(claim.get("explanation"))
    ]


def _all_claims_contradicted(validation: dict) -> bool:
    claims = validation.get("claims") or []
    return bool(claims) and all(
        claim.get("status") == "UNSUPPORTED" and is_contradiction(claim.get("explanation")) for claim in claims
    )


def build_pipeline(pdf_path: str = DEFAULT_PDF, config: PipelineConfig | None = None, generation_options: dict | None = None) -> RAGPipeline:
    """Production pipeline with the project's retrievers and models."""

    from app.rag.generator import Generator
    from app.rag.hybrid_retriever import HybridRetriever
    from app.rag.reranker import Reranker

    config = config or PipelineConfig()
    retriever = HybridRetriever(pdf_path=pdf_path)
    scope_index = build_scope_index(retriever, pdf_path) if config.scope_context else None
    return RAGPipeline(retriever, Reranker(), Generator(options=generation_options), config, scope_index)
