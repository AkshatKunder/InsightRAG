"""Retrieval evaluation for InsightRAG (Phase 2).

``retrieval_metrics`` holds pure metric functions; ``retrieval_evaluator``
matches retrieved chunks to benchmark ground truth and aggregates the metrics.
Neither module loads models, Qdrant or Ollama.
"""
