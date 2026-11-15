"""Pure retrieval metrics: Recall@K, Precision@K and (Mean) Reciprocal Rank.

All functions take a ranked list of *evidence keys* (one per retrieved result,
in retrieval order) and the ground truth. A key is any hashable value that
identifies an evidence unit; the evaluator uses ``(source, page)`` tuples.
A retrieved result whose metadata is unusable is represented by ``None``: it
still occupies its rank but can never be relevant.

Duplicates are handled so that they never inflate a score: several results
with the same key count as one evidence unit. The first occurrence is the one
that can be relevant; later occurrences take up a rank but add nothing.

Nothing here loads models, touches Qdrant or depends on the benchmark format.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence


def validate_k(k) -> int:
    """Return ``k`` if it is a positive int (bools rejected), else raise ValueError."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"K must be a positive integer, got {k!r}")
    return k


def normalise_k_values(k_values: Iterable) -> list[int]:
    """Validate a collection of K values and return them sorted and de-duplicated."""

    values = sorted({validate_k(k) for k in k_values})
    if not values:
        raise ValueError("At least one K value is required")
    return values


def _distinct_in_top_k(retrieved: Sequence[Hashable | None], k: int) -> set[Hashable]:
    return {key for key in retrieved[:k] if key is not None}


def recall_at_k(retrieved: Sequence[Hashable | None], relevant: Iterable[Hashable], k: int) -> float:
    """Share of distinct relevant keys found in the top ``k`` results.

    Recall@K = |distinct relevant keys in top K| / |relevant keys|.
    Returns 0.0 when there is no ground truth (callers should not average
    such questions; the evaluator excludes unanswerable questions for this reason).
    """

    validate_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(_distinct_in_top_k(retrieved, k) & relevant_set) / len(relevant_set)


def group_recall_at_k(
    retrieved: Sequence[Hashable | None],
    groups: Sequence[Iterable[Hashable]],
    k: int,
) -> float:
    """Share of evidence groups with at least one member in the top ``k`` results.

    Each group is one required evidence unit that can be satisfied by any of
    several keys (for example a fact printed on two different pages). With
    single-key groups this equals :func:`recall_at_k`.
    """

    validate_k(k)
    group_sets = [set(group) for group in groups]
    if not group_sets:
        return 0.0
    found = _distinct_in_top_k(retrieved, k)
    covered = sum(1 for group in group_sets if group & found)
    return covered / len(group_sets)


def precision_at_k(retrieved: Sequence[Hashable | None], relevant: Iterable[Hashable], k: int) -> float:
    """Share of the top ``k`` result slots that bring a new relevant evidence unit.

    Precision@K = |distinct relevant keys in top K| / min(K, number of results).
    Duplicate results for an already-counted key still use up a slot, so they
    lower precision instead of inflating it. Returns 0.0 for an empty result list.
    """

    validate_k(k)
    denominator = min(k, len(retrieved))
    if denominator == 0:
        return 0.0
    hits = _distinct_in_top_k(retrieved, k) & set(relevant)
    return len(hits) / denominator


def first_relevant_rank(retrieved: Sequence[Hashable | None], relevant: Iterable[Hashable]) -> int | None:
    """1-based rank of the first relevant result, or None if there is none."""

    relevant_set = set(relevant)
    for rank, key in enumerate(retrieved, start=1):
        if key is not None and key in relevant_set:
            return rank
    return None


def reciprocal_rank(retrieved: Sequence[Hashable | None], relevant: Iterable[Hashable]) -> float:
    """1 / rank of the first relevant result; 0.0 if no result is relevant."""

    rank = first_relevant_rank(retrieved, relevant)
    return 0.0 if rank is None else 1.0 / rank


def mean_reciprocal_rank(reciprocal_ranks: Iterable[float]) -> float:
    """Mean of per-question reciprocal ranks; 0.0 when given no questions."""

    values = list(reciprocal_ranks)
    if not values:
        return 0.0
    return sum(values) / len(values)
