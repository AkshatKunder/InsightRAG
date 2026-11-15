"""Pure retrieval metrics: Recall@K, Precision@K and (Mean) Reciprocal Rank.

Evidence unit and K
-------------------
Every function takes the retriever's ranked output as a list of *evidence
keys*, one per retrieved result in retrieval order. A key is any hashable
value identifying an evidence unit (the evaluator uses ``(source, page)``).
A result whose metadata is unusable is represented by ``None``.

Before scoring, the list is collapsed into a ranking of **distinct evidence
units** (:func:`deduplicate_ranking`): each key is kept at the position of its
first occurrence, later duplicates are dropped, and the original order is
otherwise preserved. Every ``None`` is kept as its own non-relevant unit,
because unusable results cannot be identified as duplicates.

**K counts positions in that distinct-unit ranking**, not raw chunks. So
"top K" means the first K distinct pages, and all metrics share the same unit:

* Recall@K    = |relevant units in top K| / |relevant units|
* Precision@K = |relevant units in top K| / min(K, number of distinct units)
* RR          = 1 / position of the first relevant unit (0 if none)

Duplicate chunks therefore never inflate or deflate a score, and duplicates of
an irrelevant page cannot push relevant pages out of the top K.

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


def deduplicate_ranking(retrieved: Sequence[Hashable | None]) -> list[Hashable | None]:
    """Collapse a ranked key list into distinct evidence units, keeping first-occurrence order.

    ``None`` entries (unusable metadata) are each kept as a separate unit.
    """

    seen: set[Hashable] = set()
    ranking: list[Hashable | None] = []
    for key in retrieved:
        if key is None:
            ranking.append(None)
        elif key not in seen:
            seen.add(key)
            ranking.append(key)
    return ranking


def _relevant_in_top_k(retrieved: Sequence[Hashable | None], relevant: set, k: int) -> set[Hashable]:
    return {key for key in deduplicate_ranking(retrieved)[:k] if key is not None and key in relevant}


def recall_at_k(retrieved: Sequence[Hashable | None], relevant: Iterable[Hashable], k: int) -> float:
    """Share of relevant units among the top ``k`` distinct units.

    Returns 0.0 when there is no ground truth (callers should not average such
    questions; the evaluator excludes unanswerable questions for this reason).
    """

    validate_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(_relevant_in_top_k(retrieved, relevant_set, k)) / len(relevant_set)


def group_recall_at_k(
    retrieved: Sequence[Hashable | None],
    groups: Sequence[Iterable[Hashable]],
    k: int,
) -> float:
    """Share of evidence groups with at least one member among the top ``k`` distinct units.

    Each group is one required piece of evidence that any of several units can
    satisfy (for example a fact printed on two pages). With one-unit groups this
    equals :func:`recall_at_k`.
    """

    validate_k(k)
    group_sets = [set(group) for group in groups]
    if not group_sets:
        return 0.0
    found = _relevant_in_top_k(retrieved, set().union(*group_sets), k)
    return sum(1 for group in group_sets if group & found) / len(group_sets)


def precision_at_k(retrieved: Sequence[Hashable | None], relevant: Iterable[Hashable], k: int) -> float:
    """Share of the top ``k`` distinct units that are relevant.

    The denominator is min(k, number of distinct units), so a ranking with fewer
    than ``k`` distinct units is not penalised for its length. Returns 0.0 when
    nothing was retrieved.
    """

    validate_k(k)
    denominator = min(k, len(deduplicate_ranking(retrieved)))
    if denominator == 0:
        return 0.0
    return len(_relevant_in_top_k(retrieved, set(relevant), k)) / denominator


def first_relevant_rank(retrieved: Sequence[Hashable | None], relevant: Iterable[Hashable]) -> int | None:
    """1-based position of the first relevant unit in the distinct-unit ranking, or None."""

    relevant_set = set(relevant)
    for rank, key in enumerate(deduplicate_ranking(retrieved), start=1):
        if key is not None and key in relevant_set:
            return rank
    return None


def reciprocal_rank(retrieved: Sequence[Hashable | None], relevant: Iterable[Hashable]) -> float:
    """1 / position of the first relevant unit; 0.0 if no unit is relevant."""

    rank = first_relevant_rank(retrieved, relevant)
    return 0.0 if rank is None else 1.0 / rank


def mean_reciprocal_rank(reciprocal_ranks: Iterable[float]) -> float:
    """Mean of per-question reciprocal ranks; 0.0 when given no questions."""

    values = list(reciprocal_ranks)
    if not values:
        return 0.0
    return sum(values) / len(values)
