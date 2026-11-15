import pytest

from app.evaluation.retrieval_metrics import (
    deduplicate_ranking,
    first_relevant_rank,
    group_recall_at_k,
    mean_reciprocal_rank,
    normalise_k_values,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    validate_k,
)


# ---------------------------------------------------------------- Recall@K


def test_recall_all_relevant_retrieved():
    assert recall_at_k([1, 2, 3], {1, 2}, 3) == 1.0


def test_recall_some_relevant_retrieved():
    assert recall_at_k([1, 9, 8], {1, 2}, 3) == 0.5


def test_recall_no_relevant_retrieved():
    assert recall_at_k([7, 8, 9], {1, 2}, 3) == 0.0


def test_recall_multiple_relevant_pages():
    assert recall_at_k([4, 1, 3, 9], {1, 2, 3, 4}, 4) == 0.75


def test_recall_duplicate_chunks_from_same_page_count_once():
    # Three chunks from page 1 are still one of two relevant pages.
    assert recall_at_k([1, 1, 1], {1, 2}, 3) == 0.5


def test_recall_depends_on_k():
    retrieved = [9, 1, 8, 2]
    assert recall_at_k(retrieved, {1, 2}, 1) == 0.0
    assert recall_at_k(retrieved, {1, 2}, 2) == 0.5
    assert recall_at_k(retrieved, {1, 2}, 4) == 1.0
    assert recall_at_k(retrieved, {1, 2}, 10) == 1.0


def test_recall_k_counts_distinct_pages_not_chunks():
    # Raw chunks [9, 9, 9, 1]: the relevant page is 2nd distinct unit, so it is in the top 2.
    assert recall_at_k([9, 9, 9, 1], {1}, 2) == 1.0
    assert recall_at_k([9, 9, 9, 1], {1}, 1) == 0.0


def test_recall_fewer_than_k_distinct_units():
    assert recall_at_k([1, 1, 1], {1, 2}, 10) == 0.5


def test_recall_and_precision_share_the_same_units():
    retrieved = [1, 1, 9, 2, 2, 8, 3]  # distinct: [1, 9, 2, 8, 3]
    relevant = {1, 2, 3}
    for k in range(1, 7):
        top = deduplicate_ranking(retrieved)[:k]
        hits = len(set(top) & relevant)
        assert recall_at_k(retrieved, relevant, k) == pytest.approx(hits / len(relevant))
        assert precision_at_k(retrieved, relevant, k) == pytest.approx(hits / len(top))


def test_recall_without_ground_truth_is_zero():
    assert recall_at_k([1, 2], set(), 2) == 0.0


def test_recall_ignores_unusable_results():
    assert recall_at_k([None, 1], {1}, 2) == 1.0
    assert recall_at_k([None, 1], {1}, 1) == 0.0


def test_group_recall_credits_any_page_of_a_fact():
    # Fact A is on page 95 or 172; fact B only on page 130.
    groups = [{95, 172}, {130}]
    assert group_recall_at_k([172, 5], groups, 2) == 0.5
    assert group_recall_at_k([172, 130], groups, 2) == 1.0
    assert group_recall_at_k([95, 172, 172], groups, 3) == 0.5


def test_group_recall_with_single_key_groups_matches_recall():
    retrieved = [3, 1, 3, 7]
    assert group_recall_at_k(retrieved, [{1}, {2}, {3}], 3) == recall_at_k(retrieved, {1, 2, 3}, 3)


# ---------------------------------------------------------------- Precision@K


def test_precision_all_relevant():
    assert precision_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0


def test_precision_some_relevant():
    assert precision_at_k([1, 9, 2, 8], {1, 2}, 4) == 0.5


def test_precision_none_relevant():
    assert precision_at_k([7, 8, 9], {1, 2}, 3) == 0.0


def test_precision_fewer_results_than_k_uses_returned_count():
    assert precision_at_k([1, 9], {1}, 10) == 0.5


def test_precision_empty_results_is_zero():
    assert precision_at_k([], {1}, 5) == 0.0


def test_precision_duplicate_chunks_count_as_one_unit():
    # Three chunks of one relevant page = one distinct unit, and it is relevant.
    assert precision_at_k([1, 1, 1], {1}, 3) == 1.0
    # Duplicates of a relevant page cannot raise the count above the distinct pages.
    assert precision_at_k([1, 1, 9], {1}, 3) == 0.5


def test_precision_duplicate_irrelevant_chunks_do_not_dilute():
    # Distinct ranking is [9, 1]; one of two units relevant.
    assert precision_at_k([9, 9, 9, 1], {1}, 3) == 0.5


def test_precision_fewer_than_k_distinct_units():
    # Five chunks but only two distinct pages: denominator is 2, not 5 or K.
    assert precision_at_k([1, 1, 2, 2, 1], {1, 2}, 10) == 1.0
    assert precision_at_k([1, 9, 1, 9], {1}, 5) == 0.5


def test_precision_multiple_pages_with_interleaved_duplicates():
    # Distinct ranking is [1, 9, 2, 8]; @2 -> 1/2, @3 -> 2/3, @4 -> 2/4.
    retrieved = [1, 1, 9, 1, 2, 9, 8]
    assert precision_at_k(retrieved, {1, 2}, 2) == 0.5
    assert precision_at_k(retrieved, {1, 2}, 3) == pytest.approx(2 / 3)
    assert precision_at_k(retrieved, {1, 2}, 4) == 0.5


def test_precision_counts_unusable_results_in_denominator():
    assert precision_at_k([None, 1], {1}, 2) == 0.5


# ---------------------------------------------------------------- MRR


def test_reciprocal_rank_first_position():
    assert reciprocal_rank([1, 9], {1}) == 1.0


def test_reciprocal_rank_second_position():
    assert reciprocal_rank([9, 1], {1}) == 0.5


def test_reciprocal_rank_third_position():
    assert reciprocal_rank([9, 8, 1], {1}) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_relevant():
    assert reciprocal_rank([9, 8, 7], {1}) == 0.0
    assert first_relevant_rank([9, 8, 7], {1}) is None


def test_reciprocal_rank_uses_first_of_multiple_relevant():
    assert reciprocal_rank([9, 2, 1, 3], {1, 2, 3}) == 0.5
    assert first_relevant_rank([9, 2, 1, 3], {1, 2, 3}) == 2


def test_reciprocal_rank_skips_unusable_results():
    assert reciprocal_rank([None, 1], {1}) == 0.5
    # Each unusable result is its own unit (they cannot be recognised as duplicates).
    assert reciprocal_rank([None, None, 1], {1}) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_distinct_unit_position():
    # Chunk rank 4, but the 2nd distinct page.
    assert reciprocal_rank([9, 9, 9, 1], {1}) == 0.5
    assert first_relevant_rank([9, 9, 9, 1], {1}) == 2


# ---------------------------------------------------------------- deduplication


def test_deduplicate_ranking_keeps_first_occurrence_order():
    assert deduplicate_ranking([3, 1, 3, 2, 1]) == [3, 1, 2]


def test_deduplicate_ranking_keeps_each_unusable_result():
    assert deduplicate_ranking([None, 1, None, 1]) == [None, 1, None]


def test_deduplicate_ranking_empty():
    assert deduplicate_ranking([]) == []


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank([1.0, 0.5, 0.0, 1 / 3]) == pytest.approx((1 + 0.5 + 0 + 1 / 3) / 4)


def test_mean_reciprocal_rank_empty_is_zero():
    assert mean_reciprocal_rank([]) == 0.0


# ---------------------------------------------------------------- K validation


@pytest.mark.parametrize("bad_k", [0, -1, 2.5, "3", None, True])
def test_invalid_k_rejected(bad_k):
    with pytest.raises(ValueError):
        validate_k(bad_k)
    with pytest.raises(ValueError):
        recall_at_k([1], {1}, bad_k)
    with pytest.raises(ValueError):
        precision_at_k([1], {1}, bad_k)


def test_normalise_k_values_sorts_and_deduplicates():
    assert normalise_k_values([10, 1, 5, 1]) == [1, 5, 10]


def test_normalise_k_values_rejects_empty():
    with pytest.raises(ValueError):
        normalise_k_values([])
