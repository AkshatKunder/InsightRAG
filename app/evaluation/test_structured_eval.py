from app.evaluation.structured_eval import db_fingerprint, derived_correct, expected_period_change, expected_shares, result_match


def test_result_match_unordered_with_extra_columns_and_rounding():
    gold = [["Europe", 10.0], ["Asia", 5]]
    assert result_match(gold, [["Asia", 5.001, "x"], ["Europe", 10, "y"]])
    assert not result_match(gold, [["Asia", 6], ["Europe", 10]])
    assert not result_match(gold, [["Europe", 10]])  # missing row


def test_result_match_keeps_row_pairs_together():
    gold = [["A", 1], ["B", 2]]
    assert not result_match(gold, [["A", 2], ["B", 1]])  # same column values, wrong pairing


def test_result_match_ordered():
    gold = [["2025-01", 1], ["2025-02", 2]]
    assert result_match(gold, [["2025-01", 1], ["2025-02", 2]], ordered=True)
    assert not result_match(gold, [["2025-02", 2], ["2025-01", 1]], ordered=True)


def test_single_value_results_ignore_column_names():
    assert result_match([[350.5]], [[350.5]])


def test_expected_derived_figures_are_computed_from_gold_rows():
    assert expected_period_change([[2025, 28793], [2024, 29952]]) == {"absolute_change": -1159, "percent_change": -3.87}
    assert expected_shares([["A", 30], ["B", 10]]) == {"a": 75.0, "b": 25.0}
    analytics = {"period_changes": [{"absolute_change": -1159, "percent_change": -3.87}]}
    assert derived_correct("period_change", [[2024, 29952], [2025, 28793]], analytics)
    assert not derived_correct("period_change", [[2024, 29952], [2025, 28793]], {})
    assert derived_correct("shares", [["A", 30], ["B", 10]], {"shares": [{"label": "A", "share_percent": 75.0},
                                                                         {"label": "B", "share_percent": 25.0}]})


def test_db_fingerprint_detects_changes(tmp_path):
    import sqlite3

    path = tmp_path / "f.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (a)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    before = db_fingerprint(path)
    conn.execute("DELETE FROM t")
    conn.commit()
    conn.close()
    assert db_fingerprint(path) != before
