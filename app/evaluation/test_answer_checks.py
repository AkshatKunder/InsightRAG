import pytest

from app.evaluation.answer_checks import (
    check_calculation,
    check_comparison,
    check_fact,
    date_present,
    detect_direction,
    parse_quantities,
    ungrounded_numbers,
    year_for_quantity,
)

SALES = {"metric": "total_sales", "value": 61352, "unit": "USD million", "period": "2025"}
Q_SALES = "What were ArcelorMittal's total sales in 2025?"


# ---------------------------------------------------------------- parsing


def test_parse_scales_units_and_years():
    qs = parse_quantities("Sales were $61.4 billion in 2025, 55.6 million tonnes, 54.0 Mt, 125,554 FTE and 1.7%.")
    by_raw = {q["raw"]: q for q in qs}
    assert by_raw["$61.4 billion"]["base_value"] == pytest.approx(61.4e9)
    assert by_raw["$61.4 billion"]["unit_class"] == "currency"
    assert by_raw["55.6 million tonnes"]["base_value"] == pytest.approx(55.6e6)
    assert by_raw["54.0 Mt"]["base_value"] == pytest.approx(54.0e6)
    assert by_raw["125,554 FTE"]["unit_class"] == "count"
    assert by_raw["1.7%"]["unit_class"] == "percent"
    assert not any(q["raw"] == "2025" for q in qs)


# ---------------------------------------------------------------- numeric facts


@pytest.mark.parametrize("answer", [
    "Total sales were $61,352 million in 2025.",
    "ArcelorMittal had sales of $61.4 billion in 2025.",
    "Sales were 61,352 million USD.",
    "In 2025, total sales amounted to $61,352 million.",
])
def test_sales_matched(answer):
    assert check_fact(SALES, answer, Q_SALES)["matched"]


@pytest.mark.parametrize("answer,reason", [
    ("Sales were 61,352 in 2025.", "unit"),                      # bare number, no currency
    ("Total sales were $62,441 million.", "not stated"),         # 2024 value
    ("Europe sales were $28,793 million.", "not stated"),        # segment value
    ("In 2024 sales were $61,352 million.", "period"),           # wrong year paired
    ("Sales were about $60 billion.", "not stated"),             # not rounding-consistent
    ("Sales were $6 billion.", "not stated"),
])
def test_sales_not_matched(answer, reason):
    result = check_fact(SALES, answer, Q_SALES)
    assert not result["matched"]
    assert reason in result["reason"]


def test_one_significant_figure_is_not_enough():
    fact = {"metric": "net_income_attributable_to_parent", "value": 3152, "unit": "USD million", "period": "2025"}
    assert check_fact(fact, "Net income was $3.2 billion in 2025.")["matched"]
    assert not check_fact(fact, "Net income was $3 billion in 2025.")["matched"]
    # The non-controlling-interest figure is a different metric value.
    assert not check_fact(fact, "Net income was $3,243 million in 2025.")["matched"]


def test_rounded_flag():
    assert check_fact(SALES, "Sales were $61.4 billion in 2025.", Q_SALES)["rounded"] is True
    assert check_fact(SALES, "Sales were $61,352 million in 2025.", Q_SALES)["rounded"] is False


def test_period_from_question_when_answer_has_no_year():
    assert check_fact(SALES, "Total sales were $61,352 million.", Q_SALES)["matched"]
    assert not check_fact(SALES, "Total sales were $61,352 million.", "What were total sales?")["matched"]


def test_segment_entity_required():
    fact = {"metric": "segment_sales", "segment": "Europe", "value": 28793, "unit": "USD million", "period": "2025"}
    assert check_fact(fact, "Europe had sales of $28,793 million in 2025.")["matched"]
    result = check_fact(fact, "The segment had sales of $28,793 million in 2025.")
    assert not result["matched"] and result["entity_ok"] is False


def test_comparative_values_paired_with_years():
    f25 = {"metric": "iron_ore_production", "value": 48.8, "unit": "million tonnes", "period": "2025"}
    f24 = {"metric": "iron_ore_production", "value": 42.4, "unit": "million tonnes", "period": "2024"}
    good = "Production increased from 42.4 million tonnes in 2024 to 48.8 Mt in 2025."
    swapped = "Production was 42.4 million tonnes in 2025 and 48.8 million tonnes in 2024."
    assert check_fact(f25, good)["matched"] and check_fact(f24, good)["matched"]
    assert not check_fact(f25, swapped)["matched"] and not check_fact(f24, swapped)["matched"]


def test_year_for_quantity_prefers_following_year():
    text = "In 2025, sales fell to $61,352 million from $62,441 million in 2024."
    first, second = [q for q in parse_quantities(text)]
    assert year_for_quantity(text, first["start"], first["end"]) == "2025"
    assert year_for_quantity(text, second["start"], second["end"]) == "2024"


def test_employee_count_unit():
    fact = {"metric": "employees_fte", "value": 125554, "unit": "FTE employees", "period": "2025-12-31"}
    assert check_fact(fact, "Approximately 125,554 employees as of December 31, 2025.")["matched"]
    assert not check_fact(fact, "Approximately $125,554 as of December 31, 2025.")["matched"]


# ---------------------------------------------------------------- non-numeric facts


def test_names_lists_dates_and_counts():
    assert check_fact({"metric": "chief_executive_officer", "value": "Aditya Mittal"}, "The CEO is Aditya Mittal.")["matched"]
    assert not check_fact({"metric": "chief_executive_officer", "value": "Aditya Mittal"}, "Lakshmi N. Mittal.")["matched"]
    companies = {"metric": "predecessor_companies", "value": ["Mittal Steel Company N.V.", "Arcelor"]}
    assert check_fact(companies, "Mittal Steel and Arcelor merged in 2007.")["matched"]
    assert not check_fact(companies, "ArcelorMittal was formed from Mittal Steel.")["matched"]
    assert check_fact({"metric": "merger_year", "value": 2007}, "They merged in 2007.")["matched"]
    count = {"metric": "reportable_segment_count", "value": 6, "unit": "segments"}
    assert check_fact(count, "It has six reportable segments.")["matched"]
    assert check_fact(count, "organized in six operating and reportable segments")["matched"]
    # List numbering is not a stated count (observed false positive in run 1 of Q002).
    assert not check_fact(count, "Segments are: 1. North America 2. Brazil 3. Europe 4. India 5. Mining 6. Others.")["matched"]
    assert check_fact({"metric": "primary_business", "value": "integrated steel and mining company"},
                      "A leading integrated steel and mining company.")["matched"]
    assert date_present("2001-06-08", "incorporated on June 8, 2001") and date_present("2001-06-08", "on 8 June 2001")
    assert not date_present("2001-06-08", "in June 2001")


# ---------------------------------------------------------------- comparisons and calculations


@pytest.mark.parametrize("text,expected", [
    ("Sales decreased by 1.7%.", "decrease"),
    ("Operating income rose to $3,628 million.", "increase"),
    ("Shipments remained stable.", "unchanged"),
    ("Sales increased in Europe but declined overall.", "mixed"),
    ("Sales were $61,352 million.", None),
])
def test_detect_direction(text, expected):
    assert detect_direction(text) == expected


def test_check_comparison():
    comp = {"direction": "decrease", "absolute_change": -1089, "unit": "USD million", "percent_change": -1.7}
    result = check_comparison(comp, "Sales decreased by $1,089 million (1.7%) to $61,352 million in 2025.")
    assert result == {"expected_direction": "decrease", "detected_direction": "decrease", "direction_ok": True,
                      "absolute_change_ok": True, "percent_change_ok": True}
    wrong = check_comparison(comp, "Sales increased by $1,089 million.")
    assert wrong["direction_ok"] is False


def test_check_calculation_tolerance_and_direction():
    per_employee = {"result": 488650, "unit": "USD per FTE employee", "tolerance_percent": 1.0}
    assert check_calculation(per_employee, "About $488,650 per employee.")["result_ok"]
    assert check_calculation(per_employee, "Roughly $0.49 million per employee.")["result_ok"]
    assert not check_calculation(per_employee, "About $500,000 per employee.")["result_ok"]
    diff = {"result": 1.6, "unit": "million tonnes", "tolerance_percent": 0.0}
    question = "Was production higher or lower than shipments, and by how much?"
    ok = check_calculation(diff, "Production was higher by 1.6 million tonnes.", question)
    assert ok["result_ok"] and ok["direction_ok"]
    assert check_calculation(diff, "Production was lower by 1.6 million tonnes.", question)["direction_ok"] is False


# ---------------------------------------------------------------- grounding


def test_ungrounded_numbers_use_table_scales_and_allow_derived():
    context = ["(in $ millions) Total 61,352 62,441 (1,089)"]
    assert ungrounded_numbers("Sales were $61.4 billion, down from $62,441 million.", context) == []
    assert ungrounded_numbers("Sales were $70 billion.", context) == ["$70 billion"]
    assert ungrounded_numbers("A difference of 1.6 million tonnes.", ["55.6 54.0"], allowed=[1.6e6]) == []
