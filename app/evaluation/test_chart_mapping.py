"""Geometry logic of scripts/verify_chart_mapping.py on a synthetic chart (no PDF)."""

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location("verify_chart_mapping", ROOT / "scripts" / "verify_chart_mapping.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(module)
    return module


TAN, BLUE = (0.9, 0.9, 0.8), (0.6, 0.7, 0.8)


def synthetic_chart(m, swap_legend=False):
    """Two categories with a 2025 (tan) and 2024 (blue) horizontal bar each."""

    boxes = [
        m.Box(TAN, 90, 200, 190, 205), m.Box(BLUE, 90, 210, 172, 187),   # category A
        m.Box(TAN, 90, 190, 110, 125), m.Box(BLUE, 90, 180, 92, 107),    # category B
        m.Box(BLUE if swap_legend else TAN, 89, 100, 16, 25),            # legend swatch 2025
        m.Box(TAN if swap_legend else BLUE, 136, 147, 16, 25),           # legend swatch 2024
    ]
    labels = [
        m.Label("10.0", 204, 195), m.Label("11.0", 214, 177),
        m.Label("20.0", 194, 115), m.Label("19.0", 184, 97),
        m.Label("2025", 103, 17), m.Label("2024", 150, 17),
        m.Label("Metric A", 30, 200), m.Label("(units)", 30, 190),
        m.Label("Metric B", 30, 118), m.Label("(units)", 30, 108),
    ]
    return boxes, labels


def as_dict(mapping):
    return {(v.category, v.series): v.value for v in mapping}


def test_maps_values_to_category_and_year():
    m = _module()
    mapping = as_dict(m.map_chart(*synthetic_chart(m)))

    assert mapping == {
        ("Metric A (units)", "2025"): "10.0",
        ("Metric A (units)", "2024"): "11.0",
        ("Metric B (units)", "2025"): "20.0",
        ("Metric B (units)", "2024"): "19.0",
    }


def test_series_comes_from_legend_colour_not_bar_order():
    m = _module()
    mapping = as_dict(m.map_chart(*synthetic_chart(m, swap_legend=True)))

    assert mapping[("Metric A (units)", "2025")] == "11.0"
    assert mapping[("Metric A (units)", "2024")] == "10.0"


def test_value_without_bar_is_rejected():
    m = _module()
    boxes, labels = synthetic_chart(m)
    labels.append(m.Label("99.9", 400, 400))

    with pytest.raises(ValueError, match="not next to any bar"):
        m.map_chart(boxes, labels)


def test_missing_legend_swatch_is_rejected():
    m = _module()
    boxes, labels = synthetic_chart(m)
    boxes = [box for box in boxes if not (box.y0 == 16 and box.x0 == 136)]

    with pytest.raises(ValueError, match="legend swatch"):
        m.map_chart(boxes, labels)


def test_to_unicode_parser_handles_ranges_and_chars():
    m = _module()
    cmap = b"""
    2 beginbfrange
    <0013> <001C> <0030>
    <0011> <0011> <002E>
    endbfrange
    1 beginbfchar
    <0003> <0020>
    endbfchar
    """
    table = m._parse_to_unicode(cmap)

    assert "".join(table[c] for c in (0x18, 0x17, 0x11, 0x13)) == "54.0"
    assert table[0x03] == " "
