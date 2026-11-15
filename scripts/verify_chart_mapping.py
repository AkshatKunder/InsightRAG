"""Verify which value belongs to which metric and year in a PDF bar chart.

pypdf's text extraction flattens the page 3 "Company overview" KPI chart of
the annual report into a run of numbers ("55.6 48.8 54.0 61.4 57.9 42.4 54.3
62.4 2025 2024 Crude steel production ...") with no indication of which value
belongs to which metric or year. This script reads the chart's actual drawing
instructions (the form XObject that holds it) and reconstructs the mapping
from geometry:

* every filled bar has a colour and a vertical extent;
* every value label sits next to the end of one bar;
* every legend label ("2025", "2024") sits next to a colour swatch;
* every category label is vertically aligned with a group of bars.

Usage (from the project root):

    python scripts/verify_chart_mapping.py
    python scripts/verify_chart_mapping.py --page 3 --json

It is used to justify the page 3 alternative evidence in benchmark_v2.json.
The geometry functions are pure and unit-tested; only ``extract_chart``
needs the PDF.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The geometry logic lives in app.rag.chart_extractor (also used to annotate chunks).
from app.rag.chart_extractor import (  # noqa: E402,F401
    Box,
    ChartValue,
    Label,
    _parse_to_unicode,
    extract_chart,
    map_chart,
)

DEFAULT_PDF = PROJECT_ROOT / "data" / "documents" / "annual_report.pdf"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--page", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Print the mapping as JSON")
    args = parser.parse_args(argv)

    boxes, labels, name = extract_chart(args.pdf, args.page)
    mapping = map_chart(boxes, labels)
    if args.json:
        print(json.dumps({"page": args.page, "xobject": name, "values": [asdict(v) for v in mapping]}, indent=2))
    else:
        print(f"Page {args.page}, chart {name}: {len(boxes)} filled boxes, {len(labels)} text labels")
        for value in mapping:
            print(f"  {value.category:<45} {value.series}  {value.value:>6}   (bar colour {value.colour})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
