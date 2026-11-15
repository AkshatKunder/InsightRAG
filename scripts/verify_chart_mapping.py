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
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = PROJECT_ROOT / "data" / "documents" / "annual_report.pdf"

NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
YEAR = re.compile(r"^(19|20)\d{2}$")


@dataclass(frozen=True)
class Box:
    """A filled rectangle (bar or legend swatch) in chart coordinates."""

    colour: tuple
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def y_mid(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass(frozen=True)
class Label:
    text: str
    x: float
    y: float


@dataclass(frozen=True)
class ChartValue:
    category: str
    series: str
    value: str
    colour: tuple


# ---------------------------------------------------------------------------
# Pure geometry (unit-tested without the PDF)
# ---------------------------------------------------------------------------


def _nearest(items, key):
    return min(items, key=key) if items else None


def map_chart(boxes: list[Box], labels: list[Label], max_gap: float = 12.0) -> list[ChartValue]:
    """Map every value label to (category, series) using bar geometry.

    * Legend swatches are small boxes with a year label just to their right;
      they define colour -> series.
    * A value label belongs to the bar whose vertical extent contains it.
    * A category is the run of non-numeric, non-year labels; a bar belongs to
      the category whose label block is vertically closest.
    Raises ValueError when the geometry is ambiguous.
    """

    years = [label for label in labels if YEAR.match(label.text)]
    values = [label for label in labels if NUMBER.match(label.text) and not YEAR.match(label.text)]
    words = [label for label in labels if label not in years and label not in values]

    # Legend: for each year label, the box whose right edge is closest to the
    # label's left side on the same line.
    series_by_colour: dict[tuple, str] = {}
    legend_boxes = set()
    for year in years:
        candidates = [
            box for box in boxes
            if box.y0 - 2 <= year.y <= box.y1 + 2 and 0 <= year.x - box.x1 <= max_gap
        ]
        swatch = _nearest(candidates, key=lambda box: year.x - box.x1)
        if swatch is None:
            raise ValueError(f"No legend swatch found for series label {year.text!r}")
        if swatch.colour in series_by_colour:
            raise ValueError(f"Colour {swatch.colour} is used by two legend entries")
        series_by_colour[swatch.colour] = year.text
        legend_boxes.add(swatch)

    bars = [box for box in boxes if box not in legend_boxes]

    # Category label blocks: group word labels whose lines are close together.
    blocks: list[list[Label]] = []
    for word in sorted(words, key=lambda label: -label.y):
        if blocks and blocks[-1][-1].y - word.y <= max_gap:
            blocks[-1].append(word)
        else:
            blocks.append([word])
    categories = [
        (" ".join(" ".join(label.text.split()) for label in block), sum(l.y for l in block) / len(block))
        for block in blocks
    ]

    mapped = []
    for value in values:
        containing = [bar for bar in bars if bar.y0 - 1 <= value.y <= bar.y1 + 1]
        bar = _nearest(containing, key=lambda box: abs(value.x - box.x1))
        if bar is None:
            raise ValueError(f"Value label {value.text!r} is not next to any bar")
        if bar.colour not in series_by_colour:
            raise ValueError(f"Bar colour {bar.colour} for {value.text!r} is not in the legend")
        category = _nearest(categories, key=lambda item: abs(item[1] - bar.y_mid))
        if category is None:
            raise ValueError("No category labels found")
        mapped.append(ChartValue(category[0], series_by_colour[bar.colour], value.text, bar.colour))
    return mapped


# ---------------------------------------------------------------------------
# PDF reading
# ---------------------------------------------------------------------------


def _multiply(a, b):
    return [
        a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5],
    ]


def _parse_to_unicode(data: bytes) -> dict[int, str]:
    """Minimal ToUnicode CMap parser (bfchar and bfrange with single targets)."""

    mapping: dict[int, str] = {}
    text = data.decode("latin-1")
    for block in re.findall(r"beginbfchar(.*?)endbfchar", text, re.S):
        for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            mapping[int(src, 16)] = bytes.fromhex(dst).decode("utf-16-be")
    for block in re.findall(r"beginbfrange(.*?)endbfrange", text, re.S):
        for start, end, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            base = int(dst, 16)
            for offset, code in enumerate(range(int(start, 16), int(end, 16) + 1)):
                mapping[code] = chr(base + offset)
    return mapping


def _decoder(font):
    font = font.get_object()
    if "/ToUnicode" in font and font.get("/Subtype") == "/Type0":
        table = _parse_to_unicode(font["/ToUnicode"].get_object().get_data())

        def decode(raw: bytes) -> str:
            return "".join(table.get(int.from_bytes(raw[i:i + 2], "big"), "?") for i in range(0, len(raw) - 1, 2))

        return decode
    return lambda raw: raw.decode("latin-1")


def _raw_bytes(operand) -> bytes:
    if isinstance(operand, bytes):
        return operand
    original = getattr(operand, "original_bytes", None)
    if original is not None:
        return original
    return str(operand).encode("latin-1", "replace")


def extract_chart(pdf_path: Path, page_number: int) -> tuple[list[Box], list[Label], str]:
    """Read boxes and text labels from the first form XObject on a page that contains text."""

    from pypdf import PdfReader
    from pypdf.generic import ContentStream

    reader = PdfReader(pdf_path)
    page = reader.pages[page_number - 1]
    xobjects = page.get("/Resources", {}).get("/XObject", {})
    for name, ref in xobjects.items():
        form = ref.get_object()
        if form.get("/Subtype") != "/Form":
            continue
        fonts = form.get("/Resources", {}).get("/Font", {})
        decoders = {font_name: _decoder(font) for font_name, font in fonts.items()}

        boxes: list[Box] = []
        labels: list[Label] = []
        ctm = [1, 0, 0, 1, 0, 0]
        stack = []
        fill = None
        path = []
        text_matrix = line_matrix = [1, 0, 0, 1, 0, 0]
        decode = lambda raw: raw.decode("latin-1")  # noqa: E731

        for operands, operator in ContentStream(form, reader).operations:
            op = operator.decode() if isinstance(operator, bytes) else operator
            if op == "q":
                stack.append((ctm[:], fill))
            elif op == "Q":
                ctm, fill = stack.pop()
            elif op == "cm":
                ctm = _multiply([float(v) for v in operands], ctm)
            elif op == "rg":
                fill = tuple(round(float(v), 3) for v in operands)
            elif op in ("m", "l"):
                path.append(_multiply([1, 0, 0, 1, float(operands[0]), float(operands[1])], ctm)[4:])
            elif op == "re":
                x, y, w, h = (float(v) for v in operands)
                path += [_multiply([1, 0, 0, 1, x, y], ctm)[4:], _multiply([1, 0, 0, 1, x + w, y + h], ctm)[4:]]
            elif op in ("f", "F", "f*"):
                if path and fill is not None:
                    xs, ys = [p[0] for p in path], [p[1] for p in path]
                    boxes.append(Box(fill, round(min(xs), 1), round(max(xs), 1), round(min(ys), 1), round(max(ys), 1)))
                path = []
            elif op in ("n", "S", "s", "B", "b", "B*", "b*"):
                path = []
            elif op == "Tf":
                decode = decoders.get(operands[0], decode)
            elif op == "BT":
                text_matrix = line_matrix = [1, 0, 0, 1, 0, 0]
            elif op == "Tm":
                text_matrix = line_matrix = [float(v) for v in operands]
            elif op in ("Td", "TD"):
                line_matrix = _multiply([1, 0, 0, 1, float(operands[0]), float(operands[1])], line_matrix)
                text_matrix = line_matrix
            elif op in ("Tj", "TJ"):
                parts = [operands[0]] if op == "Tj" else [p for p in operands[0] if not isinstance(p, (int, float)) and not hasattr(p, "as_numeric")]
                text = "".join(decode(_raw_bytes(p)) for p in parts).strip()
                if text:
                    position = _multiply(text_matrix, ctm)
                    labels.append(Label(text, round(position[4], 1), round(position[5], 1)))
        if labels:
            return boxes, labels, str(name)
    raise ValueError(f"No form XObject with text found on page {page_number}")


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
