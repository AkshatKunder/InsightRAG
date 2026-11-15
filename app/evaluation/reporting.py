"""Markdown and SVG helpers for evaluation reports (no third-party dependencies).

Charts are static SVG files referenced from the Markdown reports. Every chart
is accompanied by a table with the same numbers in the report (the table view),
bars carry a ``<title>`` tooltip, and series use a validated categorical
palette in fixed order (blue, orange, aqua: light-surface CVD check passes;
the aqua slot is below 3:1 contrast, so the adjacent table is the relief).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path

SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#1f1f1e"
TEXT_SECONDARY = "#5f5e58"
GRID = "#e4e3de"


def fmt(value, digits: int = 3) -> str:
    """Format a metric for tables: floats to ``digits`` places, None as an em dash."""

    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def md_table(headers: Sequence[str], rows: Sequence[Sequence], align: Sequence[str] | None = None) -> str:
    """GitHub-flavoured Markdown table. ``align`` items: 'l' or 'r' (default: first l, rest r)."""

    align = align or ["l"] + ["r"] * (len(headers) - 1)
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(str(h) for h in headers) + " |", "| " + " | ".join(sep) + " |"]
    for row in rows:
        cells = [fmt(cell) if not isinstance(cell, str) else cell for cell in row]
        lines.append("| " + " | ".join(cell.replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
    return "\n".join(lines)


def svg_grouped_bars(
    title: str,
    groups: Sequence[str],
    series: Mapping[str, Sequence[float]],
    y_max: float = 1.0,
    y_label: str = "",
    width: int = 680,
    height: int = 320,
) -> str:
    """Vertical grouped bar chart as an SVG string (values in [0, y_max])."""

    if len(series) > len(SERIES_COLOURS):
        raise ValueError(f"At most {len(SERIES_COLOURS)} series are supported")
    names = list(series)
    for name in names:
        if len(series[name]) != len(groups):
            raise ValueError(f"Series {name!r} has {len(series[name])} values for {len(groups)} groups")

    left, right, top, bottom = 52, 16, 56, 44
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / max(len(groups), 1)
    gap = 2
    bar_w = min(28.0, (group_w * 0.7 - gap * (len(names) - 1)) / max(len(names), 1))
    cluster_w = bar_w * len(names) + gap * (len(names) - 1)

    def y(value: float) -> float:
        return top + plot_h * (1 - max(0.0, min(value, y_max)) / y_max)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{escape(title)}" font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        f'<text x="{left}" y="22" font-size="14" font-weight="600" fill="{TEXT_PRIMARY}">{escape(title)}</text>',
    ]
    # Legend (always present for >= 2 series), one row under the title.
    lx = left
    for index, name in enumerate(names):
        out.append(f'<rect x="{lx}" y="32" width="10" height="10" rx="2" fill="{SERIES_COLOURS[index]}"/>')
        out.append(f'<text x="{lx + 14}" y="41" font-size="11" fill="{TEXT_SECONDARY}">{escape(name)}</text>')
        lx += 14 + 7 * len(name) + 18
    # Recessive grid and axis labels.
    for step in range(5):
        value = y_max * step / 4
        gy = y(value)
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{gy:.1f}" y2="{gy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{left - 6}" y="{gy + 4:.1f}" font-size="10" text-anchor="end" fill="{TEXT_SECONDARY}">{value:.2f}</text>')
    if y_label:
        out.append(f'<text x="12" y="{top + plot_h / 2:.1f}" font-size="10" fill="{TEXT_SECONDARY}" '
                   f'transform="rotate(-90 12 {top + plot_h / 2:.1f})" text-anchor="middle">{escape(y_label)}</text>')
    baseline = y(0)
    for g, group in enumerate(groups):
        x0 = left + g * group_w + (group_w - cluster_w) / 2
        for s, name in enumerate(names):
            value = float(series[name][g])
            bx = x0 + s * (bar_w + gap)
            by = y(value)
            h = max(baseline - by, 0.0)
            radius = min(4.0, h / 2, bar_w / 2)
            # Rounded data end at the top, square at the baseline.
            path = (
                f"M{bx:.1f},{baseline:.1f} V{by + radius:.1f} Q{bx:.1f},{by:.1f} {bx + radius:.1f},{by:.1f} "
                f"H{bx + bar_w - radius:.1f} Q{bx + bar_w:.1f},{by:.1f} {bx + bar_w:.1f},{by + radius:.1f} "
                f"V{baseline:.1f} Z"
            )
            out.append(f'<path d="{path}" fill="{SERIES_COLOURS[s]}"><title>{escape(group)} · {escape(name)}: {value:.3f}</title></path>')
        out.append(f'<text x="{left + g * group_w + group_w / 2:.1f}" y="{baseline + 18:.1f}" font-size="11" '
                   f'text-anchor="middle" fill="{TEXT_SECONDARY}">{escape(group)}</text>')
    out.append(f'<line x1="{left}" x2="{width - right}" y1="{baseline:.1f}" y2="{baseline:.1f}" stroke="{TEXT_SECONDARY}" stroke-width="1"/>')
    out.append("</svg>")
    return "\n".join(out)


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return path
