"""Small HTML and SVG helpers used by the generated report."""

import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

def esc(value):
    return html.escape(str(value))


def svg_bar(path, labels, values, title, x_label="", color="#2563eb", width=980, height=520):
    labels = list(labels)
    values = np.asarray(values, dtype=float)
    margin = {"left": 310, "right": 45, "top": 72, "bottom": 65}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    row_h = plot_h / max(len(labels), 1)
    lo = min(0.0, float(values.min(initial=0)))
    hi = max(0.0, float(values.max(initial=1)))
    if math.isclose(lo, hi):
        hi = lo + 1

    def x_scale(value):
        return margin["left"] + (value - lo) / (hi - lo) * plot_w

    zero = x_scale(0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{esc(title)}</text>',
        f'<line x1="{zero:.1f}" y1="{margin["top"]}" x2="{zero:.1f}" y2="{height-margin["bottom"]}" stroke="#94a3b8"/>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        y = margin["top"] + index * row_h + row_h * 0.18
        bar_h = row_h * 0.64
        x_value = x_scale(value)
        x = min(zero, x_value)
        bar_w = max(abs(x_value - zero), 1)
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>'
        )
        parts.append(
            f'<text x="{margin["left"]-10}" y="{y+bar_h*0.72:.1f}" text-anchor="end" font-family="Arial" font-size="13">{esc(label)}</text>'
        )
        anchor = "start" if value >= 0 else "end"
        dx = 7 if value >= 0 else -7
        parts.append(
            f'<text x="{x_value+dx:.1f}" y="{y+bar_h*0.72:.1f}" text-anchor="{anchor}" font-family="Arial" font-size="12">{value:.3g}</text>'
        )
    parts.append(
        f'<text x="{margin["left"]+plot_w/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="14">{esc(x_label)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_line(path, series, title, y_label, width=1040, height=460):
    margin = {"left": 78, "right": 30, "top": 65, "bottom": 60}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    all_dates = pd.concat([frame[["date"]] for _, frame, _ in series])["date"]
    all_values = np.concatenate([frame["value"].to_numpy(float) for _, frame, _ in series])
    x_min, x_max = all_dates.min(), all_dates.max()
    y_min = min(0.0, float(np.nanmin(all_values)))
    y_max = float(np.nanmax(all_values)) * 1.05

    def xs(date):
        return margin["left"] + (date - x_min).days / max((x_max - x_min).days, 1) * plot_w

    def ys(value):
        return margin["top"] + (y_max - value) / max(y_max - y_min, 1e-9) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{esc(title)}</text>',
    ]
    for tick in np.linspace(y_min, y_max, 6):
        y = ys(tick)
        parts.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width-margin["right"]}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{margin["left"]-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.0f}</text>')
    for year in range(x_min.year, x_max.year + 1):
        date = pd.Timestamp(year=year, month=1, day=1)
        if x_min <= date <= x_max:
            x = xs(date)
            parts.append(f'<text x="{x:.1f}" y="{height-28}" text-anchor="middle" font-family="Arial" font-size="12">{year}</text>')
    for name, frame, color in series:
        points = " ".join(f"{xs(row.date):.1f},{ys(row.value):.1f}" for row in frame.itertuples())
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" opacity="0.9"/>')
    legend_x = margin["left"]
    for index, (name, _, color) in enumerate(series):
        x = legend_x + index * 190
        parts.append(f'<line x1="{x}" y1="50" x2="{x+25}" y2="50" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text x="{x+32}" y="55" font-family="Arial" font-size="13">{esc(name)}</text>')
    parts.append(f'<text transform="translate(20 {margin["top"]+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">{esc(y_label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def html_table(frame, columns=None, formats=None, max_rows=20):
    data = frame.copy()
    if columns:
        data = data[columns]
    data = data.head(max_rows)
    formats = formats or {}
    rows = ["<table><thead><tr>" + "".join(f"<th>{esc(c)}</th>" for c in data.columns) + "</tr></thead><tbody>"]
    for _, row in data.iterrows():
        cells = []
        for column, value in row.items():
            if pd.isna(value):
                rendered = ""
            elif column in formats:
                rendered = formats[column](value)
            else:
                rendered = str(value)
            cells.append(f"<td>{esc(rendered)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)
