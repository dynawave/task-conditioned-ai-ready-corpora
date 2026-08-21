#!/usr/bin/env python3
"""Render editable SCI-style SVG and vector PDF RQ2 budget figures."""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import pandas as pd
import reportlab
from reportlab.pdfgen import canvas


TASKS = ("T1a_ACRONYM", "T2_NUMERIC", "T3_CLAIM")
TASK_LABELS = {"T1a_ACRONYM": "T1a", "T2_NUMERIC": "T2", "T3_CLAIM": "T3"}
STYLES = {
    "task_specific": {"color": "#1f4e79", "label": "Task-specific", "marker": "circle", "dash": None},
    "common": {"color": "#b24a35", "label": "Common", "marker": "square", "dash": "5,3"},
    "random": {"color": "#666666", "label": "Random", "marker": "triangle", "dash": "2,3"},
}
WIDTH, HEIGHT = 720.0, 245.0
LEFT, RIGHT, TOP, BOTTOM, GAP = 48.0, 13.0, 42.0, 35.0, 20.0
PANEL_W = (WIDTH - LEFT - RIGHT - 2 * GAP) / 3
PLOT_H = HEIGHT - TOP - BOTTOM


def marker_svg(kind: str, x: float, y: float, color: str) -> str:
    if kind == "circle":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.6" fill="white" stroke="{color}" stroke-width="1.25"/>'
    if kind == "square":
        return f'<rect x="{x-2.5:.2f}" y="{y-2.5:.2f}" width="5" height="5" fill="white" stroke="{color}" stroke-width="1.25"/>'
    points = f"{x:.2f},{y-3:.2f} {x-2.9:.2f},{y+2.4:.2f} {x+2.9:.2f},{y+2.4:.2f}"
    return f'<polygon points="{points}" fill="white" stroke="{color}" stroke-width="1.25"/>'


def marker_pdf(pdf: canvas.Canvas, kind: str, x: float, y: float) -> None:
    if kind == "circle":
        pdf.circle(x, y, 2.6, stroke=1, fill=0)
    elif kind == "square":
        pdf.rect(x - 2.5, y - 2.5, 5, 5, stroke=1, fill=0)
    else:
        path = pdf.beginPath()
        path.moveTo(x, y + 3)
        path.lineTo(x - 2.9, y - 2.4)
        path.lineTo(x + 2.9, y - 2.4)
        path.close()
        pdf.drawPath(path, stroke=1, fill=0)


def render(curves: pd.DataFrame, budget_type: str, output_svg: Path, output_pdf: Path) -> None:
    subset = curves[curves.budget_type.eq(budget_type)]
    ymax = max(0.6, math.ceil(float(subset.capture_fraction.max()) * 10) / 10)
    yticks = [index * ymax / 4 for index in range(5)]
    xmap = lambda value, panel_left: panel_left + (value - 5.0) / 45.0 * PANEL_W
    ymap_svg = lambda value: TOP + PLOT_H - value / ymax * PLOT_H
    ymap_pdf = lambda value: HEIGHT - (TOP + PLOT_H - value / ymax * PLOT_H)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#222">',
    ]
    pdf = canvas.Canvas(str(output_pdf), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    pdf.setTitle(f"RQ2 {budget_type} budget-capture curves")
    pdf.setAuthor("AI corpus project")

    legend_x = 222.0
    for index, strategy in enumerate(("task_specific", "common", "random")):
        style = STYLES[strategy]
        x = legend_x + index * 112
        dash = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
        svg.append(f'<line x1="{x:.1f}" y1="15" x2="{x+20:.1f}" y2="15" stroke="{style["color"]}" stroke-width="1.5"{dash}/>' )
        svg.append(marker_svg(style["marker"], x + 10, 15, style["color"]))
        svg.append(f'<text x="{x+25:.1f}" y="18" font-size="8">{html.escape(style["label"])}</text>')
        pdf.setStrokeColor(style["color"])
        pdf.setFillColor(style["color"])
        pdf.setLineWidth(1.5)
        pdf.setDash([5, 3] if style["dash"] == "5,3" else ([2, 3] if style["dash"] else []))
        pdf.line(x, HEIGHT - 15, x + 20, HEIGHT - 15)
        pdf.setDash([])
        marker_pdf(pdf, style["marker"], x + 10, HEIGHT - 15)
        pdf.setFillColor("#222222")
        pdf.setFont("Helvetica", 8)
        pdf.drawString(x + 25, HEIGHT - 18, style["label"])

    for task_index, task in enumerate(TASKS):
        panel_left = LEFT + task_index * (PANEL_W + GAP)
        panel_right = panel_left + PANEL_W
        bottom_svg = TOP + PLOT_H
        bottom_pdf = HEIGHT - bottom_svg
        svg.append(f'<text x="{(panel_left+panel_right)/2:.2f}" y="31" font-size="9" text-anchor="middle">{TASK_LABELS[task]}</text>')
        pdf.setFillColor("#222222")
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawCentredString((panel_left + panel_right) / 2, HEIGHT - 31, TASK_LABELS[task])
        for tick in yticks:
            y_svg, y_pdf = ymap_svg(tick), ymap_pdf(tick)
            svg.append(f'<line x1="{panel_left:.2f}" y1="{y_svg:.2f}" x2="{panel_right:.2f}" y2="{y_svg:.2f}" stroke="#d9d9d9" stroke-width="0.55"/>')
            pdf.setStrokeColor("#d9d9d9")
            pdf.setLineWidth(0.55)
            pdf.line(panel_left, y_pdf, panel_right, y_pdf)
            if task_index == 0:
                svg.append(f'<text x="{panel_left-6:.2f}" y="{y_svg+2.6:.2f}" font-size="7.5" text-anchor="end">{tick:.2f}</text>')
                pdf.setFillColor("#222222")
                pdf.setFont("Helvetica", 7.5)
                pdf.drawRightString(panel_left - 6, y_pdf - 2.6, f"{tick:.2f}")
        svg.append(f'<line x1="{panel_left:.2f}" y1="{TOP:.2f}" x2="{panel_left:.2f}" y2="{bottom_svg:.2f}" stroke="#222" stroke-width="0.8"/>')
        svg.append(f'<line x1="{panel_left:.2f}" y1="{bottom_svg:.2f}" x2="{panel_right:.2f}" y2="{bottom_svg:.2f}" stroke="#222" stroke-width="0.8"/>')
        pdf.setStrokeColor("#222222")
        pdf.setLineWidth(0.8)
        pdf.line(panel_left, HEIGHT - TOP, panel_left, bottom_pdf)
        pdf.line(panel_left, bottom_pdf, panel_right, bottom_pdf)
        for xtick in (5, 10, 20, 30, 40, 50):
            x = xmap(float(xtick), panel_left)
            svg.append(f'<line x1="{x:.2f}" y1="{bottom_svg:.2f}" x2="{x:.2f}" y2="{bottom_svg+3:.2f}" stroke="#222" stroke-width="0.7"/>')
            svg.append(f'<text x="{x:.2f}" y="{bottom_svg+13:.2f}" font-size="7.5" text-anchor="middle">{xtick}</text>')
            pdf.line(x, bottom_pdf, x, bottom_pdf - 3)
            pdf.setFont("Helvetica", 7.5)
            pdf.drawCentredString(x, bottom_pdf - 13, str(xtick))
        svg.append(f'<text x="{(panel_left+panel_right)/2:.2f}" y="{HEIGHT-5:.2f}" font-size="8" text-anchor="middle">Budget (%)</text>')
        pdf.setFont("Helvetica", 8)
        pdf.drawCentredString((panel_left + panel_right) / 2, 5, "Budget (%)")
        task_data = subset[subset.task.eq(task)]
        for strategy in ("task_specific", "common", "random"):
            style = STYLES[strategy]
            data = task_data[task_data.strategy.eq(strategy)].sort_values("budget_fraction")
            points_svg = [(xmap(float(row.budget_fraction * 100), panel_left), ymap_svg(float(row.capture_fraction))) for _, row in data.iterrows()]
            points_pdf = [(x, HEIGHT - y) for x, y in points_svg]
            dash = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
            svg.append('<polyline points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in points_svg) + f'" fill="none" stroke="{style["color"]}" stroke-width="1.5"{dash}/>')
            for x, y in points_svg:
                svg.append(marker_svg(style["marker"], x, y, style["color"]))
            pdf.setStrokeColor(style["color"])
            pdf.setFillColor("white")
            pdf.setLineWidth(1.5)
            pdf.setDash([5, 3] if style["dash"] == "5,3" else ([2, 3] if style["dash"] else []))
            for first, second in zip(points_pdf, points_pdf[1:]):
                pdf.line(first[0], first[1], second[0], second[1])
            pdf.setDash([])
            for x, y in points_pdf:
                marker_pdf(pdf, style["marker"], x, y)

    svg.append(f'<text transform="translate(12,{TOP+PLOT_H/2:.2f}) rotate(-90)" font-size="8" text-anchor="middle">Verified-instance capture</text>')
    svg.append("</g></svg>")
    output_svg.write_text("\n".join(svg) + "\n", encoding="utf-8")
    pdf.setFillColor("#222222")
    pdf.saveState()
    pdf.translate(12, HEIGHT - (TOP + PLOT_H / 2))
    pdf.rotate(90)
    pdf.setFont("Helvetica", 8)
    pdf.drawCentredString(0, 0, "Verified-instance capture")
    pdf.restoreState()
    pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    curves = pd.read_csv(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    render(curves, "paper_count", args.output / "fig_rq2_paper_budget.svg", args.output / "fig_rq2_paper_budget.pdf")
    render(curves, "source_token", args.output / "fig_rq2_token_budget.svg", args.output / "fig_rq2_token_budget.pdf")
    print(json.dumps({"renderer": "editable_svg_plus_reportlab_vector_pdf", "reportlab": reportlab.Version, "status": "PASS"}))


if __name__ == "__main__":
    main()
