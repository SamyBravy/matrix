#!/usr/bin/env python3
"""Update README coverage tables from cargo llvm-cov JSON summary."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PER_FILE_START = "<!-- COVERAGE:PER-FILE-START -->"
PER_FILE_END = "<!-- COVERAGE:PER-FILE-END -->"
CRATE_START = "<!-- COVERAGE:CRATE-START -->"
CRATE_END = "<!-- COVERAGE:CRATE-END -->"
WORKSPACE_START = "<!-- COVERAGE:WORKSPACE-START -->"
WORKSPACE_END = "<!-- COVERAGE:WORKSPACE-END -->"
BADGES_START = "<!-- COVERAGE:BADGES-START -->"
BADGES_END = "<!-- COVERAGE:BADGES-END -->"

MetricDict = Dict[str, Any]


@dataclass
class CoverageRow:
    crate: str
    path: str
    functions: MetricDict
    lines: MetricDict
    regions: MetricDict
    branches: MetricDict

    @property
    def formatted(self) -> str:
        return (
            f"| {self.crate} | {self.path} | "
            f"{format_percent(self.functions)} | "
            f"{format_percent(self.lines)} | "
            f"{format_percent(self.regions)} | "
            f"{format_percent(self.branches)} |"
        )


def format_percent(metric: MetricDict) -> str:
    count = metric.get("count", 0)
    percent = metric.get("percent", 0.0) or 0.0
    if count in (0, None):
        return "—"
    return f"{percent:.2f}%"


def aggregate_metrics(rows: Iterable[CoverageRow]) -> Dict[str, MetricDict]:
    total: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0.0, "covered": 0.0})
    for row in rows:
        for name, metric in (
            ("functions", row.functions),
            ("lines", row.lines),
            ("regions", row.regions),
            ("branches", row.branches),
        ):
            total[name]["count"] += float(metric.get("count", 0) or 0)
            total[name]["covered"] += float(metric.get("covered", 0) or 0)
    result: Dict[str, MetricDict] = {}
    for name, values in total.items():
        count = values["count"]
        covered = values["covered"]
        percent = (covered / count * 100.0) if count else 0.0
        result[name] = {"count": count, "covered": covered, "percent": percent}
    return result


def format_aggregate_row(label: str, metrics: Dict[str, MetricDict]) -> str:
    return (
        f"| {label} | "
        f"{format_percent(metrics['functions'])} | "
        f"{format_percent(metrics['lines'])} | "
        f"{format_percent(metrics['regions'])} | "
        f"{format_percent(metrics['branches'])} |"
    )


def replace_section(content: str, start: str, end: str, payload: str) -> str:
    pattern = re.compile(
        re.escape(start) + r".*?" + re.escape(end),
        flags=re.DOTALL,
    )
    replacement = f"{start}\n{payload}\n{end}"
    if not pattern.search(content):
        raise SystemExit(f"Markers {start} / {end} not found in README")
    return pattern.sub(replacement, content)


def detect_crate(path: Path, repo_root: Path) -> Tuple[str, Path]:
    try:
        rel = path.relative_to(repo_root)
    except ValueError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"File {path} is outside repo root {repo_root}") from exc
    parts = rel.parts
    crate = next((p for p in parts if p in {"Lorenzo", "Samuele"}), "workspace")
    return crate, rel


def load_rows(summary: Dict[str, Any]) -> List[CoverageRow]:
    cargo_info = summary.get("cargo_llvm_cov", {})
    manifest_path = cargo_info.get("manifest_path")
    if not manifest_path:
        raise SystemExit("manifest_path missing from coverage summary")
    repo_root = Path(manifest_path).parent.resolve()

    rows: List[CoverageRow] = []
    data = summary.get("data", [])
    if not data:
        return rows
    files = data[0].get("files", [])
    for entry in files:
        filename = entry.get("filename")
        if not filename:
            continue
        crate, rel_path = detect_crate(Path(filename), repo_root)
        summary_metrics = entry.get("summary", {})
        row = CoverageRow(
            crate=crate,
            path=str(rel_path),
            functions=summary_metrics.get("functions", {}),
            lines=summary_metrics.get("lines", {}),
            regions=summary_metrics.get("regions", {}),
            branches=summary_metrics.get("branches", {}),
        )
        rows.append(row)
    rows.sort(key=lambda r: (r.crate, r.path))
    return rows


def build_tables(rows: List[CoverageRow]) -> Tuple[str, str, str, Dict[str, MetricDict]]:
    if not rows:
        placeholder = "_No data available; run coverage generation first._"
        return placeholder, placeholder, placeholder, {
            "functions": {"count": 0, "covered": 0, "percent": 0.0},
            "lines": {"count": 0, "covered": 0, "percent": 0.0},
            "regions": {"count": 0, "covered": 0, "percent": 0.0},
            "branches": {"count": 0, "covered": 0, "percent": 0.0},
        }

    per_file_lines = [
        "| Crate | File | Function Coverage | Line Coverage | Region Coverage | Branch Coverage |",
        "|-------|------|-------------------|---------------|-----------------|-----------------|",
    ]
    for row in rows:
        per_file_lines.append(row.formatted)

    crate_groups: Dict[str, List[CoverageRow]] = defaultdict(list)
    for row in rows:
        crate_groups[row.crate].append(row)

    crate_lines = [
        "| Crate | Function Coverage | Line Coverage | Region Coverage | Branch Coverage |",
        "|-------|-------------------|---------------|-----------------|-----------------|",
    ]
    for crate in sorted(crate_groups):
        metrics = aggregate_metrics(crate_groups[crate])
        crate_lines.append(format_aggregate_row(crate, metrics))

    workspace_metrics = aggregate_metrics(rows)
    workspace_lines = [
        "| Scope | Function Coverage | Line Coverage | Region Coverage | Branch Coverage |",
        "|-------|-------------------|---------------|-----------------|-----------------|",
        format_aggregate_row("Workspace", workspace_metrics),
    ]

    return (
        "\n".join(per_file_lines),
        "\n".join(crate_lines),
        "\n".join(workspace_lines),
        workspace_metrics,
    )


def select_badge_color(percent: float | None) -> str:
    if percent is None:
        return "#9e9e9e"
    if percent >= 90:
        return "#4c1"
    if percent >= 80:
        return "#97ca00"
    if percent >= 70:
        return "#a4a61d"
    if percent >= 60:
        return "#dfb317"
    if percent >= 50:
        return "#fe7d37"
    return "#e05d44"


def calc_segment_width(text: str) -> int:
    return max(60, int(len(text) * 6.5) + 20)


def render_badge(label: str, value_text: str, color: str, output_path: Path) -> None:
    label_width = calc_segment_width(label)
    value_width = calc_segment_width(value_text)
    total_width = label_width + value_width
    svg = f"""
<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{total_width}\" height=\"20\" role=\"img\" aria-label=\"{label}: {value_text}\">
  <linearGradient id=\"a\" x2=\"0\" y2=\"100%\">
    <stop offset=\"0\" stop-color=\"#bbb\" stop-opacity=\".1\"/>
    <stop offset=\"1\" stop-opacity=\".1\"/>
  </linearGradient>
  <rect rx=\"3\" width=\"{total_width}\" height=\"20\" fill=\"#555\"/>
  <rect rx=\"3\" x=\"{label_width}\" width=\"{value_width}\" height=\"20\" fill=\"{color}\"/>
  <rect rx=\"3\" width=\"{total_width}\" height=\"20\" fill=\"url(#a)\"/>
  <g fill=\"#fff\" text-anchor=\"middle\" font-family=\"DejaVu Sans,Verdana,Geneva,sans-serif\" font-size=\"11\">
    <text x=\"{label_width / 2:.1f}\" y=\"15\" fill=\"#010101\" fill-opacity=\".3\">{label}</text>
    <text x=\"{label_width / 2:.1f}\" y=\"14\">{label}</text>
    <text x=\"{label_width + value_width / 2:.1f}\" y=\"15\" fill=\"#010101\" fill-opacity=\".3\">{value_text}</text>
    <text x=\"{label_width + value_width / 2:.1f}\" y=\"14\">{value_text}</text>
  </g>
</svg>
"""
    output_path.write_text(svg.strip() + "\n")


def generate_badges(workspace_metrics: Dict[str, MetricDict], output_dir: Path) -> Dict[str, Path]:
    badge_specs = [
        ("Functions", "functions"),
        ("Lines", "lines"),
        ("Regions", "regions"),
        ("Branches", "branches"),
    ]
    generated: Dict[str, Path] = {}
    for label, key in badge_specs:
        metric = workspace_metrics.get(key, {"count": 0, "percent": 0.0})
        count = metric.get("count", 0)
        percent = metric.get("percent", 0.0) if count else None
        value_text = "N/A" if percent is None else f"{percent:.1f}%"
        color = select_badge_color(percent)
        filename = f"coverage-{key}.svg"
        output_path = output_dir / filename
        render_badge(label, value_text, color, output_path)
        generated[key] = output_path
    return generated


def main() -> None:
    if len(sys.argv) not in {1, 3}:
        raise SystemExit("Usage: update_readme_coverage.py [coverage-summary.json README.md]")

    summary_path = Path(sys.argv[1]) if len(sys.argv) == 3 else Path("coverage-summary.json")
    readme_path = Path(sys.argv[2]) if len(sys.argv) == 3 else Path("README.md")

    if not summary_path.is_file():
        raise SystemExit(f"Coverage summary not found: {summary_path}")
    if not readme_path.is_file():
        raise SystemExit(f"README not found: {readme_path}")

    summary = json.loads(summary_path.read_text())
    rows = load_rows(summary)
    per_file, per_crate, workspace, workspace_metrics = build_tables(rows)

    content = readme_path.read_text()
    content = replace_section(content, PER_FILE_START, PER_FILE_END, per_file)
    content = replace_section(content, CRATE_START, CRATE_END, per_crate)
    content = replace_section(content, WORKSPACE_START, WORKSPACE_END, workspace)

    generated_badges = generate_badges(workspace_metrics, readme_path.parent)
    badge_markup = " ".join(
        f"![{label.capitalize()}](./{path.name})"
        for label, path in (
            ("functions", generated_badges["functions"]),
            ("lines", generated_badges["lines"]),
            ("regions", generated_badges["regions"]),
            ("branches", generated_badges["branches"]),
        )
    )
    content = replace_section(content, BADGES_START, BADGES_END, badge_markup)

    readme_path.write_text(content)

if __name__ == "__main__":  # pragma: no cover
    main()
