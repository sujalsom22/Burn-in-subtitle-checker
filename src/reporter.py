"""
reporter.py — HTML report generation module.

Generates a human-readable HTML report listing all comparison results,
with flagged segments highlighted for human review. The report format
matches the expected output specified in Issue #3.

Expected output format from Issue #3:
    Timestamp | Audio Text | Subtitle Text | Match Score
    10.2s     | वो कहाँ गई थी | वो कहाँ गया था | 0.61 — REVIEW
    45.7s     | ठीक है भाई   | ठीक है भाई    | 0.95 — OK
"""

import os
import logging
from datetime import datetime
from typing import List

from src.comparator import MismatchResult

logger = logging.getLogger(__name__)


class HTMLReporter:
    """
    Generates an HTML mismatch report from comparison results.

    Usage:
        reporter = HTMLReporter()
        reporter.generate(results, output_path="mismatch_report.html")
    """

    def generate(
        self,
        results: List[MismatchResult],
        output_path: str = "mismatch_report.html",
        video_name: str = "",
    ) -> str:
        """
        Generate HTML report and write to file.

        Args:
            results:     List of MismatchResult objects from MismatchDetector.
            output_path: Path to write the HTML report.
            video_name:  Optional video filename shown in report header.

        Returns:
            Path to the generated report file.

        Raises:
            IOError: If the output file cannot be written.
        """
        html = self._build_html(results, video_name)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        flagged_count = sum(1 for r in results if r.flagged)
        logger.info(
            f"Report written to '{output_path}' — "
            f"{len(results)} segments, {flagged_count} flagged."
        )

        return output_path

    def _build_html(
        self,
        results: List[MismatchResult],
        video_name: str,
    ) -> str:
        """Build complete HTML string from results."""
        flagged_count = sum(1 for r in results if r.flagged)
        total_count = len(results)
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        rows = "\n".join(
            self._build_row(r) for r in results
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Subtitle Mismatch Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: #f8f9fa;
            color: #212529;
        }}
        h1 {{
            color: #343a40;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 0.5rem;
        }}
        .meta {{
            background: #fff;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 1rem 1.5rem;
            margin-bottom: 1.5rem;
            display: flex;
            gap: 2rem;
            flex-wrap: wrap;
        }}
        .meta-item {{
            font-size: 0.9rem;
            color: #6c757d;
        }}
        .meta-item strong {{
            color: #212529;
            display: block;
            font-size: 1.1rem;
        }}
        .summary-flagged {{ color: #dc3545; }}
        .summary-ok {{ color: #28a745; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #fff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th {{
            background: #343a40;
            color: #fff;
            padding: 0.75rem 1rem;
            text-align: left;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #dee2e6;
            font-size: 0.95rem;
            vertical-align: top;
        }}
        tr:last-child td {{ border-bottom: none; }}
        tr.flagged {{ background: #fff5f5; }}
        tr.flagged:hover {{ background: #ffe0e0; }}
        tr.ok:hover {{ background: #f8f9fa; }}
        .timestamp {{
            font-family: monospace;
            color: #6c757d;
            white-space: nowrap;
        }}
        .score {{
            font-family: monospace;
            font-weight: 600;
            white-space: nowrap;
        }}
        .score.review {{ color: #dc3545; }}
        .score.ok {{ color: #28a745; }}
        .badge {{
            display: inline-block;
            padding: 0.2em 0.6em;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-review {{
            background: #f8d7da;
            color: #721c24;
        }}
        .badge-ok {{
            background: #d4edda;
            color: #155724;
        }}
        .text-indic {{
            font-size: 1.05rem;
            line-height: 1.5;
        }}
        .no-results {{
            text-align: center;
            padding: 3rem;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <h1>Subtitle Mismatch Report</h1>

    <div class="meta">
        <div class="meta-item">
            <strong>{video_name if video_name else "—"}</strong>
            Video File
        </div>
        <div class="meta-item">
            <strong>{total_count}</strong>
            Total Segments
        </div>
        <div class="meta-item">
            <strong class="summary-flagged">{flagged_count}</strong>
            Flagged for Review
        </div>
        <div class="meta-item">
            <strong class="summary-ok">{total_count - flagged_count}</strong>
            OK
        </div>
        <div class="meta-item">
            <strong>{generated_at}</strong>
            Generated At
        </div>
    </div>

    {"<p class='no-results'>No segments to display.</p>" if not results else f"""
    <table>
        <thead>
            <tr>
                <th>Timestamp</th>
                <th>Audio Text</th>
                <th>Subtitle Text</th>
                <th>Match Score</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """}
</body>
</html>"""

        return html

    def _build_row(self, result: MismatchResult) -> str:
        """Build one HTML table row for a MismatchResult."""
        row_class = "flagged" if result.flagged else "ok"
        score_class = "review" if result.flagged else "ok"
        badge_class = "badge-review" if result.flagged else "badge-ok"

        score_display = f"{result.final_score / 100:.2f}"

        return f"""            <tr class="{row_class}">
                <td class="timestamp">{result.timestamp:.1f}s</td>
                <td class="text-indic">{self._escape(result.audio_text)}</td>
                <td class="text-indic">{self._escape(result.subtitle_text)}</td>
                <td class="score {score_class}">{score_display}</td>
                <td><span class="badge {badge_class}">{result.status}</span></td>
            </tr>"""

    @staticmethod
    def _escape(text: str) -> str:
        """Escape HTML special characters in text content."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )


def generate_report(
    results: List[MismatchResult],
    output_path: str = "mismatch_report.html",
    video_name: str = "",
) -> str:
    """
    Convenience function — generate report without instantiating the class.

    Args:
        results:     List of MismatchResult objects.
        output_path: Output HTML file path.
        video_name:  Optional video filename for report header.

    Returns:
        Path to generated report.
    """
    reporter = HTMLReporter()
    return reporter.generate(results, output_path, video_name)