"""
cli.py — Command line interface entry point.

Usage:
    python -m src.cli --video path/to/video.mp4
    python -m src.cli --video video.mp4 --model-size small --threshold 75
    python -m src.cli --video video.mp4 --lang hin+kan+eng --output report.html
    python -m src.cli --video video.mp4 --evaluate --ground-truth truth.json
"""

import argparse
import logging
import os
import sys

from config import (
    WHISPER_MODEL_SIZE,
    OCR_LANGUAGES,
    SIMILARITY_THRESHOLD,
    SUBTITLE_REGION_START,
    OUTPUT_REPORT_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="subtitle-checker",
        description=(
            "Automatically detect mismatches between audio dialogue and "
            "burned-in subtitles in video files. Supports Hindi and Kannada."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage (Hindi video):
    python -m src.cli --video lecture.mp4

  Kannada video with custom threshold:
    python -m src.cli --video kannada_video.mp4 --lang kan+eng --threshold 70

  Use medium model for better Indic accuracy:
    python -m src.cli --video video.mp4 --model-size medium

  Evaluate accuracy against ground truth:
    python -m src.cli --video video.mp4 --evaluate --ground-truth truth.json

  Run threshold sweep to find best settings:
    python -m src.cli --video video.mp4 --evaluate --ground-truth truth.json --sweep
        """,
    )

    parser.add_argument(
        "--video",
        required=True,
        metavar="PATH",
        help="Path to the input video file (.mp4, .avi, .mkv, etc.)",
    )

    parser.add_argument(
        "--model-size",
        default=WHISPER_MODEL_SIZE,
        choices=["tiny", "base", "small", "medium", "large"],
        metavar="SIZE",
        help=(
            f"Whisper model size. Larger = more accurate but slower. "
            f"Recommended: 'small' for most content, 'medium' for "
            f"challenging Indic audio. Default: {WHISPER_MODEL_SIZE}"
        ),
    )

    parser.add_argument(
        "--language",
        default=None,
        metavar="CODE",
        help=(
            "Force Whisper language detection. Use ISO codes: "
            "'hi' for Hindi, 'kn' for Kannada. "
            "Default: auto-detect."
        ),
    )

    parser.add_argument(
        "--lang",
        default=OCR_LANGUAGES,
        metavar="LANGS",
        help=(
            f"Tesseract OCR language string. Use '+' to combine: "
            f"'hin+kan+eng' for Hindi+Kannada+English. "
            f"Default: {OCR_LANGUAGES}"
        ),
    )

    parser.add_argument(
        "--subtitle-region",
        type=float,
        default=SUBTITLE_REGION_START,
        metavar="FRACTION",
        help=(
            f"Fraction of frame height where subtitle region starts. "
            f"0.85 = bottom 15%% of frame. "
            f"Default: {SUBTITLE_REGION_START}"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        metavar="SCORE",
        help=(
            f"Similarity score (0–100) below which a segment is flagged. "
            f"Lower = more flags (higher recall, lower precision). "
            f"Default: {SIMILARITY_THRESHOLD}"
        ),
    )

    parser.add_argument(
        "--output",
        default=OUTPUT_REPORT_NAME,
        metavar="PATH",
        help=f"Output HTML report path. Default: {OUTPUT_REPORT_NAME}",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run accuracy evaluation against ground truth after detection.",
    )

    parser.add_argument(
        "--ground-truth",
        default=None,
        metavar="PATH",
        help=(
            "Path to ground truth JSON file (required when --evaluate is set). "
            "Format: {\"mismatches\": [{\"timestamp\": 10.2, \"reason\": \"...\"}]}"
        ),
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Run threshold sweep (50–95) to find optimal threshold. "
            "Requires --evaluate and --ground-truth."
        ),
    )

    parser.add_argument(
        "--eval-output",
        default="evaluation.json",
        metavar="PATH",
        help="Output path for evaluation JSON results. Default: evaluation.json",
    )

    return parser


def run(args) -> int:
    """
    Main pipeline execution.

    Args:
        args: Parsed argparse namespace.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: '{args.video}'")
        return 1

    if args.evaluate and not args.ground_truth:
        logger.error("--ground-truth is required when --evaluate is set.")
        return 1

    if args.ground_truth and not os.path.exists(args.ground_truth):
        logger.error(f"Ground truth file not found: '{args.ground_truth}'")
        return 1

    video_name = os.path.basename(args.video)

    logger.info(f"Step 1/3 — Transcribing audio ({args.model_size} model)...")
    from src.transcriber import AudioTranscriber
    transcriber = AudioTranscriber(
        model_size=args.model_size,
        language=args.language,
    )
    segments = transcriber.transcribe(args.video)

    if not segments:
        logger.warning("No audio segments found. Check that video has audio.")
        return 1

    logger.info(f"Found {len(segments)} audio segments.")

    logger.info("Step 2/3 — Extracting subtitle text from frames...")
    from src.ocr_extractor import SubtitleExtractor
    extractor = SubtitleExtractor(
        languages=args.lang,
        subtitle_region_start=args.subtitle_region,
    )
    subtitle_frames = extractor.extract(args.video, segments)

    non_empty = sum(1 for f in subtitle_frames if not f.is_empty)
    logger.info(f"Extracted {non_empty}/{len(subtitle_frames)} non-empty subtitle frames.")

    logger.info(f"Step 3/3 — Detecting mismatches (threshold={args.threshold})...")
    from src.comparator import MismatchDetector
    detector = MismatchDetector(threshold=args.threshold)
    results = detector.compare(segments, subtitle_frames)

    flagged = [r for r in results if r.flagged]
    logger.info(
        f"Detection complete: {len(flagged)}/{len(results)} segments flagged."
    )

    from src.reporter import HTMLReporter
    reporter = HTMLReporter()
    report_path = reporter.generate(results, args.output, video_name=video_name)
    logger.info(f"Report written to: {report_path}")
    print(f"\nReport: {os.path.abspath(report_path)}")
    print(f"Flagged: {len(flagged)} / {len(results)} segments")

    if args.evaluate:
        from src.evaluator import MismatchEvaluator
        evaluator = MismatchEvaluator()

        if args.sweep:
            logger.info("Running threshold sweep (50–95)...")
            analysis = evaluator.threshold_sweep(results, args.ground_truth)
            evaluator.save_results(analysis, args.eval_output)
            if analysis.best:
                print(f"\nBest threshold: {analysis.best.threshold}")
                print(f"Best F1:        {analysis.best.f1:.3f}")
                print(f"Precision:      {analysis.best.precision:.3f}")
                print(f"Recall:         {analysis.best.recall:.3f}")
        else:
            eval_result = evaluator.evaluate(
                results,
                args.ground_truth,
                threshold=args.threshold,
            )
            evaluator.save_results(eval_result, args.eval_output)
            print(f"\nEvaluation:")
            print(f"  Precision: {eval_result.precision:.3f}")
            print(f"  Recall:    {eval_result.recall:.3f}")
            print(f"  F1:        {eval_result.f1:.3f}")

        logger.info(f"Evaluation results saved to: {args.eval_output}")

    return 0


def main():
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    exit_code = run(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()