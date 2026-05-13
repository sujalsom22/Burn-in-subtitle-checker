"""
comparator.py — Mismatch detection module.

Compares transcribed audio text against OCR-extracted subtitle text for each
segment using a two-layer similarity scoring approach. Segments scoring below
a configurable threshold are flagged for human review.

Two-layer approach rationale:
- Single metric is insufficient for noisy Indic OCR output.
- rapidfuzz token_sort_ratio: handles word order variation and character
  substitution common in OCR output (e.g. 'वो' vs 'वह').
- difflib SequenceMatcher: handles character-level sequence alignment
  as a secondary signal for short segments where token sorting is less reliable.
- Final score = weighted average of both signals.
"""

import difflib
import logging
from dataclasses import dataclass
from typing import List, Tuple

from rapidfuzz import fuzz

from config import SIMILARITY_THRESHOLD
from src.transcriber import Segment
from src.ocr_extractor import SubtitleFrame

logger = logging.getLogger(__name__)


@dataclass
class MismatchResult:
    """
    Comparison result for one audio segment vs its subtitle frame.

    Attributes:
        timestamp:      Midpoint timestamp in seconds.
        audio_text:     Transcribed spoken text from Whisper.
        subtitle_text:  OCR-extracted subtitle text from video frame.
        fuzzy_score:    rapidfuzz token_sort_ratio score (0–100).
        sequence_score: difflib SequenceMatcher ratio score (0–100).
        final_score:    Weighted average of both scores (0–100).
        flagged:        True if final_score is below the threshold.
        segment_start:  Start time of the audio segment.
        segment_end:    End time of the audio segment.
    """
    timestamp: float
    audio_text: str
    subtitle_text: str
    fuzzy_score: float
    sequence_score: float
    final_score: float
    flagged: bool
    segment_start: float
    segment_end: float

    @property
    def status(self) -> str:
        """Human-readable status label for the HTML report."""
        return "REVIEW" if self.flagged else "OK"

    def __repr__(self) -> str:
        return (
            f"MismatchResult(timestamp={self.timestamp}s, "
            f"score={self.final_score:.1f}, status={self.status})"
        )


class MismatchDetector:
    """
    Compares audio transcription against OCR subtitle text using
    a two-layer similarity scoring system.

    Usage:
        detector = MismatchDetector(threshold=75.0)
        results = detector.compare(segments, subtitle_frames)
        flagged = [r for r in results if r.flagged]
    """

    FUZZY_WEIGHT = 0.7
    SEQUENCE_WEIGHT = 0.3

    def __init__(self, threshold: float = SIMILARITY_THRESHOLD):
        """
        Args:
            threshold: Similarity score below which a segment is flagged.
                       Range: 0.0 to 100.0. Default: 75.0.
        """
        if not 0.0 <= threshold <= 100.0:
            raise ValueError(
                f"Threshold must be between 0 and 100, got {threshold}"
            )
        self.threshold = threshold

    def compare(
        self,
        segments: List[Segment],
        subtitle_frames: List[SubtitleFrame],
    ) -> List[MismatchResult]:
        """
        Compare audio segments against subtitle frames.

        Args:
            segments:        List of Segment objects from AudioTranscriber.
            subtitle_frames: List of SubtitleFrame objects from SubtitleExtractor.
                             Must be same length and order as segments.

        Returns:
            List of MismatchResult objects, one per segment.

        Raises:
            ValueError: If segments and subtitle_frames lengths do not match.
        """
        if len(segments) != len(subtitle_frames):
            raise ValueError(
                f"segments ({len(segments)}) and subtitle_frames "
                f"({len(subtitle_frames)}) must have equal length."
            )

        results = []

        for segment, frame in zip(segments, subtitle_frames):
            audio_text = segment.text.strip()
            subtitle_text = frame.normalised_text.strip()

            fuzzy_score, sequence_score, final_score = self._compute_scores(
                audio_text, subtitle_text
            )

            flagged = final_score < self.threshold

            results.append(MismatchResult(
                timestamp=frame.timestamp,
                audio_text=audio_text,
                subtitle_text=subtitle_text,
                fuzzy_score=round(fuzzy_score, 2),
                sequence_score=round(sequence_score, 2),
                final_score=round(final_score, 2),
                flagged=flagged,
                segment_start=segment.start,
                segment_end=segment.end,
            ))

            logger.debug(
                f"[{frame.timestamp}s] score={final_score:.1f} "
                f"({'FLAGGED' if flagged else 'OK'}) | "
                f"audio={audio_text!r} | subtitle={subtitle_text!r}"
            )

        flagged_count = sum(1 for r in results if r.flagged)
        logger.info(
            f"Comparison complete: {len(results)} segments, "
            f"{flagged_count} flagged (threshold={self.threshold})"
        )

        return results

    def _compute_scores(
        self,
        audio_text: str,
        subtitle_text: str,
    ) -> Tuple[float, float, float]:
        """
        Compute fuzzy score, sequence score, and weighted final score.

        Handles edge cases:
        - Both empty: perfect match (100.0)
        - One empty, one not: complete mismatch (0.0)

        Args:
            audio_text:    Normalised audio transcription.
            subtitle_text: Normalised OCR subtitle text.

        Returns:
            Tuple of (fuzzy_score, sequence_score, final_score).
            All values in range 0.0 to 100.0.
        """
        if not audio_text and not subtitle_text:
            return 100.0, 100.0, 100.0

        if not audio_text or not subtitle_text:
            return 0.0, 0.0, 0.0

        fuzzy_score = fuzz.token_sort_ratio(audio_text, subtitle_text)

        sequence_score = (
            difflib.SequenceMatcher(
                None, audio_text, subtitle_text
            ).ratio() * 100
        )

        final_score = (
            self.FUZZY_WEIGHT * fuzzy_score
            + self.SEQUENCE_WEIGHT * sequence_score
        )

        return fuzzy_score, sequence_score, final_score


def compare_segments(
    segments: List[Segment],
    subtitle_frames: List[SubtitleFrame],
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[MismatchResult]:
    """
    Convenience function — compare without instantiating the class.

    Args:
        segments:        List of Segment objects.
        subtitle_frames: List of SubtitleFrame objects.
        threshold:       Flagging threshold (default from config).

    Returns:
        List of MismatchResult objects.
    """
    detector = MismatchDetector(threshold=threshold)
    return detector.compare(segments, subtitle_frames)