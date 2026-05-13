"""
evaluator.py — Accuracy evaluation framework for the mismatch detector.

Computes precision, recall, and F1 score against human-verified ground truth.
Enables PlanetRead to validate tool accuracy on their own content and tune
the similarity threshold before deploying to production editorial workflows.

This module answers the question that detection alone cannot:
"How accurately is the tool actually detecting real mismatches?"

Ground truth format (JSON):
    {
        "video": "sample_hindi.mp4",
        "mismatches": [
            {"timestamp": 10.2, "reason": "wrong gender pronoun"},
            {"timestamp": 45.7, "reason": "missing word"}
        ]
    }
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.comparator import MismatchResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Accuracy metrics for the mismatch detector against ground truth.

    Attributes:
        precision:        Of all flagged segments, fraction that are
                          actual mismatches. High precision = few false alarms.
        recall:           Of all actual mismatches, fraction that were
                          correctly flagged. High recall = few missed mismatches.
        f1:               Harmonic mean of precision and recall.
        threshold:        Similarity threshold used for this evaluation.
        true_positives:   Flagged segments that are actual mismatches.
        false_positives:  Flagged segments that are NOT actual mismatches.
        false_negatives:  Actual mismatches that were NOT flagged.
        total_flagged:    Total segments flagged by the detector.
        total_ground_truth: Total mismatches in ground truth.
        tolerance_seconds: Timestamp tolerance used for matching (seconds).
    """
    precision: float
    recall: float
    f1: float
    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_flagged: int
    total_ground_truth: int
    tolerance_seconds: float

    def __repr__(self) -> str:
        return (
            f"EvaluationResult("
            f"precision={self.precision:.3f}, "
            f"recall={self.recall:.3f}, "
            f"f1={self.f1:.3f}, "
            f"threshold={self.threshold})"
        )

    def summary(self) -> str:
        """Human-readable summary string."""
        return (
            f"Threshold: {self.threshold:.1f} | "
            f"Precision: {self.precision:.3f} | "
            f"Recall: {self.recall:.3f} | "
            f"F1: {self.f1:.3f} | "
            f"TP: {self.true_positives} | "
            f"FP: {self.false_positives} | "
            f"FN: {self.false_negatives}"
        )

    def to_dict(self) -> Dict:
        """Serialise to dictionary for JSON output."""
        return {
            "threshold": self.threshold,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_flagged": self.total_flagged,
            "total_ground_truth": self.total_ground_truth,
            "tolerance_seconds": self.tolerance_seconds,
        }


@dataclass
class ThresholdAnalysis:
    """
    F1 score across a range of threshold values.
    Used to find the optimal threshold for a specific video/content type.
    """
    results: List[EvaluationResult] = field(default_factory=list)

    @property
    def best(self) -> Optional[EvaluationResult]:
        """Return the EvaluationResult with the highest F1 score."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.f1)

    def to_dict(self) -> Dict:
        """Serialise to dictionary for JSON output."""
        return {
            "best_threshold": self.best.threshold if self.best else None,
            "best_f1": round(self.best.f1, 4) if self.best else None,
            "sweep": [r.to_dict() for r in self.results],
        }


class MismatchEvaluator:
    """
    Evaluates mismatch detector accuracy against human-verified ground truth.

    Usage:
        evaluator = MismatchEvaluator(tolerance_seconds=2.0)

        # Single threshold evaluation
        result = evaluator.evaluate(detections, ground_truth_path)
        print(result.summary())

        # Threshold sweep — find best threshold for your content
        analysis = evaluator.threshold_sweep(detections, ground_truth_path)
        print(f"Best threshold: {analysis.best.threshold}")
        evaluator.save_results(analysis, "evaluation.json")
    """

    def __init__(self, tolerance_seconds: float = 2.0):
        """
        Args:
            tolerance_seconds: Maximum timestamp difference (seconds) for a
                               detection to be considered matching a ground
                               truth mismatch. Default 2.0s accounts for
                               Whisper segment boundary variation.
        """
        if tolerance_seconds < 0:
            raise ValueError(
                f"tolerance_seconds must be non-negative, "
                f"got {tolerance_seconds}"
            )
        self.tolerance_seconds = tolerance_seconds

    def evaluate(
        self,
        detections: List[MismatchResult],
        ground_truth_path: str,
        threshold: Optional[float] = None,
    ) -> EvaluationResult:
        """
        Evaluate detector accuracy against ground truth at a single threshold.

        Args:
            detections:        List of MismatchResult from MismatchDetector.
            ground_truth_path: Path to ground truth JSON file.
            threshold:         Override threshold for evaluation. If None,
                               uses the threshold already applied in detections
                               (i.e. uses result.flagged as-is).

        Returns:
            EvaluationResult with precision, recall, F1 metrics.

        Raises:
            FileNotFoundError: If ground_truth_path does not exist.
            ValueError:        If ground truth JSON is malformed.
        """
        ground_truth = self._load_ground_truth(ground_truth_path)
        gt_timestamps = [m["timestamp"] for m in ground_truth["mismatches"]]

        if threshold is not None:
            flagged = [
                d for d in detections
                if d.final_score < threshold
            ]
            used_threshold = threshold
        else:
            flagged = [d for d in detections if d.flagged]
            used_threshold = detections[0].final_score if detections else 0.0

        flagged_timestamps = [d.timestamp for d in flagged]

        tp, fp, fn = self._compute_tp_fp_fn(
            flagged_timestamps, gt_timestamps
        )

        precision = self._precision(tp, fp)
        recall = self._recall(tp, fn)
        f1 = self._f1(precision, recall)

        return EvaluationResult(
            precision=precision,
            recall=recall,
            f1=f1,
            threshold=used_threshold,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            total_flagged=len(flagged),
            total_ground_truth=len(gt_timestamps),
            tolerance_seconds=self.tolerance_seconds,
        )

    def threshold_sweep(
        self,
        detections: List[MismatchResult],
        ground_truth_path: str,
        start: float = 50.0,
        end: float = 95.0,
        step: float = 5.0,
    ) -> ThresholdAnalysis:
        """
        Evaluate at multiple thresholds to find the optimal setting.

        Sweeps threshold from start to end (inclusive) in steps,
        computing precision, recall, and F1 at each value.

        Args:
            detections:        List of MismatchResult objects.
            ground_truth_path: Path to ground truth JSON file.
            start:             Lowest threshold to evaluate (default 50.0).
            end:               Highest threshold to evaluate (default 95.0).
            step:              Step size between thresholds (default 5.0).

        Returns:
            ThresholdAnalysis with results for each threshold and best F1.
        """
        analysis = ThresholdAnalysis()
        threshold = start

        while threshold <= end + 1e-9:
            result = self.evaluate(
                detections,
                ground_truth_path,
                threshold=round(threshold, 1),
            )
            analysis.results.append(result)
            logger.debug(f"Threshold {threshold:.1f}: {result.summary()}")
            threshold += step

        if analysis.best:
            logger.info(
                f"Threshold sweep complete. "
                f"Best F1={analysis.best.f1:.3f} "
                f"at threshold={analysis.best.threshold}"
            )

        return analysis

    def save_results(
        self,
        result,
        output_path: str,
    ) -> str:
        """
        Save evaluation results to JSON file.

        Args:
            result:      EvaluationResult or ThresholdAnalysis object.
            output_path: Path to write JSON output.

        Returns:
            Path to written file.
        """
        if isinstance(result, ThresholdAnalysis):
            data = result.to_dict()
        elif isinstance(result, EvaluationResult):
            data = result.to_dict()
        else:
            raise TypeError(
                f"Expected EvaluationResult or ThresholdAnalysis, "
                f"got {type(result)}"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation results saved to '{output_path}'")
        return output_path

    def _load_ground_truth(self, path: str) -> Dict:
        """Load and validate ground truth JSON file."""
        if not __import__("os").path.exists(path):
            raise FileNotFoundError(
                f"Ground truth file not found: '{path}'"
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "mismatches" not in data:
            raise ValueError(
                f"Ground truth JSON must contain a 'mismatches' key. "
                f"Got keys: {list(data.keys())}"
            )

        if not isinstance(data["mismatches"], list):
            raise ValueError(
                "'mismatches' must be a list of objects with 'timestamp' keys."
            )

        for i, m in enumerate(data["mismatches"]):
            if "timestamp" not in m:
                raise ValueError(
                    f"Mismatch entry {i} is missing 'timestamp' key: {m}"
                )

        return data

    def _compute_tp_fp_fn(
        self,
        flagged_timestamps: List[float],
        gt_timestamps: List[float],
    ):
        """
        Compute true positives, false positives, false negatives.

        Matching strategy: a flagged timestamp is a true positive if there
        exists a ground truth timestamp within tolerance_seconds.
        Each ground truth mismatch can only be matched once (greedy matching).

        Args:
            flagged_timestamps: Timestamps flagged by the detector.
            gt_timestamps:      Timestamps from ground truth.

        Returns:
            Tuple of (true_positives, false_positives, false_negatives).
        """
        matched_gt = set()
        tp = 0
        fp = 0

        for flagged_ts in flagged_timestamps:
            matched = False
            for i, gt_ts in enumerate(gt_timestamps):
                if i in matched_gt:
                    continue
                if abs(flagged_ts - gt_ts) <= self.tolerance_seconds:
                    tp += 1
                    matched_gt.add(i)
                    matched = True
                    break
            if not matched:
                fp += 1

        fn = len(gt_timestamps) - len(matched_gt)
        return tp, fp, fn

    @staticmethod
    def _precision(tp: int, fp: int) -> float:
        """Precision = TP / (TP + FP). Returns 0.0 if no detections."""
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    @staticmethod
    def _recall(tp: int, fn: int) -> float:
        """Recall = TP / (TP + FN). Returns 0.0 if no ground truth."""
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        """F1 = 2 * P * R / (P + R). Returns 0.0 if both are zero."""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)