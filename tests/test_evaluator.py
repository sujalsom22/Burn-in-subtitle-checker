"""
Tests for MismatchEvaluator, EvaluationResult, and ThresholdAnalysis.
"""

import json
import pytest
from src.comparator import MismatchResult
from src.evaluator import MismatchEvaluator, EvaluationResult, ThresholdAnalysis

def make_result(timestamp, flagged=True, score=50.0):
    return MismatchResult(
        timestamp=timestamp,
        audio_text="audio",
        subtitle_text="subtitle",
        fuzzy_score=score,
        sequence_score=score,
        final_score=score,
        flagged=flagged,
        segment_start=timestamp - 1.0,
        segment_end=timestamp + 1.0,
    )

def write_ground_truth(path, mismatches):
    data = {"video": "test.mp4", "mismatches": mismatches}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path

class TestMismatchEvaluatorInit:

    def test_default_tolerance(self):
        e = MismatchEvaluator()
        assert e.tolerance_seconds == 2.0

    def test_custom_tolerance(self):
        e = MismatchEvaluator(tolerance_seconds=1.0)
        assert e.tolerance_seconds == 1.0

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError):
            MismatchEvaluator(tolerance_seconds=-1.0)

    def test_zero_tolerance_accepted(self):
        e = MismatchEvaluator(tolerance_seconds=0.0)
        assert e.tolerance_seconds == 0.0

class TestGroundTruthLoading:

    def test_file_not_found_raises(self):
        e = MismatchEvaluator()
        with pytest.raises(FileNotFoundError):
            e._load_ground_truth("nonexistent.json")

    def test_missing_mismatches_key_raises(self, tmp_path):
        f = tmp_path / "gt.json"
        f.write_text('{"video": "test.mp4"}')
        e = MismatchEvaluator()
        with pytest.raises(ValueError, match="mismatches"):
            e._load_ground_truth(str(f))

    def test_missing_timestamp_in_entry_raises(self, tmp_path):
        f = tmp_path / "gt.json"
        f.write_text('{"mismatches": [{"reason": "test"}]}')
        e = MismatchEvaluator()
        with pytest.raises(ValueError, match="timestamp"):
            e._load_ground_truth(str(f))

    def test_valid_ground_truth_loads(self, tmp_path):
        f = tmp_path / "gt.json"
        write_ground_truth(str(f), [{"timestamp": 10.0, "reason": "test"}])
        e = MismatchEvaluator()
        data = e._load_ground_truth(str(f))
        assert len(data["mismatches"]) == 1

class TestTPFPFN:

    def test_perfect_detection(self):
        e = MismatchEvaluator(tolerance_seconds=2.0)
        tp, fp, fn = e._compute_tp_fp_fn([10.0, 45.0], [10.0, 45.0])
        assert tp == 2
        assert fp == 0
        assert fn == 0

    def test_all_false_positives(self):
        e = MismatchEvaluator(tolerance_seconds=2.0)
        tp, fp, fn = e._compute_tp_fp_fn([10.0, 20.0], [50.0, 60.0])
        assert tp == 0
        assert fp == 2
        assert fn == 2

    def test_all_false_negatives(self):
        e = MismatchEvaluator(tolerance_seconds=2.0)
        tp, fp, fn = e._compute_tp_fp_fn([], [10.0, 20.0])
        assert tp == 0
        assert fp == 0
        assert fn == 2

    def test_within_tolerance_is_tp(self):
        e = MismatchEvaluator(tolerance_seconds=2.0)
        tp, fp, fn = e._compute_tp_fp_fn([10.0], [11.5])
        assert tp == 1
        assert fp == 0
        assert fn == 0

    def test_outside_tolerance_is_fp_and_fn(self):
        e = MismatchEvaluator(tolerance_seconds=2.0)
        tp, fp, fn = e._compute_tp_fp_fn([10.0], [15.0])
        assert tp == 0
        assert fp == 1
        assert fn == 1

    def test_each_gt_matched_only_once(self):
        e = MismatchEvaluator(tolerance_seconds=2.0)
        tp, fp, fn = e._compute_tp_fp_fn([10.0, 10.5], [10.0])
        assert tp == 1
        assert fp == 1
        assert fn == 0

    def test_empty_both(self):
        e = MismatchEvaluator(tolerance_seconds=2.0)
        tp, fp, fn = e._compute_tp_fp_fn([], [])
        assert tp == 0
        assert fp == 0
        assert fn == 0

class TestMetrics:

    def test_precision_perfect(self):
        assert MismatchEvaluator._precision(5, 0) == 1.0

    def test_precision_zero_detections(self):
        assert MismatchEvaluator._precision(0, 0) == 0.0

    def test_precision_half(self):
        assert MismatchEvaluator._precision(1, 1) == 0.5

    def test_recall_perfect(self):
        assert MismatchEvaluator._recall(5, 0) == 1.0

    def test_recall_zero_ground_truth(self):
        assert MismatchEvaluator._recall(0, 0) == 0.0

    def test_recall_half(self):
        assert MismatchEvaluator._recall(1, 1) == 0.5

    def test_f1_perfect(self):
        assert MismatchEvaluator._f1(1.0, 1.0) == 1.0

    def test_f1_both_zero(self):
        assert MismatchEvaluator._f1(0.0, 0.0) == 0.0

    def test_f1_harmonic_mean(self):
        f1 = MismatchEvaluator._f1(0.8, 0.6)
        assert abs(f1 - 0.6857) < 0.001

class TestEvaluate:

    def test_perfect_detection_gives_f1_1(self, tmp_path):
        gt = tmp_path / "gt.json"
        write_ground_truth(str(gt), [
            {"timestamp": 10.0, "reason": "test"},
            {"timestamp": 45.0, "reason": "test"},
        ])
        detections = [
            make_result(10.0, flagged=True, score=50.0),
            make_result(45.0, flagged=True, score=50.0),
            make_result(30.0, flagged=False, score=90.0),  # score above threshold
        ]
        e = MismatchEvaluator()
        result = e.evaluate(detections, str(gt), threshold=75.0)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_no_detections_gives_zero_precision_recall(self, tmp_path):
        gt = tmp_path / "gt.json"
        write_ground_truth(str(gt), [{"timestamp": 10.0, "reason": "test"}])
        detections = [make_result(10.0, flagged=False, score=90.0)]
        e = MismatchEvaluator()
        result = e.evaluate(detections, str(gt), threshold=75.0)
        assert result.recall == 0.0

    def test_result_contains_correct_threshold(self, tmp_path):
        gt = tmp_path / "gt.json"
        write_ground_truth(str(gt), [{"timestamp": 10.0, "reason": "test"}])
        detections = [make_result(10.0)]
        e = MismatchEvaluator()
        result = e.evaluate(detections, str(gt), threshold=80.0)
        assert result.threshold == 80.0

class TestThresholdSweep:

    def test_sweep_produces_correct_number_of_results(self, tmp_path):
        gt = tmp_path / "gt.json"
        write_ground_truth(str(gt), [{"timestamp": 10.0, "reason": "test"}])
        detections = [make_result(10.0)]
        e = MismatchEvaluator()
        analysis = e.threshold_sweep(
            detections, str(gt), start=50.0, end=75.0, step=5.0
        )
        assert len(analysis.results) == 6

    def test_best_returns_highest_f1(self, tmp_path):
        gt = tmp_path / "gt.json"
        write_ground_truth(str(gt), [{"timestamp": 10.0, "reason": "test"}])
        detections = [make_result(10.0, score=60.0)]
        e = MismatchEvaluator()
        analysis = e.threshold_sweep(detections, str(gt))
        assert analysis.best is not None
        assert analysis.best.f1 == max(r.f1 for r in analysis.results)

    def test_empty_analysis_best_is_none(self):
        analysis = ThresholdAnalysis()
        assert analysis.best is None

class TestSaveResults:

    def test_saves_evaluation_result_to_json(self, tmp_path):
        output = tmp_path / "eval.json"
        result = EvaluationResult(
            precision=0.875, recall=0.875, f1=0.875,
            threshold=75.0, true_positives=7, false_positives=1,
            false_negatives=1, total_flagged=8, total_ground_truth=8,
            tolerance_seconds=2.0,
        )
        e = MismatchEvaluator()
        e.save_results(result, str(output))
        assert output.exists()
        with open(output) as f:
            data = json.load(f)
        assert data["f1"] == 0.875
        assert data["threshold"] == 75.0

    def test_saves_threshold_analysis_to_json(self, tmp_path):
        output = tmp_path / "sweep.json"
        analysis = ThresholdAnalysis(results=[
            EvaluationResult(
                precision=0.8, recall=0.9, f1=0.847,
                threshold=75.0, true_positives=9, false_positives=2,
                false_negatives=1, total_flagged=11, total_ground_truth=10,
                tolerance_seconds=2.0,
            )
        ])
        e = MismatchEvaluator()
        e.save_results(analysis, str(output))
        assert output.exists()
        with open(output) as f:
            data = json.load(f)
        assert "sweep" in data
        assert "best_threshold" in data