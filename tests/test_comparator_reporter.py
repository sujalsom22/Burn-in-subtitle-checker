"""
Tests for MismatchDetector, MismatchResult, and HTMLReporter.
"""

import os
import pytest
from src.transcriber import Segment
from src.ocr_extractor import SubtitleFrame
from src.comparator import MismatchDetector, MismatchResult, compare_segments
from src.reporter import HTMLReporter, generate_report

def make_segment(start=0.0, end=4.0, text="test"):
    return Segment(start=start, end=end, text=text)

def make_frame(timestamp=2.0, text="test", start=0.0, end=4.0):
    return SubtitleFrame(
        timestamp=timestamp,
        raw_text=text,
        normalised_text=text,
        segment_start=start,
        segment_end=end,
    )

class TestMismatchDetectorInit:

    def test_default_threshold(self):
        d = MismatchDetector()
        assert d.threshold == 75.0

    def test_custom_threshold(self):
        d = MismatchDetector(threshold=80.0)
        assert d.threshold == 80.0

    def test_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="Threshold"):
            MismatchDetector(threshold=-1.0)

    def test_threshold_above_100_raises(self):
        with pytest.raises(ValueError):
            MismatchDetector(threshold=101.0)

    def test_boundary_threshold_zero_accepted(self):
        d = MismatchDetector(threshold=0.0)
        assert d.threshold == 0.0

    def test_boundary_threshold_100_accepted(self):
        d = MismatchDetector(threshold=100.0)
        assert d.threshold == 100.0

class TestScoreComputation:

    def test_identical_text_scores_100(self):
        d = MismatchDetector()
        fuzzy, seq, final = d._compute_scores("hello world", "hello world")
        assert fuzzy == 100.0
        assert final == 100.0

    def test_both_empty_scores_100(self):
        d = MismatchDetector()
        fuzzy, seq, final = d._compute_scores("", "")
        assert final == 100.0

    def test_one_empty_scores_0(self):
        d = MismatchDetector()
        _, _, final = d._compute_scores("hello", "")
        assert final == 0.0

    def test_other_empty_scores_0(self):
        d = MismatchDetector()
        _, _, final = d._compute_scores("", "hello")
        assert final == 0.0

    def test_completely_different_text_scores_low(self):
        d = MismatchDetector()
        _, _, final = d._compute_scores("वो कहाँ गई थी", "ठीक है भाई")
        assert final < 50.0

    def test_identical_hindi_text_scores_100(self):
        d = MismatchDetector()
        text = "वो कहाँ गई थी"
        _, _, final = d._compute_scores(text, text)
        assert final == 100.0

    def test_similar_hindi_text_scores_high(self):
        d = MismatchDetector()
        _, _, final = d._compute_scores("वो कहाँ गई थी", "वो कहाँ गया था")
        assert final > 50.0

    def test_word_order_variation_handled(self):
        d = MismatchDetector()
        score_same_order, _, final_same = d._compute_scores("a b c", "a b c")
        _, _, final_diff_order = d._compute_scores("a b c", "c b a")
        assert final_same == 100.0
        assert final_diff_order > 80.0

    def test_final_score_is_weighted_average(self):
        d = MismatchDetector()
        fuzzy, seq, final = d._compute_scores("hello", "hello")
        expected = d.FUZZY_WEIGHT * fuzzy + d.SEQUENCE_WEIGHT * seq
        assert abs(final - expected) < 0.01

class TestCompare:

    def test_mismatched_lengths_raises(self):
        d = MismatchDetector()
        segments = [make_segment()]
        frames = [make_frame(), make_frame()]
        with pytest.raises(ValueError, match="equal length"):
            d.compare(segments, frames)

    def test_returns_one_result_per_segment(self):
        d = MismatchDetector()
        segments = [make_segment(text="hello"), make_segment(text="world")]
        frames = [make_frame(text="hello"), make_frame(text="world")]
        results = d.compare(segments, frames)
        assert len(results) == 2

    def test_identical_text_not_flagged(self):
        d = MismatchDetector(threshold=75.0)
        segments = [make_segment(text="ठीक है भाई")]
        frames = [make_frame(text="ठीक है भाई")]
        results = d.compare(segments, frames)
        assert results[0].flagged is False
        assert results[0].status == "OK"

    def test_different_text_flagged(self):
        d = MismatchDetector(threshold=75.0)
        segments = [make_segment(text="वो कहाँ गई थी")]
        frames = [make_frame(text="ठीक है भाई")]
        results = d.compare(segments, frames)
        assert results[0].flagged is True
        assert results[0].status == "REVIEW"

    def test_empty_segments_returns_empty(self):
        d = MismatchDetector()
        results = d.compare([], [])
        assert results == []

    def test_result_contains_correct_texts(self):
        d = MismatchDetector()
        segments = [make_segment(text="audio text")]
        frames = [make_frame(text="subtitle text")]
        results = d.compare(segments, frames)
        assert results[0].audio_text == "audio text"
        assert results[0].subtitle_text == "subtitle text"

    def test_result_timestamp_matches_frame(self):
        d = MismatchDetector()
        segments = [make_segment(start=10.0, end=14.0, text="test")]
        frames = [make_frame(timestamp=12.0, text="test")]
        results = d.compare(segments, frames)
        assert results[0].timestamp == 12.0

    def test_high_threshold_flags_more(self):
        d_strict = MismatchDetector(threshold=95.0)
        d_lenient = MismatchDetector(threshold=50.0)
        segments = [make_segment(text="वो कहाँ गई थी")]
        frames = [make_frame(text="वो कहाँ गया था")]
        strict_results = d_strict.compare(segments, frames)
        lenient_results = d_lenient.compare(segments, frames)
        strict_flagged = sum(1 for r in strict_results if r.flagged)
        lenient_flagged = sum(1 for r in lenient_results if r.flagged)
        assert strict_flagged >= lenient_flagged

class TestMismatchResult:

    def test_status_review_when_flagged(self):
        r = MismatchResult(
            timestamp=10.0, audio_text="a", subtitle_text="b",
            fuzzy_score=20.0, sequence_score=20.0, final_score=20.0,
            flagged=True, segment_start=9.0, segment_end=11.0,
        )
        assert r.status == "REVIEW"

    def test_status_ok_when_not_flagged(self):
        r = MismatchResult(
            timestamp=10.0, audio_text="a", subtitle_text="a",
            fuzzy_score=100.0, sequence_score=100.0, final_score=100.0,
            flagged=False, segment_start=9.0, segment_end=11.0,
        )
        assert r.status == "OK"

class TestHTMLReporter:

    def _make_result(self, flagged=False, score=95.0,
                     audio="ठीक है भाई", subtitle="ठीक है भाई"):
        return MismatchResult(
            timestamp=10.0,
            audio_text=audio,
            subtitle_text=subtitle,
            fuzzy_score=score,
            sequence_score=score,
            final_score=score,
            flagged=flagged,
            segment_start=9.0,
            segment_end=11.0,
        )

    def test_generates_html_file(self, tmp_path):
        reporter = HTMLReporter()
        output = tmp_path / "report.html"
        result = reporter.generate([], str(output))
        assert os.path.exists(result)

    def test_html_contains_table_headers(self, tmp_path):
        reporter = HTMLReporter()
        output = tmp_path / "report.html"
        reporter.generate([self._make_result()], str(output))
        with open(output, encoding="utf-8") as f:
            content = f.read()
        assert "Timestamp" in content
        assert "Audio Text" in content
        assert "Subtitle Text" in content
        assert "Match Score" in content

    def test_html_contains_hindi_text(self, tmp_path):
        reporter = HTMLReporter()
        output = tmp_path / "report.html"
        reporter.generate([self._make_result()], str(output))
        with open(output, encoding="utf-8") as f:
            content = f.read()
        assert "ठीक है भाई" in content

    def test_flagged_row_contains_review_badge(self, tmp_path):
        reporter = HTMLReporter()
        output = tmp_path / "report.html"
        reporter.generate(
            [self._make_result(flagged=True, score=20.0,
                               audio="वो गई थी", subtitle="वो गया था")],
            str(output)
        )
        with open(output, encoding="utf-8") as f:
            content = f.read()
        assert "REVIEW" in content

    def test_ok_row_contains_ok_badge(self, tmp_path):
        reporter = HTMLReporter()
        output = tmp_path / "report.html"
        reporter.generate([self._make_result(flagged=False)], str(output))
        with open(output, encoding="utf-8") as f:
            content = f.read()
        assert "OK" in content

    def test_empty_results_shows_no_results_message(self, tmp_path):
        reporter = HTMLReporter()
        output = tmp_path / "report.html"
        reporter.generate([], str(output))
        with open(output, encoding="utf-8") as f:
            content = f.read()
        assert "No segments" in content

    def test_html_escape_prevents_injection(self, tmp_path):
        reporter = HTMLReporter()
        output = tmp_path / "report.html"
        reporter.generate(
            [self._make_result(audio="<script>alert('xss')</script>")],
            str(output)
        )
        with open(output, encoding="utf-8") as f:
            content = f.read()
        assert "<script>" not in content
        assert "&lt;script&gt;" in content

    def test_score_formatted_as_0_to_1(self, tmp_path):
        reporter = HTMLReporter()
        output = tmp_path / "report.html"
        reporter.generate(
            [self._make_result(score=95.0)],
            str(output)
        )
        with open(output, encoding="utf-8") as f:
            content = f.read()
        assert "0.95" in content

    def test_video_name_appears_in_report(self, tmp_path):
        reporter = HTMLReporter()
        output = tmp_path / "report.html"
        reporter.generate([], str(output), video_name="sample_hindi.mp4")
        with open(output, encoding="utf-8") as f:
            content = f.read()
        assert "sample_hindi.mp4" in content