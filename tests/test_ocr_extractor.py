"""
Tests for SubtitleExtractor and SubtitleFrame.
Mocks OpenCV and Tesseract — keeps tests fast and CI-friendly.
"""

import pytest
import unicodedata
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

from src.transcriber import Segment
from src.ocr_extractor import SubtitleExtractor, SubtitleFrame, extract_subtitles

def make_segment(start=0.0, end=4.0, text="test"):
    return Segment(start=start, end=end, text=text)

def make_black_frame(h=480, w=640):
    """Create a dummy black BGR frame for testing."""
    return np.zeros((h, w, 3), dtype=np.uint8)

def make_white_frame(h=480, w=640):
    """Create a dummy white BGR frame for testing."""
    return np.ones((h, w, 3), dtype=np.uint8) * 255

class TestSubtitleFrame:

    def test_is_empty_true_for_blank_text(self):
        frame = SubtitleFrame(
            timestamp=1.0,
            raw_text="",
            normalised_text="",
            segment_start=0.0,
            segment_end=2.0,
        )
        assert frame.is_empty is True

    def test_is_empty_true_for_whitespace(self):
        frame = SubtitleFrame(
            timestamp=1.0,
            raw_text="   ",
            normalised_text="   ",
            segment_start=0.0,
            segment_end=2.0,
        )
        assert frame.is_empty is True

    def test_is_empty_false_for_text(self):
        frame = SubtitleFrame(
            timestamp=1.0,
            raw_text="hello",
            normalised_text="hello",
            segment_start=0.0,
            segment_end=2.0,
        )
        assert frame.is_empty is False

    def test_hindi_text_stored_correctly(self):
        hindi = "वो कहाँ गई थी"
        frame = SubtitleFrame(
            timestamp=10.2,
            raw_text=hindi,
            normalised_text=hindi,
            segment_start=9.0,
            segment_end=11.4,
        )
        assert frame.normalised_text == hindi

    def test_kannada_text_stored_correctly(self):
        kannada = "ಅವಳು ಎಲ್ಲಿ ಹೋದಳು"
        frame = SubtitleFrame(
            timestamp=5.0,
            raw_text=kannada,
            normalised_text=kannada,
            segment_start=4.0,
            segment_end=6.0,
        )
        assert frame.normalised_text == kannada

    def test_repr_contains_timestamp_and_text(self):
        frame = SubtitleFrame(
            timestamp=15.5,
            raw_text="test",
            normalised_text="test",
            segment_start=14.0,
            segment_end=17.0,
        )
        r = repr(frame)
        assert "15.5" in r
        assert "test" in r

class TestSubtitleExtractorInit:

    def test_default_values_set(self):
        ext = SubtitleExtractor()
        assert ext.languages == "hin+kan+eng"
        assert ext.subtitle_region_start == 0.85
        assert ext.psm_mode == "6"

    def test_custom_values_accepted(self):
        ext = SubtitleExtractor(
            languages="eng",
            subtitle_region_start=0.9,
            psm_mode="7",
        )
        assert ext.languages == "eng"
        assert ext.subtitle_region_start == 0.9
        assert ext.psm_mode == "7"

    def test_invalid_subtitle_region_raises(self):
        with pytest.raises(ValueError, match="subtitle_region_start"):
            SubtitleExtractor(subtitle_region_start=1.5)

    def test_zero_subtitle_region_raises(self):
        with pytest.raises(ValueError):
            SubtitleExtractor(subtitle_region_start=0.0)

class TestPreprocessing:

    def test_crop_subtitle_region_correct_size(self):
        ext = SubtitleExtractor(subtitle_region_start=0.85)
        frame = make_black_frame(h=480, w=640)
        crop = ext._crop_subtitle_region(frame)
        expected_height = 480 - int(480 * 0.85)  # bottom 15%
        assert crop.shape[0] == expected_height
        assert crop.shape[1] == 640

    def test_crop_full_width_preserved(self):
        ext = SubtitleExtractor()
        frame = make_white_frame(h=720, w=1280)
        crop = ext._crop_subtitle_region(frame)
        assert crop.shape[1] == 1280

    def test_preprocess_returns_2d_array(self):
        """Preprocessed image should be grayscale (2D), not BGR (3D)."""
        ext = SubtitleExtractor()
        crop = make_black_frame(h=72, w=640)
        processed = ext._preprocess(crop)
        assert processed.ndim == 2

    def test_preprocess_upscales_image(self):
        """Preprocessed image should be 2x the input size."""
        ext = SubtitleExtractor()
        crop = make_black_frame(h=72, w=640)
        processed = ext._preprocess(crop)
        assert processed.shape[0] == 72 * 2
        assert processed.shape[1] == 640 * 2

class TestNormalisation:

    def test_nfc_normalisation_applied(self):
        """
        Decomposed Devanagari should be normalised to composed NFC form.
        This is the core fix for Tesseract's Indic output inconsistency.
        """
        ext = SubtitleExtractor()
        decomposed = unicodedata.normalize("NFD", "क़")
        normalised = ext._normalise(decomposed)
        assert unicodedata.is_normalized("NFC", normalised)

    def test_normalise_strips_whitespace(self):
        ext = SubtitleExtractor()
        result = ext._normalise("  hello  ")
        assert result == "hello"

    def test_normalise_empty_string(self):
        ext = SubtitleExtractor()
        result = ext._normalise("")
        assert result == ""

    def test_normalise_hindi_text(self):
        ext = SubtitleExtractor()
        hindi = "वो कहाँ गई थी"
        result = ext._normalise(hindi)
        assert unicodedata.is_normalized("NFC", result)

class TestExtract:

    def test_file_not_found_raises(self):
        ext = SubtitleExtractor()
        segments = [make_segment()]
        with pytest.raises(FileNotFoundError):
            ext.extract("nonexistent_video.mp4", segments)

    @patch("src.ocr_extractor.cv2.VideoCapture")
    @patch("src.ocr_extractor.pytesseract.image_to_string")
    def test_returns_one_frame_per_segment(
        self, mock_ocr, mock_cap_class, tmp_path
    ):
        dummy_video = tmp_path / "test.mp4"
        dummy_video.write_bytes(b"fake")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, make_black_frame())
        mock_cap_class.return_value = mock_cap

        mock_ocr.return_value = "ठीक है भाई"

        segments = [
            make_segment(0.0, 3.0, "ठीक है भाई"),
            make_segment(4.0, 7.0, "वो कहाँ गई थी"),
        ]

        ext = SubtitleExtractor()
        frames = ext.extract(str(dummy_video), segments)

        assert len(frames) == 2

    @patch("src.ocr_extractor.cv2.VideoCapture")
    @patch("src.ocr_extractor.pytesseract.image_to_string")
    def test_timestamps_match_segment_midpoints(
        self, mock_ocr, mock_cap_class, tmp_path
    ):
        dummy_video = tmp_path / "test.mp4"
        dummy_video.write_bytes(b"fake")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, make_black_frame())
        mock_cap_class.return_value = mock_cap
        mock_ocr.return_value = "test"

        seg = make_segment(start=10.0, end=14.0)
        ext = SubtitleExtractor()
        frames = ext.extract(str(dummy_video), [seg])

        assert frames[0].timestamp == 12.0  # midpoint of 10.0 and 14.0

    @patch("src.ocr_extractor.cv2.VideoCapture")
    @patch("src.ocr_extractor.pytesseract.image_to_string")
    def test_empty_segments_list_returns_empty(
        self, mock_ocr, mock_cap_class, tmp_path
    ):
        dummy_video = tmp_path / "test.mp4"
        dummy_video.write_bytes(b"fake")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap_class.return_value = mock_cap

        ext = SubtitleExtractor()
        frames = ext.extract(str(dummy_video), [])

        assert frames == []

    @patch("src.ocr_extractor.cv2.VideoCapture")
    def test_cannot_open_video_raises_runtime_error(
        self, mock_cap_class, tmp_path
    ):
        dummy_video = tmp_path / "test.mp4"
        dummy_video.write_bytes(b"fake")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_class.return_value = mock_cap

        ext = SubtitleExtractor()
        with pytest.raises(RuntimeError, match="could not open video"):
            ext.extract(str(dummy_video), [make_segment()])

    @patch("src.ocr_extractor.cv2.VideoCapture")
    @patch("src.ocr_extractor.pytesseract.image_to_string")
    def test_failed_frame_capture_appends_empty_frame(
        self, mock_ocr, mock_cap_class, tmp_path
    ):
        dummy_video = tmp_path / "test.mp4"
        dummy_video.write_bytes(b"fake")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)  # frame capture fails
        mock_cap_class.return_value = mock_cap

        ext = SubtitleExtractor()
        frames = ext.extract(str(dummy_video), [make_segment()])

        assert len(frames) == 1
        assert frames[0].is_empty is True

    @patch("src.ocr_extractor.cv2.VideoCapture")
    @patch("src.ocr_extractor.pytesseract.image_to_string")
    def test_ocr_text_is_normalised(
        self, mock_ocr, mock_cap_class, tmp_path
    ):
        dummy_video = tmp_path / "test.mp4"
        dummy_video.write_bytes(b"fake")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, make_black_frame())
        mock_cap_class.return_value = mock_cap

        mock_ocr.return_value = "  वो कहाँ  "

        ext = SubtitleExtractor()
        frames = ext.extract(str(dummy_video), [make_segment()])

        assert frames[0].normalised_text == "वो कहाँ"
        assert unicodedata.is_normalized("NFC", frames[0].normalised_text)