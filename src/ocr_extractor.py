import os
import unicodedata
import logging
from dataclasses import dataclass
from typing import List, Optional

import cv2
import pytesseract
import numpy as np

from config import (
    OCR_LANGUAGES,
    TESSERACT_PSM,
    SUBTITLE_REGION_START,
    TESSERACT_CMD,
)
from src.transcriber import Segment

logger = logging.getLogger(__name__)

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


@dataclass
class SubtitleFrame:
    """
    OCR result for one video frame corresponding to an audio segment.

    Attributes:
        timestamp:       Midpoint timestamp (seconds) at which frame was captured.
        raw_text:        Raw text returned by Tesseract before normalisation.
        normalised_text: Unicode NFC normalised text — resolves composed vs
                         decomposed Devanagari character inconsistencies.
        segment_start:   Start time of the corresponding audio segment.
        segment_end:     End time of the corresponding audio segment.
    """
    timestamp: float
    raw_text: str
    normalised_text: str
    segment_start: float
    segment_end: float

    @property
    def is_empty(self) -> bool:
        """True if no subtitle text was extracted from the frame."""
        return len(self.normalised_text.strip()) == 0

    def __repr__(self) -> str:
        return (
            f"SubtitleFrame(timestamp={self.timestamp}s, "
            f"text={self.normalised_text!r})"
        )


class SubtitleExtractor:
    """
    Extracts burned-in subtitle text from video frames using OCR.

    For each Segment from AudioTranscriber, captures the frame at the
    segment midpoint, crops the subtitle region, applies adaptive
    preprocessing, and runs Tesseract with Indic language support.

    Usage:
        extractor = SubtitleExtractor()
        frames = extractor.extract("video.mp4", segments)
        for frame in frames:
            print(frame.timestamp, frame.normalised_text)
    """

    def __init__(
        self,
        languages: str = OCR_LANGUAGES,
        subtitle_region_start: float = SUBTITLE_REGION_START,
        psm_mode: str = TESSERACT_PSM,
    ):
        """
        Args:
            languages:             Tesseract language string (e.g. 'hin+kan+eng').
            subtitle_region_start: Fraction of frame height where subtitle region
                                   begins. 0.85 = bottom 15% of frame.
            psm_mode:              Tesseract PSM mode. Mode 6 = uniform block of
                                   text, optimal for subtitle regions.
        """
        if not 0.0 < subtitle_region_start < 1.0:
            raise ValueError(
                f"subtitle_region_start must be between 0 and 1, "
                f"got {subtitle_region_start}"
            )

        self.languages = languages
        self.subtitle_region_start = subtitle_region_start
        self.psm_mode = psm_mode
        self._tesseract_config = f"--psm {psm_mode}"

    def extract(
        self,
        video_path: str,
        segments: List[Segment],
    ) -> List[SubtitleFrame]:
        """
        Extract subtitle text for each audio segment.

        Args:
            video_path: Path to the video file.
            segments:   List of Segment objects from AudioTranscriber.

        Returns:
            List of SubtitleFrame objects, one per segment.

        Raises:
            FileNotFoundError: If video_path does not exist.
            RuntimeError:      If video cannot be opened by OpenCV.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Video file not found: '{video_path}'"
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(
                f"OpenCV could not open video: '{video_path}'. "
                f"Check that the file is a valid video format."
            )

        subtitle_frames = []

        try:
            for segment in segments:
                frame = self._capture_frame(cap, segment.midpoint)

                if frame is None:
                    logger.warning(
                        f"Could not capture frame at {segment.midpoint}s "
                        f"— skipping segment [{segment.start}s → {segment.end}s]"
                    )
            
                    subtitle_frames.append(SubtitleFrame(
                        timestamp=segment.midpoint,
                        raw_text="",
                        normalised_text="",
                        segment_start=segment.start,
                        segment_end=segment.end,
                    ))
                    continue

                crop = self._crop_subtitle_region(frame)

                processed = self._preprocess(crop)

                raw_text = self._run_ocr(processed)

                normalised = self._normalise(raw_text)

                subtitle_frames.append(SubtitleFrame(
                    timestamp=segment.midpoint,
                    raw_text=raw_text,
                    normalised_text=normalised,
                    segment_start=segment.start,
                    segment_end=segment.end,
                ))

                logger.debug(
                    f"Frame at {segment.midpoint}s → OCR: {normalised!r}"
                )

        finally:
            cap.release()

        logger.info(
            f"OCR complete: {len(subtitle_frames)} frames processed, "
            f"{sum(1 for f in subtitle_frames if not f.is_empty)} non-empty."
        )
        return subtitle_frames

    def _capture_frame(
        self,
        cap: cv2.VideoCapture,
        timestamp_seconds: float,
    ) -> Optional[np.ndarray]:
        """
        Capture a single frame at the given timestamp.

        Args:
            cap:               OpenCV VideoCapture object (already opened).
            timestamp_seconds: Target timestamp in seconds.

        Returns:
            Frame as numpy array (BGR), or None if capture failed.
        """

        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
        ret, frame = cap.read()

        if not ret or frame is None:
            return None

        return frame

    def _crop_subtitle_region(self, frame: np.ndarray) -> np.ndarray:
        """
        Crop the subtitle region from the bottom of the frame.

        Default: bottom 15% (SUBTITLE_REGION_START = 0.85).
        This covers the standard subtitle placement in most video formats.

        Args:
            frame: Full video frame as numpy array.

        Returns:
            Cropped subtitle region.
        """
        h, w = frame.shape[:2]
        start_row = int(h * self.subtitle_region_start)
        crop = frame[start_row:, :]
        return crop

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess the subtitle crop for optimal Tesseract accuracy.

        Pipeline:
        1. Convert to grayscale
        2. Upscale 2x — improves Tesseract accuracy on small text
        3. Apply adaptive thresholding — handles variable backgrounds
           better than simple binary threshold (critical for Indian
           educational videos with dynamic backgrounds behind subtitles)
        4. Denoise — removes compression artifacts

        Args:
            crop: Subtitle region crop (BGR).

        Returns:
            Preprocessed grayscale image ready for Tesseract.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        upscaled = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )

        thresholded = cv2.adaptiveThreshold(
            upscaled,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=15,
            C=8,
        )

        denoised = cv2.fastNlMeansDenoising(
            thresholded,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21,
        )

        return denoised

    def _run_ocr(self, processed_image: np.ndarray) -> str:
        """
        Run Tesseract OCR on the preprocessed image.

        Args:
            processed_image: Preprocessed grayscale image.

        Returns:
            Raw OCR text string (may contain noise and artifacts).
        """
        try:
            text = pytesseract.image_to_string(
                processed_image,
                lang=self.languages,
                config=self._tesseract_config,
            )
            return text.strip()
        except pytesseract.TesseractError as e:
            logger.warning(f"Tesseract OCR failed: {e}")
            return ""

    def _normalise(self, text: str) -> str:
        """
        Apply Unicode NFC normalisation to OCR output.

        Critical for Devanagari text: Tesseract can return composed vs
        decomposed character forms for the same visual character.
        For example, 'क' + '़' (decomposed) vs 'क़' (composed NFC form)
        appear identical but have different byte representations, causing
        false mismatches in text comparison.

        NFC normalisation ensures consistent representation.

        Args:
            text: Raw OCR text.

        Returns:
            NFC-normalised, stripped text.
        """
        return unicodedata.normalize("NFC", text).strip()


def extract_subtitles(
    video_path: str,
    segments: List[Segment],
    languages: str = OCR_LANGUAGES,
    subtitle_region_start: float = SUBTITLE_REGION_START,
) -> List[SubtitleFrame]:
    """
    Convenience function — extract subtitles without instantiating the class.

    Args:
        video_path:            Path to video file.
        segments:              List of Segment objects from AudioTranscriber.
        languages:             Tesseract language string.
        subtitle_region_start: Subtitle region start fraction.

    Returns:
        List of SubtitleFrame objects.
    """
    extractor = SubtitleExtractor(
        languages=languages,
        subtitle_region_start=subtitle_region_start,
    )
    return extractor.extract(video_path, segments)