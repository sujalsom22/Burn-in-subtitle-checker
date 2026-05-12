import os
import logging
from dataclasses import dataclass
from typing import List, Optional

import whisper
from tqdm import tqdm

from config import WHISPER_MODEL_SIZE

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """
    One transcribed audio segment with timing information.

    Attributes:
        start:  Start time in seconds.
        end:    End time in seconds.
        text:   Transcribed spoken text (may be Hindi, Kannada, or English).
    """
    start: float
    end: float
    text: str

    @property
    def midpoint(self) -> float:
        """
        Midpoint timestamp between start and end.
        Used by SubtitleExtractor to capture the frame most likely
        to show the subtitle for this audio segment.
        """
        return round((self.start + self.end) / 2, 3)

    @property
    def duration(self) -> float:
        """Duration of this segment in seconds."""
        return round(self.end - self.start, 3)

    def __repr__(self) -> str:
        return (
            f"Segment(start={self.start}s, end={self.end}s, "
            f"midpoint={self.midpoint}s, text={self.text!r})"
        )


class AudioTranscriber:
    """
    Transcribes the audio track of a video file using OpenAI Whisper.

    Usage:
        transcriber = AudioTranscriber(model_size="small")
        segments = transcriber.transcribe("video.mp4")
        for seg in segments:
            print(seg.midpoint, seg.text)
    """

    VALID_MODEL_SIZES = {"tiny", "base", "small", "medium", "large"}

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        language: Optional[str] = None,
    ):
        """
        Args:
            model_size: Whisper model size. Use 'small' for Indic languages
                        (better accuracy than 'tiny' for Hindi/Kannada).
                        'medium' is best but slow on CPU.
            language:   ISO language code to force ('hi' for Hindi,
                        'kn' for Kannada). None = auto-detect.
        """
        if model_size not in self.VALID_MODEL_SIZES:
            raise ValueError(
                f"Invalid model_size '{model_size}'. "
                f"Choose from: {self.VALID_MODEL_SIZES}"
            )

        self.model_size = model_size
        self.language = language
        self._model = None  

    def _load_model(self):
        """Load Whisper model lazily on first use."""
        if self._model is None:
            logger.info(f"Loading Whisper '{self.model_size}' model...")
            self._model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded.")
        return self._model

    def transcribe(self, video_path: str) -> List[Segment]:
        """
        Transcribe the audio track of a video file.

        Args:
            video_path: Path to the video file (.mp4, .avi, .mkv, etc.)

        Returns:
            List of Segment objects sorted by start time.

        Raises:
            FileNotFoundError: If video_path does not exist.
            RuntimeError:      If Whisper transcription fails.
        """
       
        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Video file not found: '{video_path}'"
            )

        logger.info(f"Transcribing: {video_path}")

        model = self._load_model()

        try:
            result = model.transcribe(
                video_path,
                language=self.language,
                fp16=False,
                verbose=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"Whisper transcription failed for '{video_path}': {e}"
            ) from e

        raw_segments = result.get("segments", [])

        if not raw_segments:
            logger.warning("Whisper returned no segments. "
                         "Check if the video has an audio track.")
            return []

        segments = []
        for seg in tqdm(raw_segments, desc="Parsing segments", unit="seg"):
            text = seg.get("text", "").strip()

            if not text:
                continue

            segments.append(Segment(
                start=round(float(seg["start"]), 3),
                end=round(float(seg["end"]), 3),
                text=text,
            ))

        logger.info(f"Transcription complete: {len(segments)} segments found.")
        return segments


def transcribe_video(
    video_path: str,
    model_size: str = WHISPER_MODEL_SIZE,
    language: Optional[str] = None,
) -> List[Segment]:
    """
    Convenience function — transcribe a video without instantiating the class.

    Args:
        video_path:  Path to video file.
        model_size:  Whisper model size (default from config).
        language:    Force language ('hi', 'kn', None=auto).

    Returns:
        List of Segment objects.
    """
    transcriber = AudioTranscriber(model_size=model_size, language=language)
    return transcriber.transcribe(video_path)