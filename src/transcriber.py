from dataclasses import dataclass
from typing import List


@dataclass
class Segment:
    """Represents one transcribed audio segment with timing."""
    start: float    # start time in seconds
    end: float      # end time in seconds
    text: str       # transcribed text

    @property
    def midpoint(self) -> float:
        """Midpoint timestamp — used to capture subtitle frame."""
        return (self.start + self.end) / 2


class AudioTranscriber:
    """Transcribes audio track of a video file using Whisper ASR."""

    def __init__(self, model_size: str = "small"):
        # TODO: implement in next PR
        pass

    def transcribe(self, video_path: str) -> List[Segment]:
        # TODO: implement in next PR
        raise NotImplementedError