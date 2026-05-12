import pytest
from unittest.mock import patch, MagicMock
from src.transcriber import AudioTranscriber, Segment, transcribe_video

class TestSegment:

    def test_midpoint_calculated_correctly(self):
        seg = Segment(start=10.0, end=20.0, text="hello")
        assert seg.midpoint == 15.0

    def test_midpoint_odd_numbers(self):
        seg = Segment(start=10.0, end=11.0, text="test")
        assert seg.midpoint == 10.5

    def test_duration_calculated_correctly(self):
        seg = Segment(start=5.0, end=8.0, text="hello")
        assert seg.duration == 3.0

    def test_hindi_text_stored_correctly(self):
        hindi_text = "वो कहाँ गई थी"
        seg = Segment(start=0.0, end=5.0, text=hindi_text)
        assert seg.text == hindi_text

    def test_kannada_text_stored_correctly(self):
        kannada_text = "ಅವಳು ಎಲ್ಲಿ ಹೋದಳು"
        seg = Segment(start=0.0, end=5.0, text=kannada_text)
        assert seg.text == kannada_text

    def test_repr_contains_key_info(self):
        seg = Segment(start=1.0, end=3.0, text="test")
        r = repr(seg)
        assert "1.0" in r
        assert "3.0" in r
        assert "test" in r

class TestAudioTranscriberInit:

    def test_valid_model_size_accepted(self):
        t = AudioTranscriber(model_size="tiny")
        assert t.model_size == "tiny"

    def test_invalid_model_size_raises(self):
        with pytest.raises(ValueError, match="Invalid model_size"):
            AudioTranscriber(model_size="ultrafast")

    def test_language_stored(self):
        t = AudioTranscriber(language="hi")
        assert t.language == "hi"

    def test_model_not_loaded_on_init(self):
        t = AudioTranscriber()
        assert t._model is None

class TestTranscribe:

    def test_file_not_found_raises(self):
        t = AudioTranscriber()
        with pytest.raises(FileNotFoundError):
            t.transcribe("nonexistent_video.mp4")

    @patch("src.transcriber.whisper.load_model")
    def test_returns_segments_from_whisper_output(self, mock_load, tmp_path):
        dummy_video = tmp_path / "test.mp4"
        dummy_video.write_bytes(b"fake video content")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 3.5, "text": " वो कहाँ गई थी"},
                {"start": 4.0, "end": 7.0, "text": " ठीक है भाई"},
            ]
        }
        mock_load.return_value = mock_model

        t = AudioTranscriber(model_size="tiny")
        segments = t.transcribe(str(dummy_video))

        assert len(segments) == 2
        assert segments[0].text == "वो कहाँ गई थी"  # stripped
        assert segments[0].start == 0.0
        assert segments[0].midpoint == 1.75

    @patch("src.transcriber.whisper.load_model")
    def test_empty_segments_filtered_out(self, mock_load, tmp_path):
        dummy_video = tmp_path / "test.mp4"
        dummy_video.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "  "},  # whitespace only
                {"start": 2.0, "end": 4.0, "text": "valid"},
            ]
        }
        mock_load.return_value = mock_model

        t = AudioTranscriber(model_size="tiny")
        segments = t.transcribe(str(dummy_video))

        assert len(segments) == 1
        assert segments[0].text == "valid"

    @patch("src.transcriber.whisper.load_model")
    def test_no_segments_returns_empty_list(self, mock_load, tmp_path):
        dummy_video = tmp_path / "test.mp4"
        dummy_video.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": []}
        mock_load.return_value = mock_model

        t = AudioTranscriber(model_size="tiny")
        segments = t.transcribe(str(dummy_video))
        assert segments == []

    @patch("src.transcriber.whisper.load_model")
    def test_whisper_error_raises_runtime_error(self, mock_load, tmp_path):
        dummy_video = tmp_path / "test.mp4"
        dummy_video.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("GPU out of memory")
        mock_load.return_value = mock_model

        t = AudioTranscriber(model_size="tiny")
        with pytest.raises(RuntimeError, match="Whisper transcription failed"):
            t.transcribe(str(dummy_video))

    @patch("src.transcriber.whisper.load_model")
    def test_model_lazy_loaded_once(self, mock_load, tmp_path):
        dummy_video = tmp_path / "test.mp4"
        dummy_video.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": []}
        mock_load.return_value = mock_model

        t = AudioTranscriber(model_size="tiny")
        t.transcribe(str(dummy_video))
        t.transcribe(str(dummy_video))

        mock_load.assert_called_once()