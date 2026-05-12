# Burn-in Subtitle Checker

An open-source Python tool that automatically detects mismatches between
spoken audio and burned-in subtitles in video files.

Built for PlanetRead as part of DMP 2026.

## What it does

- Transcribes audio using OpenAI Whisper (ASR)
- Extracts burned-in subtitle text from video frames using OCR
- Compares audio vs subtitle text using fuzzy + semantic similarity
- Generates an HTML report flagging timestamp-level mismatches for human review
- Supports Hindi (Devanagari) and Kannada scripts

## Installation

### 1. System dependencies

**Tesseract OCR** (required — not a pip package):

- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
  - During install, select Hindi and Kannada language packs
- Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-kan`
- Mac: `brew install tesseract tesseract-lang`

**ffmpeg** (required by Whisper):

- Windows: `winget install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`
- Mac: `brew install ffmpeg`

### 2. Python environment

```bash
git clone https://github.com/PlanetRead/Burn-in-subtitle-checker.git
cd Burn-in-subtitle-checker
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Windows: set Tesseract path

Set the environment variable before running:

```bash
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Usage

```bash
python -m src.cli --video path/to/video.mp4
```

Optional flags:
- `--model-size` : Whisper model size (tiny/small/medium, default: small)
- `--threshold`  : Similarity threshold 0-100 (default: 75)
- `--lang`       : OCR language string (default: hin+kan+eng)

## Project structure

```
src/
  transcriber.py    # Whisper ASR module
  ocr_extractor.py  # OpenCV + Tesseract OCR module
  comparator.py     # Similarity scoring module
  reporter.py       # HTML report generator
  evaluator.py      # Accuracy evaluation — precision, recall, F1 against ground truth
  cli.py            # CLI entry point
tests/
  # pytest test suite
samples/
  # Sample videos for testing
config.py           # Central configuration
```

## Expected output

| Timestamp | Audio Text | Subtitle Text | Match Score |
|-----------|------------|---------------|-------------|
| 10.2s | वो कहाँ गई थी | वो कहाँ गया था | 0.61 — REVIEW |
| 45.7s | ठीक है भाई | ठीक है भाई | 0.95 — OK |

## Tech stack

- Python 3.9+
- OpenAI Whisper (ASR)
- Tesseract OCR with Indic language packs
- OpenCV (frame extraction + preprocessing)
- rapidfuzz (fuzzy text similarity)
- difflib (sequence matching)

## Mentors

@abinash-sketch @keerthiseelan-planetread