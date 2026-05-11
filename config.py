import os

WHISPER_MODEL_SIZE = "small"

OCR_LANGUAGES = "hin+kan+eng"

TESSERACT_PSM = "6"

SUBTITLE_REGION_START = 0.85

SIMILARITY_THRESHOLD = 75.0

OUTPUT_REPORT_NAME = "mismatch_report.html"

TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
