import sys

print(f"Python: {sys.version}")

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    langs = pytesseract.get_languages()
    print(f"Tesseract languages: {langs}")
    assert 'hin' in langs, "Hindi pack MISSING"
    assert 'kan' in langs, "Kannada pack MISSING"
    print("✓ pytesseract + Indic languages OK")
except Exception as e:
    print(f"✗ pytesseract FAILED: {e}")

try:
    import cv2
    print(f"✓ OpenCV OK — version {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV FAILED: {e}")

try:
    import whisper
    print("✓ Whisper OK — loading tiny model to verify...")
    model = whisper.load_model("tiny")
    print("✓ Whisper tiny model loaded OK")
except Exception as e:
    print(f"✗ Whisper FAILED: {e}")

try:
    from rapidfuzz import fuzz
    score = fuzz.token_sort_ratio("hello world", "world hello")
    print(f"✓ rapidfuzz OK — test score: {score}")
except Exception as e:
    print(f"✗ rapidfuzz FAILED: {e}")

try:
    import moviepy
    print("✓ moviepy OK")
except Exception as e:
    print(f"✗ moviepy FAILED: {e}")

print("\nAll checks complete.")