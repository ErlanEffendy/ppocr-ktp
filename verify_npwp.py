from ocr import NPWPExtractor

try:
    extractor = NPWPExtractor()
    print("NPWPExtractor initialized successfully.")
    print(f"Validation rules: {extractor.validation_rules.keys()}")
except Exception as e:
    print(f"Failed to initialize NPWPExtractor: {e}")
