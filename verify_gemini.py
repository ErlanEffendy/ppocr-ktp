import os
import sys
from ocr import KTPExtractor
import json

def test_ocr():
    # Specific image to test
    image_path = os.path.join("images", "ktp.jpg")
    
    if not os.path.exists(image_path):
        # Fallback to checking other common paths if the specific one doesn't exist
        possible_images = [
            os.path.join("images", "ktp-1.jpg"),
            os.path.join("uploaded_files", "images", "ktp.jpg")
        ]
        found = False
        for p in possible_images:
            if os.path.exists(p):
                image_path = p
                found = True
                break
        
        if not found:
            print(f"Sample image not found at {image_path} or other common locations.")
            return

    print(f"Testing OCR with image: {image_path}")
    
    try:
        extractor = KTPExtractor()
        result = extractor.extract(image_path)
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        # Check if it was a rate limit error to give better feedback
        if "429" in str(e):
            print("\nRate limit encountered. The code supports fallback, so if you see this, all fallbacks failed or the code hasn't reloaded.")

if __name__ == "__main__":
    test_ocr()
