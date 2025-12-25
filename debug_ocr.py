from paddleocr import PaddleOCR
import cv2
import json
import sys

def debug_ocr(image_path):
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Using the same preprocessing as in ocr.py
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    result = ocr.predict(processed)
    
    # PaddleOCR result structure: list of dicts with 'rec_texts', 'rec_scores', 'rec_boxes'
    texts = result[0].get('rec_texts', [])
    scores = result[0].get('rec_scores', [])
    boxes = result[0].get('rec_boxes', [])

    print("\n--- RAW OCR RESULTS ---")
    for i, (text, score, box) in enumerate(zip(texts, scores, boxes)):
        print(f"[{i:2}] Score: {score:.4f} | Box: {box} | Text: '{text}'")

if __name__ == "__main__":
    img = 'images/ktp-1.jpg'
    if len(sys.argv) > 1:
        img = sys.argv[1]
    debug_ocr(img)
