# Basic test script
from paddleocr import PaddleOCR
import cv2
import json

# Initialize OCR
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang='en'
)

# Test on a sample KTP image
img_path = 'ktp-erlan.jpg'
result = ocr.predict(img_path)

# Print results
res = result[0]
for text, score, bbox in zip(res['rec_texts'], res['rec_scores'], res['rec_boxes']):
    print(f"Text: {text} | Confidence: {score:.2f}")
    print(f"BBox: {bbox}\n")