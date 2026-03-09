# KTP OCR Extraction

This project provides tools for robust data extraction from Indonesian Identity Cards (KTP) using PaddleOCR and YOLOv11.

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install paddleocr opencv-python numpy requests fastapi uvicorn ultralytics python-multipart
```

Additionally, ensure your directory structure is set up properly with the required model:
- You must have the YOLO segmentation model located at `seg_model/yolo11/best.pt`. (Used for advanced KTP perspective correction).

## 1. Running the Core OCR Script (`ocr.py`)

If you want to run the extractor directly, you can execute `ocr.py` from the terminal. By default, it processes an example image located at `images/original.jpg`.

```bash
python ocr.py
```

### Usage in Python
You can import the `KTPExtractor` from `ocr.py` into any of your own scripts:

```python
import json
from ocr import KTPExtractor

# 1. Initialize the extractor (loads the models)
extractor = KTPExtractor()

# 2. Extract data from an image
result = extractor.extract("path/to/your/ktp_image.jpg")

# 3. Print the result
print(json.dumps(result, indent=2))
```

## 2. Running the REST API (`api.py`)

The project includes a server setup using FastAPI to expose the OCR processing logic as a web endpoint.

**Start the server:**
```bash
python api.py
```
*(The server will start at `http://0.0.0.0:8001` or `http://localhost:8001`)*

### API Usage
- **Endpoint:** `POST /extract`
- **Payload:** `multipart/form-data` with a single `file` field containing the image.

**Example Request using `curl`:**
```bash
curl -X POST "http://localhost:8001/extract" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@images/original.jpg;type=image/jpeg"
```

## 3. Running the Debugging Script (`debug_ocr.py`)

To debug and view the raw bounding boxes and text blocks that are being identified by PaddleOCR before any parsing/cleaning is done, use `debug_ocr.py`:

```bash
python debug_ocr.py <path_to_your_image>
```

**Example:**
```bash
python debug_ocr.py images/ktp-1.jpg
```
*(If no image is provided, it defaults to `images/ktp-1.jpg`)*
