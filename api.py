from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ocr import KTPExtractor
from npwp_extractor import NPWPExtractor
import os
import uuid
import shutil
import tempfile

SAVE_UPLOADED_FILES = os.getenv("SAVE_UPLOADED_FILES", "false").lower() == "true"

app = FastAPI(title="KTP and NPWP OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
extractor = KTPExtractor()
npwp_extractor = NPWPExtractor()

@app.post("/extract")
async def extract_ktp(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")
    
    # Store the uploaded file in uploaded_files/images/ when persistence is enabled
    upload_dir = os.path.join("uploaded_files", "images") if SAVE_UPLOADED_FILES else tempfile.gettempdir()
    os.makedirs(upload_dir, exist_ok=True)
    filename = f"{uuid.uuid4()}_{os.path.basename(file.filename)}"
    file_path = os.path.join(upload_dir, filename)
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract fields
        result = extractor.extract(file_path)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if not SAVE_UPLOADED_FILES and os.path.exists(file_path):
            os.remove(file_path)

@app.post("/extract-npwp")
async def extract_npwp(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")
    
    # Store the uploaded file in uploaded_files/images/ when persistence is enabled
    upload_dir = os.path.join("uploaded_files", "images") if SAVE_UPLOADED_FILES else tempfile.gettempdir()
    os.makedirs(upload_dir, exist_ok=True)
    filename = f"{uuid.uuid4()}_{os.path.basename(file.filename)}"
    file_path = os.path.join(upload_dir, filename)
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract fields
        result = npwp_extractor.extract(file_path)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if not SAVE_UPLOADED_FILES and os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
