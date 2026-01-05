from fastapi import FastAPI, File, UploadFile, HTTPException
from ocr import KTPExtractor, NPWPExtractor
import os
import uuid
import shutil

app = FastAPI(title="KTP & NPWP OCR API")
ktp_extractor = KTPExtractor()
npwp_extractor = NPWPExtractor()

@app.post("/extract")
async def extract_ktp(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")
    
    # Store the uploaded file in uploaded_files/images/
    upload_dir = os.path.join("uploaded_files", "images")
    os.makedirs(upload_dir, exist_ok=True)
    filename = f"{uuid.uuid4()}_{os.path.basename(file.filename)}"
    file_path = os.path.join(upload_dir, filename)
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract fields
        result = ktp_extractor.extract(file_path)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Keep the files in uploaded_files/images/
        pass

@app.post("/extract-npwp")
async def extract_npwp(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")
    
    # Store the uploaded file in uploaded_files/images/
    upload_dir = os.path.join("uploaded_files", "images")
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
        # Keep the files in uploaded_files/images/
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
