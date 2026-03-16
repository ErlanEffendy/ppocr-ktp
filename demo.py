import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import re
import pandas as pd
import os

# --- 1. MODEL LOADING (GLOBAL) ---
# This part runs only once when the Gradio app starts, making it efficient.
print("Initializing application...")
MODEL_PATH = "emisilab/model-ocr-ktp-v1"

# Set device to GPU (cuda) if available, otherwise fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the processor and the OCR model from Hugging Face
print(f"Loading model from {MODEL_PATH}...")
try:
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH).to(device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    processor = None

# --- 2. CORE LOGIC: THE EXTRACTION FUNCTION ---
def extract_ktp_data(image_files):
    """
    Processes a list of uploaded image files, performs OCR, and extracts structured data.
    
    Args:
        image_files (list): A list of file path objects from the Gradio File input.
    Returns:
        pandas.DataFrame: A DataFrame containing the extracted data for each image.
    """
    if not image_files:
        print("No image files provided.")
        return pd.DataFrame(columns=['Filename', 'NIK', 'Nama', 'Tempat Lahir', 'Tanggal Lahir'])
        
    if not model or not processor:
        raise gr.Error("Model could not be loaded. The application is not functional.")

    print(f"Processing {len(image_files)} image(s)...")
    all_results = []

    # Define the refined regex patterns for data extraction
    patterns = {
        "nik": r'\b\d{16}\b',
        "nama": r'(?<=\b\d{16}\b\s)(.*?)(?=\s+WNI)',
        "tempat_lahir": r'(?<=\d{2}-\d{2}-\d{4}\s)([A-Z]+)',
        "tanggal_lahir": r'\d{2}-\d{2}-\d{4}',
    }

    # Loop through each uploaded file
    for file_path in image_files:
        filename = os.path.basename(file_path)
        print(f"-> Processing: {filename}")
        
        try:
            image = Image.open(file_path).convert("RGB")
            
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values, max_length=1024)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            current_image_data = {"Filename": filename}
            
            for key, pattern in patterns.items():
                match = re.search(pattern, generated_text)
                if match:
                    if key in ['nama', 'tempat_lahir']:
                        current_image_data[key.replace('_', ' ').title()] = match.group(1).strip()
                    else:
                        current_image_data[key.replace('_', ' ').title()] = match.group(0).strip()
                else:
                    current_image_data[key.replace('_', ' ').title()] = None
            
            all_results.append(current_image_data)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            all_results.append({
                "Filename": filename, "NIK": f"Error: {e}", "Nama": None, 
                "Tempat Lahir": None, "Tanggal Lahir": None
            })

    results_df = pd.DataFrame(all_results)
    print("Processing complete.")
    return results_df

# --- 3. UI DEFINITION: THE GRADIO INTERFACE ---

app_description = f"""
# KTP (Indonesian ID Card) OCR Extractor 🇮🇩
This application extracts key information (**NIK, Nama, Tempat Lahir, Tanggal Lahir**) from Indonesian ID cards (KTP). 
You can upload one or multiple KTP images at once. The results will be displayed in a table below.
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(app_description)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.File(
                label="Upload KTP Images",
                file_count="multiple",
                file_types=["image"],
                type="filepath"
            )
            extract_button = gr.Button("Extract KTP Data", variant="primary")

        with gr.Column(scale=2):
            output_dataframe = gr.DataFrame(
                label="Extracted Information",
                headers=['Filename', 'NIK', 'Nama', 'Tempat Lahir', 'Tanggal Lahir']
            )

    extract_button.click(
        fn=extract_ktp_data,
        inputs=image_input,
        outputs=output_dataframe
    )

# --- 4. LAUNCH THE APP ---
if __name__ == "__main__":
    demo.launch()
