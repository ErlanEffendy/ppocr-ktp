from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import json
import time

load_dotenv()

class KTPExtractor:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("WARNING: GEMINI_API_KEY not found in environment variables.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)
            
        # Fallback models: User desired 2.5, but we fallback to 2.0 and 1.5 if 2.5 is rate limited or unavailable.
        self.models = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-3-flash"]
        print(f"KTPExtractor initialized with models: {self.models}")
        
        self.validation_rules = {
            'province': {'required': True},
            'city': {'required': True},
            'nik': {'pattern': r'^\d{16}$', 'required': True},
            'name': {'pattern': r'^[A-Z\s\.]{3,}$', 'required': True},
            'place_of_birth': {'required': True},
            'date_of_birth': {'required': True},
            'gender': {'values': ['LAKI-LAKI', 'PEREMPUAN', 'MALE', 'FEMALE'], 'required': True},
            'blood_type': {'values': ['A', 'B', 'AB', 'O', '-'], 'required': False},
            'address': {'required': True},
            'rt_rw': {'pattern': r'^\d{3}\s*/\s*\d{3}$', 'required': True},
            'village': {'required': True},
            'district': {'required': True},
            'religion': {'values': ['ISLAM', 'KRISTEN', 'KATOLIK', 'KATHOLIK', 'HINDU', 'BUDDHA', 'KHONGHUCU', 'PROTESTANT', 'CHRISTIAN'], 'required': False},
            'marital_status': {'values': ['BELUM KAWIN', 'KAWIN', 'CERAI HIDUP', 'CERAI MATI', 'SINGLE', 'MARRIED', 'DIVORCED', 'WIDOWED'], 'required': False},
            'occupation': {'required': True},
            'citizenship': {'required': True},
            'expiry_date': {'required': True}
        }

    def extract(self, image_path):
        """Extract all fields from KTP using Gemini with model fallback"""
        start_time = time.time()
        
        try:
            if not self.client:
                 raise ValueError("Gemini Client not initialized. Check API Key.")

            # Load image bytes
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            # Determine mime type
            mime_type = "image/jpeg"
            if image_path.lower().endswith(".png"):
                mime_type = "image/png"
            elif image_path.lower().endswith(".webp"):
                mime_type = "image/webp"

            prompt = """
            Extract the data from this Indonesian KTP (Identity Card) image. 
            Return a valid JSON object strictly matching this structure for the fields.
            
            Required JSON Structure:
            {
                "province": "PROVINSI ...",
                "city": "KOTA ... or KABUPATEN ...",
                "nik": "16 digit number",
                "name": "Full Name",
                "place_of_birth": "Place",
                "date_of_birth": "DD-MM-YYYY",
                "gender": "LAKI-LAKI or PEREMPUAN",
                "blood_type": "A/B/O/AB/-",
                "address": "Address line",
                "rt_rw": "000/000",
                "village": "Village/Desa",
                "district": "Kecamatan",
                "religion": "RELIGION",
                "marital_status": "MARITAL STATUS",
                "occupation": "OCCUPATION",
                "citizenship": "WNI/WNA",
                "expiry_date": "DD-MM-YYYY or SEUMUR HIDUP"
            }
            
            If a field is not visible or clear, return null or an empty string.
            """

            response = None
            last_error = None
            
            print(f"DEBUG: Available models: {self.models}")
            
            for i, model_name in enumerate(self.models):
                print(f"DEBUG: Attempting model {i+1}/{len(self.models)}: {model_name}")
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=[
                            types.Content(
                                parts=[
                                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                                    types.Part.from_text(text=prompt)
                                ]
                            )
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        )
                    )
                    print(f"DEBUG: Success with model: {model_name}")
                    break
                except Exception as e:
                    print(f"DEBUG: Model {model_name} failed with error type {type(e).__name__}: {e}")
                    last_error = e
                    # Explicitly continue, though it happens automatically
                    continue
                except BaseException as be:
                    print(f"DEBUG: Model {model_name} failed with CRITICAL error type {type(be).__name__}: {be}")
                    last_error = be
                    continue
            
            if response is None:
                raise last_error if last_error else Exception("All models failed to extract data.")
            
            text = response.text
             # basic cleanup if model includes markdown
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            data = json.loads(text)
            
            formatted_fields = {}
            for key in self.validation_rules.keys():
                val = data.get(key, "")
                if val is None: val = ""
                formatted_fields[key] = {
                    'value': str(val).upper(),
                    'confidence': 1.0, 
                    'bbox': []
                }

            validation = self._validate_fields(formatted_fields)
            total_time = time.time() - start_time
            
            return {
                'fields': formatted_fields,
                'validation': validation,
                'confidence_score': 1.0, 
                'performance': {
                    'total_time': total_time,
                    'preprocessing_time': 0,
                    'ocr_inference_time': total_time,
                    'extraction_time': 0,
                    'model_used': response.model_version if hasattr(response, 'model_version') else "unknown"
                }
            }
            
        except Exception as e:
            print(f"Error during Gemini extraction: {e}")
            raise e

    def _validate_fields(self, fields):
        """Reuse existing validation logic simple version"""
        validation = {}
        required_only = {k: v for k, v in self.validation_rules.items() if v.get('required', False)}
        
        for field_name, rules in required_only.items():
            if field_name not in fields:
                validation[field_name] = {'valid': False, 'error': 'Missing field'}
                continue
            
            value = fields[field_name].get('value', '')
            
            # Simple pattern validation
            if 'pattern' in rules:
                import re
                if not re.match(rules['pattern'], value):
                    validation[field_name] = {'valid': False, 'error': 'Pattern mismatch'}
                    continue
            
            if 'values' in rules:
                if value not in rules['values']:
                   validation[field_name] = {'valid': False, 'error': f'Invalid value: {value}'}
                   continue

            validation[field_name] = {'valid': True}
        return validation

    def _calculate_overall_confidence(self, fields):
        return 1.0

if __name__ == '__main__':
    # Test block
    extractor = KTPExtractor()
    # Replace with a real path if you want to test locally
    # image_path = 'path/to/test.jpg'
    # print(json.dumps(extractor.extract(image_path), indent=2))
