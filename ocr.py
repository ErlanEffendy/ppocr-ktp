from paddleocr import PaddleOCR
import cv2
import re
import json
import time

class KTPExtractor:
    def __init__(self):
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='en'
        )
        
        self.validation_rules = {
            'province': {'required': True},
            'city': {'required': True},
            'nik': {
                'pattern': r'^\d{16}$',
                'required': True
            },
            'name': {
                'pattern': r'^[A-Z\s\.]{3,}$',
                'required': True
            },
            'place_of_birth': {'required': True},
            'date_of_birth': {'required': True},
            'gender': {
                'values': ['LAKI-LAKI', 'PEREMPUAN'],
                'required': True
            },
            'blood_type': {'required': False},
            'address': {'required': True},
            'rt_rw': {'required': True},
            'village': {'required': True},
            'district': {'required': True},
            'religion': {
                'values': ['ISLAM', 'KRISTEN', 'KATHOLIK', 'HINDU', 'BUDDHA', 'KHONGHUCU'],
                'required': False
            },
            'marital_status': {
                'values': ['BELUM KAWIN', 'KAWIN', 'CERAI HIDUP', 'CERAI MATI'],
                'required': False
            },
            'occupation': {
              'values': [
                          "BELUM/TIDAK BEKERJA",
                          "MENGURUS RUMAH TANGGA",
                          "PELAJAR/MAHASISWA",
                          "PENSIUNAN",
                          "PEGAWAI NEGERI SIPIL (PNS)",
                          "TENTARA NASIONAL INDONESIA (TNI)",
                          "KEPOLISIAN RI (POLRI)",
                          "PERDAGANGAN",
                          "PETANI/PEKEBUN",
                          "PETERNAK",
                          "NELAYAN/PERIKANAN",
                          "INDUSTRI",
                          "KONSTRUKSI",
                          "TRANSPORTASI",
                          "KARYAWAN SWASTA",
                          "KARYAWAN BUMN",
                          "KARYAWAN BUMD",
                          "KARYAWAN HONORER",
                          "BURUH HARIAN LEPAS",
                          "BURUH TANI/PERKEBUNAN",
                          "BURUH NELAYAN/PERIKANAN",
                          "BURUH PETERNAKAN",
                          "AKUNTAN/KONSULTAN KEUANGAN",
                          "DOKTER",
                          "TENAGA MEDIS LAINNYA",
                          "PERAWAT",
                          "BIDAN",
                          "APOTEKER",
                          "PENGACARA/ADVOKAT",
                          "NOTARIS",
                          "ARSITEK",
                          "SENIMAN",
                          "WARTAWAN",
                          "GURU",
                          "DOSEN",
                          "ROHANIWAN",
                          "PARANORMAL",
                          "PERAJIN",
                          "SOPIR",
                          "USAHA MIKRO, KECIL, DAN MENENGAH (UMKM)",
                          "JASA PERSEWAAN PERALATAN PESTA",
                          "LAINNYA"
                        ],
              'required': True
            },
            'citizenship': {'required': True},
            'expiry_date': {'required': True}
        }
    
    def preprocess(self, image_path):
        """Apply optimal preprocessing"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        
        # Perspective correction
        image = self._correct_perspective(image)
        
        # Denoise
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # Enhance contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return image
    
    def _correct_perspective(self, image):
        """Detect and correct perspective distortion"""
        # Simplified version - implement full version from POC 2
        return image
    
    def extract(self, image_path):
        """Extract all fields from KTP"""
        start_time = time.time()
        
        # Preprocess
        pre_start = time.time()
        processed = self.preprocess(image_path)
        pre_time = time.time() - pre_start
        
        # Run OCR
        ocr_start = time.time()
        result = self.ocr.predict(processed)
        ocr_time = time.time() - ocr_start
        
        # Extract fields using hybrid approach
        extract_start = time.time()
        fields = self._extract_fields_hybrid(result[0], processed)
        
        # Validate
        validation = self._validate_fields(fields)
        extract_time = time.time() - extract_start
        
        total_time = time.time() - start_time
        
        return {
            'fields': fields,
            'validation': validation,
            'confidence_score': self._calculate_overall_confidence(fields),
            'performance': {
                'total_time': total_time,
                'preprocessing_time': pre_time,
                'ocr_inference_time': ocr_time,
                'extraction_time': extract_time
            }
        }
    
    def _extract_fields_hybrid(self, ocr_result, image):
        """Hybrid extraction combining label and template methods"""
        fields = {}
        
        texts = ocr_result.get('rec_texts', [])
        scores = ocr_result.get('rec_scores', [])
        boxes = ocr_result.get('rec_boxes', [])
        
        # Combine into lines
        lines = list(zip(texts, scores, boxes))
        
        # Define field keywords (mapping to internal keys)
        field_keywords = {
            'province': ['provinsi'],
            'city': ['kabupaten', 'kota'],
            'nik': ['nik', 'n1k'],
            'name': ['nama', 'vama', 'ama'],
            'pob_dob': ['tempat/tgl lahir', 'lahir'],
            'gender': ['jenis kelamin', 'kelamin'],
            'blood_type': ['gol. darah', 'darah'],
            'address': ['alamat', 'lamat'],
            'rt_rw': ['rt/rw'],
            'village': ['kel/desa', 'desa'],
            'district': ['kecamatan'],
            'religion': ['agama'],
            'marital_status': ['status perkawinan', 'status'],
            'occupation': ['pekerjaan'],
            'citizenship': ['kewarganegaraan'],
            'expiry_date': ['berlaku hingga', 'berlaku'],
        }
        
        # Helper to find value for a label
        def find_value_for_label(label_idx, field_key=None, search_range=4):
            label_text, _, label_box = lines[label_idx]
            
            # 1. Check if the value is in the same text block (separated by colon)
            if ':' in label_text:
                parts = label_text.split(':', 1)
                if len(parts) > 1 and parts[1].strip():
                    val = parts[1].strip()
                    if field_key == 'nik':
                        clean = val.replace(' ', '').replace('O', '0').replace('I', '1').replace('B', '8')
                        if len(clean) == 16:
                            return clean, lines[label_idx][1]
                    return val, lines[label_idx][1]
            
            # 2. Check subsequent lines for spatial proximity (Y-coordinate overlap)
            label_y_mid = label_box[1] + label_box[3] / 2
            
            candidates = []
            
            for i in range(label_idx + 1, min(label_idx + search_range + 1, len(lines))):
                curr_text, curr_score, curr_box = lines[i]
                curr_y_mid = curr_box[1] + curr_box[3] / 2
                
                # Check vertical alignment (within 45 pixels)
                if abs(label_y_mid - curr_y_mid) < 45:
                    # Also ensure it's to the right of the label
                    if curr_box[0] > label_box[0] + 5:
                        val = curr_text.lstrip(': ').strip()
                        if val:
                            candidates.append((val, curr_score, curr_box[0]))
            
            if candidates:
                # Special logic for NIK: if any candidate is exactly 16 digits, take it
                if field_key == 'nik':
                    for val, score, _ in candidates:
                        clean = val.replace(' ', '').replace('O', '0').replace('I', '1').replace('B', '8')
                        if len(clean) == 16:
                            return clean, score

                # Sort by X-position to ensure correct order
                candidates.sort(key=lambda x: x[2])
                merged_val = " ".join([c[0] for c in candidates])
                avg_score = sum([c[1] for c in candidates]) / len(candidates)
                return merged_val, avg_score
                        
            return None, 0.0

        for field_key, keywords in field_keywords.items():
            for idx, (text, score, box) in enumerate(lines):
                text_lower = text.lower()
                if any(kw in text_lower for kw in keywords):
                    val, val_score = find_value_for_label(idx, field_key=field_key)
                    if val:
                        fields[field_key] = {
                            'value': val.upper(),
                            'confidence': val_score,
                            'bbox': box.tolist() if hasattr(box, 'tolist') else box
                        }
                        break
        
        # Post-process POB/DOB split
        if 'pob_dob' in fields:
            raw_val = fields['pob_dob']['value']
            # Split by comma or space before date pattern
            parts = re.split(r'[,]', raw_val, 1)
            if len(parts) == 2:
                fields['place_of_birth'] = {'value': parts[0].strip(), 'confidence': fields['pob_dob']['confidence']}
                fields['date_of_birth'] = {'value': parts[1].strip(), 'confidence': fields['pob_dob']['confidence']}
            else:
                # Fallback: look for date pattern at the end
                date_match = re.search(r'(\d{2}-\d{2}-\d{4})', raw_val)
                if date_match:
                    dob = date_match.group(1)
                    pob = raw_val.replace(dob, '').strip(', ').strip()
                    fields['place_of_birth'] = {'value': pob, 'confidence': fields['pob_dob']['confidence']}
                    fields['date_of_birth'] = {'value': dob, 'confidence': fields['pob_dob']['confidence']}
            
            # Remove raw field if both splits succeeded
            if 'place_of_birth' in fields and 'date_of_birth' in fields:
                del fields['pob_dob']
        
        # Special handling for Province and City (usually first two lines, might not have labels)
        if 'province' not in fields and len(lines) > 0:
             if 'PROVINSI' in lines[0][0]:
                 fields['province'] = {'value': lines[0][0].replace('PROVINSI', '').strip(), 'confidence': lines[0][1]}
        
        if 'city' not in fields and len(lines) > 1:
             if any(kw in lines[1][0] for kw in ['KABUPATEN', 'KOTA']):
                 fields['city'] = {'value': lines[1][0].replace('KABUPATEN', '').replace('KOTA', '').strip(), 'confidence': lines[1][1]}

        return fields

    def _validate_fields(self, fields):
        """Validate extracted fields"""
        validation = {}
        
        for field_name, rules in self.validation_rules.items():
            if field_name not in fields:
                validation[field_name] = {
                    'valid': False,
                    'error': 'Missing field'
                }
                continue
            
            value = fields[field_name].get('value', '')
            
            # Pre-validation cleaning
            if field_name == 'nik':
                value = value.replace(' ', '').replace('O', '0').replace('I', '1').replace('B', '8')
            elif field_name == 'name':
                # Remove any stray numbers from name
                value = re.sub(r'\d', '', value).strip()
            
            # Pattern validation
            if 'pattern' in rules:
                if not re.match(rules['pattern'], value):
                    validation[field_name] = {
                        'valid': False,
                        'error': 'Pattern mismatch'
                    }
                    continue
            
            # Value set validation
            if 'values' in rules:
                if value not in rules['values']:
                    # Try fuzzy matching
                    from difflib import get_close_matches
                    matches = get_close_matches(value, rules['values'], n=1, cutoff=0.8)
                    if matches:
                        validation[field_name] = {
                            'valid': True,
                            'corrected': matches[0],
                            'original': value
                        }
                    else:
                        validation[field_name] = {
                            'valid': False,
                            'error': f'Invalid value: {value}'
                        }
                    continue
            
            validation[field_name] = {'valid': True}
        
        return validation
    
    def _calculate_overall_confidence(self, fields):
        """Calculate overall extraction confidence"""
        if not fields:
            return 0.0
        
        confidences = [f.get('confidence', 0.0) for f in fields.values()]
        return sum(confidences) / len(confidences)

if __name__ == "__main__":
    # Test end-to-end
    extractor = KTPExtractor()

    test_images = [
        'images/ktp.jpg', 
        # r'C:/Users/Probuddy/.gemini/antigravity/brain/86bf6e54-3a09-4263-8562-3ca084359ee8/uploaded_image_1766678261722.jpg'
    ]

    for img_path in test_images:
        print(f"\nProcessing: {img_path}")
        try:
            result = extractor.extract(img_path)
            print(f"Overall Confidence: {result['confidence_score']:.2f}")
            print("\nExtracted Fields:")
            print(json.dumps(result['fields'], indent=2))
            print("\nValidation:")
            print(json.dumps(result['validation'], indent=2))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
