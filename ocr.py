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
            'blood_type': {
                'values': ['A', 'B', 'AB', 'O', '-'],
                'required': False
            },
            'address': {'required': True},
            'rt_rw': {
                'pattern': r'^\d{3}/\d{3}$',
                'required': True
            },
            'village': {'required': True},
            'district': {'required': True},
            'religion': {
                'values': ['ISLAM', 'KRISTEN', 'KATOLIK', 'KATHOLIK', 'HINDU', 'BUDDHA', 'KHONGHUCU'],
                'required': False
            },
            'marital_status': {
                'values': ['BELUM KAWIN', 'KAWIN', 'CERAI HIDUP', 'CERAI MATI'],
                'required': False
            },
            'occupation': {
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
        """Hybrid extraction with robust regex and spatial logic"""
        fields = {}
        img_h, img_w = image.shape[:2]
        
        texts = ocr_result.get('rec_texts', [])
        scores = ocr_result.get('rec_scores', [])
        boxes = ocr_result.get('rec_boxes', [])
        
        # Combine into lines
        lines = list(zip(texts, scores, boxes))
        
        # Define field keywords as regex for word boundaries
        field_keywords = {
            'province': [re.compile(r'\bPROVINSI\b', re.I)],
            'city': [re.compile(r'\bKABUPATEN\b', re.I), re.compile(r'\bKOTA\b', re.I)],
            'nik': [re.compile(r'\bNIK\b', re.I), re.compile(r'\bN1K\b', re.I)],
            'name': [re.compile(r'\bNAMA\b', re.I), re.compile(r'\bVAMA\b', re.I)],
            'pob_dob': [re.compile(r'TEMPAT\s*/\s*TGL\s*LAHIR', re.I), re.compile(r'\bLAHIR\b', re.I)],
            'gender': [re.compile(r'JENIS\s*KELAMIN', re.I), re.compile(r'\bKELAMIN\b', re.I)],
            'blood_type': [re.compile(r'GOL\.\s*DARAH', re.I)],
            'address': [re.compile(r'\bAL[A\s]*M[A\s]*T\b', re.I), re.compile(r'\bLAMAT\b', re.I)],
            'rt_rw': [re.compile(r'\bRT/RW\b', re.I)],
            'village': [re.compile(r'KEL/DESA', re.I), re.compile(r'\bDESA\b', re.I)],
            'district': [re.compile(r'\bKECAMATAN\b', re.I)],
            'religion': [re.compile(r'\bAGAMA\b', re.I)],
            'marital_status': [re.compile(r'STATUS\s*PERKAWINAN', re.I), re.compile(r'\bSTATUS\b', re.I)],
            'occupation': [re.compile(r'\bPEKERJAAN\b', re.I)],
            'citizenship': [re.compile(r'\bKEWARGANEGARAAN\b', re.I)],
            'expiry_date': [re.compile(r'BERLAKU\s*HINGGA', re.I), re.compile(r'\bBERLAKU\b', re.I)],
        }

        # 1. Identify all likely label indices first with spatial constraint
        label_map = {} # label_idx -> field_key
        all_label_indices = set()
        
        for idx, (text, _, box) in enumerate(lines):
            # KTP labels are generally on the left side
            # Relaxed for Address specifically as it might be shifted
            is_address_candidate = any(p.search(text) for p in field_keywords['address'])
            x_limit = img_w * 0.45 if is_address_candidate else img_w * 0.4
            
            if box[0] > x_limit:
                continue
                
            for field_key, patterns in field_keywords.items():
                if any(p.search(text) for p in patterns):
                    label_map[idx] = field_key
                    all_label_indices.add(idx)
                    break

        # Helper to find value for a label
        def find_value_for_label(label_idx, field_key=None, search_range=4):
            label_text, _, label_box = lines[label_idx]
            label_y_mid = label_box[1] + label_box[3] / 2
            
            # Helper to extract clean 16 digits from a string
            def get_nik_from_str(s):
                clean = re.sub(r'\D', '', s.replace('O', '0').replace('I', '1').replace('B', '8'))
                if len(clean) >= 16:
                    # Take the rightmost 16 digits to avoid stray noise prefixes (like marker digits)
                    return clean[-16:]
                return None

            # Check for value in same block (after colon)
            if ':' in label_text:
                parts = label_text.split(':', 1)
                if len(parts) > 1:
                    val = parts[1].strip()
                    if val:
                        if field_key == 'nik':
                            nik = get_nik_from_str(val)
                            if nik: return nik, lines[label_idx][1]
                        else:
                            return val, lines[label_idx][1]
            
            # Check subsequent lines
            candidates = []
            for i in range(label_idx + 1, min(label_idx + search_range + 1, len(lines))):
                if i in all_label_indices: break
                    
                curr_text, curr_score, curr_box = lines[i]
                curr_y_mid = curr_box[1] + curr_box[3] / 2
                
                # Tight vertical window
                v_threshold = max(20, label_box[3] * 0.5)
                if abs(label_y_mid - curr_y_mid) < v_threshold:
                    if curr_box[0] > label_box[0] + 5:
                        val = curr_text.lstrip(': ').strip()
                        if val:
                            # For NIK, if this box alone is 16 digits, it's a very strong candidate
                            if field_key == 'nik':
                                nik = get_nik_from_str(val)
                                if nik: return nik, curr_score
                            
                            candidates.append((val, curr_score, curr_box[0]))
            
            if candidates:
                candidates.sort(key=lambda x: x[2])
                # Special NIK handling for merged candidates
                if field_key == 'nik':
                    merged_raw = "".join([c[0] for c in candidates])
                    nik = get_nik_from_str(merged_raw)
                    if nik: return nik, sum([c[1] for c in candidates]) / len(candidates)
                else:
                    merged_val = " ".join([c[0] for c in candidates])
                    avg_score = sum([c[1] for c in candidates]) / len(candidates)
                    return merged_val, avg_score
                        
            return None, 0.0

        # 2. Extract values for identified labels
        for idx, field_key in label_map.items():
            if field_key in fields: continue
            
            val, val_score = find_value_for_label(idx, field_key=field_key)
            if val:
                # Negative filtering for Name: don't pick up common address words
                if field_key == 'name':
                    addr_words = ['DUSUN', 'DESA', 'KELURAHAN', 'KECAMATAN', 'RT/RW', 'RT', 'RW', 'KABUPATEN', 'PROVINSI']
                    if any(word in val.upper() for word in addr_words):
                        continue
                        
                fields[field_key] = {
                    'value': val.upper(),
                    'confidence': val_score,
                    'bbox': lines[idx][2].tolist() if hasattr(lines[idx][2], 'tolist') else lines[idx][2]
                }

        # Pre-extract Province/City if possible for cleaning
        if 'province' not in fields:
            for text, score, _ in lines[:5]:
                if 'PROVINSI' in text.upper():
                    fields['province'] = {'value': text.upper().replace('PROVINSI', '').strip(), 'confidence': score}
                    break
        
        if 'city' not in fields:
            for text, score, _ in lines[:8]:
                if any(kw in text.upper() for kw in ['KABUPATEN', 'KOTA']):
                    val = text.upper().replace('KABUPATEN', '').replace('KOTA', '').strip()
                    fields['city'] = {'value': val, 'confidence': score}
                    break

        # 3. Alamat sub-fields (indented)
        if 'address' in fields:
            addr_idx = [i for i, k in label_map.items() if k == 'address'][0]
            for i in range(addr_idx + 1, min(addr_idx + 10, len(lines))):
                if i in all_label_indices: continue
                
                text = lines[i][0].upper().strip()
                score = lines[i][1]
                box = lines[i][2]
                
                if 'rt_rw' not in fields and re.search(r'\d{3}/\d{3}', text):
                    match = re.search(r'(\d{3}/\d{3})', text)
                    fields['rt_rw'] = {'value': match.group(1), 'confidence': score, 'bbox': box.tolist()}
                
        # 4. POB/DOB split
        if 'pob_dob' in fields:
            raw_val = fields['pob_dob']['value']
            match = re.search(r'^(.*?)[,\s]+(\d{2}[-\s][0-9OA-Z]{2}[-\s]\d{4})', raw_val)
            if match:
                pob = match.group(1).strip(', ').strip()
                dob = match.group(2).replace(' ', '-').replace('O', '0').replace('I', '1')
                fields['place_of_birth'] = {'value': pob, 'confidence': fields['pob_dob']['confidence']}
                fields['date_of_birth'] = {'value': dob, 'confidence': fields['pob_dob']['confidence']}
            else:
                parts = re.split(r'[,]', raw_val, 1)
                if len(parts) == 2:
                    fields['place_of_birth'] = {'value': parts[0].strip(), 'confidence': fields['pob_dob']['confidence']}
                    fields['date_of_birth'] = {'value': parts[1].strip(), 'confidence': fields['pob_dob']['confidence']}
            if 'place_of_birth' in fields and 'date_of_birth' in fields:
                del fields['pob_dob']
        
        # 5. Global cleanup and noise removal
        if 'gender' in fields:
            val = fields['gender']['value']
            if 'LAKI' in val: fields['gender']['value'] = 'LAKI-LAKI'
            elif 'PEREM' in val: fields['gender']['value'] = 'PEREMPUAN'
            
        if 'religion' in fields:
            val = fields['religion']['value']
            for rel in self.validation_rules['religion']['values']:
                if rel in val:
                    fields['religion']['value'] = rel
                    break

        # Cross-field cleaning (Removing city names and leaked dates from Occupation/Name)
        blacklist = []
        if 'city' in fields: blacklist.append(fields['city']['value'])
        if 'province' in fields: blacklist.append(fields['province']['value'])
        if 'place_of_birth' in fields: blacklist.append(fields['place_of_birth']['value'])
        
        for field_key in ['occupation', 'name']:
            if field_key in fields:
                val = fields[field_key]['value']
                # Remove city/province names from end
                for item in blacklist:
                    if len(item) > 3 and val.endswith(item):
                        val = val[:len(val)-len(item)].strip(', ').strip()
                
                # Remove trailing dates (common leakage from Issue Date)
                val = re.sub(r'[\s,-]+\d{2}[-\s][0-9O]{2}[-\s]\d{4}$', '', val)
                fields[field_key]['value'] = val.strip()

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
