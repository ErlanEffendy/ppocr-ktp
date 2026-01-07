from paddleocr import PaddleOCR
import cv2
import re
import json
import numpy as np
import time
import requests
import os
import pickle
from ultralytics import YOLO

class RegionalMatcher:
    """Helper to match Indonesian regional names (Province, City, District, Village)"""
    BASE_URL = "https://emsifa.github.io/api-wilayah-indonesia/api"
    CACHE_DIR = ".cache_regional"

    def __init__(self, extractor_ref):
        self.extractor = extractor_ref
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self.provinces = self._load_data("provinces.json")
        self._city_cache = {}
        self._district_cache = {}
        self._village_cache = {}

    def _load_data(self, endpoint):
        cache_path = os.path.join(self.CACHE_DIR, endpoint.replace("/", "_"))
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        try:
            url = f"{self.BASE_URL}/{endpoint}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                with open(cache_path, "wb") as f:
                    pickle.dump(data, f)
                return data
        except Exception as e:
            print(f"Warning: Could not fetch regional data from {endpoint}: {e}")
        return []

    def _fuzzy_match(self, text, items, prefixes_to_strip, threshold=0.5):
        if not text: return None, None
        
        input_raw = text.upper()
        
        # Clean versions for base comparison
        clean_input = input_raw
        for p in prefixes_to_strip:
            clean_input = re.sub(rf'^{p}\b\s*', '', clean_input, flags=re.I)
        clean_input = clean_input.strip()
        
        best_match = None
        max_score = 0
        
        for item in items:
            orig_name = item['name'].upper()
            
            # Clean candidate
            clean_cand = orig_name
            for p in prefixes_to_strip:
                clean_cand = re.sub(rf'^{p}\b\s*', '', clean_cand, flags=re.I)
            clean_cand = clean_cand.strip()
            
            # Distance between clean versions
            dist = self.extractor._levenshtein_distance(clean_input, clean_cand)
            max_len = max(len(clean_input), len(clean_cand))
            base_sim = 1 - (dist / max_len) if max_len > 0 else 1.0
            
            # Bonus for label matching
            label_match_bonus = 0
            for p in prefixes_to_strip:
                if (p in input_raw) == (p in orig_name):
                    label_match_bonus += 0.05 # Small boost for same label status
            
            score = base_sim + label_match_bonus
            
            if score > max_score:
                max_score = score
                best_match = item
        
        # Adjust threshold check to use base_sim or final score
        # We want at least a decent base similarity
        if best_match and (max_score - 0.05) >= threshold:
            return best_match['name'], best_match['id']
        return None, None

    def match_province(self, text):
        return self._fuzzy_match(text, self.provinces, ["PROVINSI"], threshold=0.5)

    def match_city(self, text, province_id):
        if not province_id: return None, None
        if province_id not in self._city_cache:
            self._city_cache[province_id] = self._load_data(f"regencies/{province_id}.json")
        return self._fuzzy_match(text, self._city_cache[province_id], ["KABUPATEN", "KOTA", "KAB"], threshold=0.5)

    def match_district(self, text, city_id):
        if not city_id: return None, None
        if city_id not in self._district_cache:
            self._district_cache[city_id] = self._load_data(f"districts/{city_id}.json")
        return self._fuzzy_match(text, self._district_cache[city_id], ["KECAMATAN", "KEC"], threshold=0.5)

    def match_village(self, text, district_id):
        if not district_id: return None, None
        if district_id not in self._village_cache:
            self._village_cache[district_id] = self._load_data(f"villages/{district_id}.json")
        return self._fuzzy_match(text, self._village_cache[district_id], ["KEL/DESA", "KELURAHAN", "DESA", "KEL"], threshold=0.5)

class KTPExtractor:
    _ocr_instance = None  # Singleton pattern for OCR
    _seg_model_instance = None # Singleton pattern for Segmentation
    
    def __init__(self):
        # Reuse OCR model across instances to avoid reloading
        if KTPExtractor._ocr_instance is None:
            KTPExtractor._ocr_instance = PaddleOCR(
                use_textline_orientation=True,
                lang='en'
            )
        self.ocr = KTPExtractor._ocr_instance
        
        # Load YOLOv11 segmentation model
        if KTPExtractor._seg_model_instance is None:
            model_path = os.path.join('seg_model', 'yolo11', 'best.pt')
            if os.path.exists(model_path):
                KTPExtractor._seg_model_instance = YOLO(model_path)
            else:
                print(f"Warning: Segmentation model not found at {model_path}")
                
        self.seg_model = KTPExtractor._seg_model_instance
        self.regional_matcher = RegionalMatcher(self)
        
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
                'values': ['LAKI-LAKI', 'PEREMPUAN', 'MALE', 'FEMALE'],
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
                'values': ['ISLAM', 'KRISTEN', 'KATOLIK', 'KATHOLIK', 'HINDU', 'BUDDHA', 'KHONGHUCU', 'PROTESTANT', 'CHRISTIAN'],
                'required': False
            },
            'marital_status': {
                'values': ['BELUM KAWIN', 'KAWIN', 'CERAI HIDUP', 'CERAI MATI', 'SINGLE', 'MARRIED', 'DIVORCED', 'WIDOWED'],
                'required': False
            },
            'occupation': {
                'required': True
            },
            'citizenship': {'required': True},
            'expiry_date': {'required': True}
        }
    
    def preprocess_warped(self, image):
        """Apply light preprocessing to warped image"""
        # Light preprocessing only: fast blur + contrast boost
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Quick contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        p2, p98 = np.percentile(l_channel, (2, 98))
        l_channel = np.clip((l_channel - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        lab[:,:,0] = l_channel
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return image
    
    def _correct_perspective(self, image):
        """Detect and correct perspective distortion"""
        # Simplified version - implement full version from POC 2
        return image
    
    def _detect_and_warp(self, image):
        """Detect KTP using YOLOv11 and apply perspective warp or bounding box crop"""
        if self.seg_model is None:
            return image
            
        results = self.seg_model(image, verbose=False)
        if not results or len(results[0]) == 0:
            return image
            
        # Get the first detection (assume it's the KTP)
        result = results[0]

        # 2. Bounding Box Crop (Fallback)
        if result.boxes is not None:
            # results[0].boxes.xyxy is (N, 4) tensor
            box = result.boxes.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            cropped = image[y1:y2, x1:x2]
            return cropped
        
        # 1. Perspective Warp (Primary)
        if result.masks is not None:
            # Get the coordinates of the mask
            mask_coords = result.masks.xy[0] # List of [x, y]
            if len(mask_coords) >= 4:
                # Find the 4 corners using approxPolyDP
                contour = mask_coords.astype(np.float32)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) != 4:
                    # If not 4, try to force 4 by finding hull and then most distant points/indices
                    hull = cv2.convexHull(contour)
                    epsilon = 0.02 * cv2.arcLength(hull, True)
                    approx = cv2.approxPolyDP(hull, epsilon, True)
                
                if len(approx) == 4:
                    # Sort corners: top-left, top-right, bottom-right, bottom-left
                    pts = approx.reshape(4, 2)
                    rect = np.zeros((4, 2), dtype="float32")
                    
                    s = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(s)]
                    rect[2] = pts[np.argmax(s)]
                    
                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)]
                    rect[3] = pts[np.argmax(diff)]
                    
                    # Define destination points (85.6mm x 54.0mm ratio)
                    width = 1000
                    height = 631
                    dst = np.array([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ], dtype="float32")
                    
                    # Perspective transform
                    M = cv2.getPerspectiveTransform(rect, dst)
                    warped = cv2.warpPerspective(image, M, (width, height))
                    return warped

        return image

    def extract(self, image_path):
        """Extract all fields from KTP"""
        start_time = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
            
        # Preprocess - Perspective Correction First
        pre_start = time.time()
        
        # 1. Detect and Warp
        warped = self._detect_and_warp(image)
        
        # 2. Regular preprocessing on warped image
        processed = self.preprocess_warped(warped)
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
            'address': [re.compile(r'[A4]L[A\s4]*M[A\s4]*[Tf7l1I]', re.I), re.compile(r'L[A\s]*M[A\s]*[Tf7l1I]', re.I)],
            'rt_rw': [re.compile(r'RT\s*/\s*RW', re.I)],
            'village': [re.compile(r'KEL/DESA', re.I), re.compile(r'DESA\b', re.I)],
            'district': [re.compile(r'KECAMATAN', re.I)],
            'religion': [re.compile(r'\bAGAMA\b', re.I)],
            'marital_status': [re.compile(r'STATUS\s*PERKAWINAN', re.I), re.compile(r'\bSTATUS\b', re.I)],
            'occupation': [re.compile(r'\bPEKERJAAN\b', re.I)],
            'citizenship': [re.compile(r'\bKEWARGANEGARAAN\b', re.I)],
            'expiry_date': [re.compile(r'BERLAKU\s*HINGGA', re.I), re.compile(r'\bBERLAKU\b', re.I)],
        }

        # 1. Identify all likely label indices first with spatial constraint
        label_map = {} # label_idx -> field_key
        all_label_indices = set()
        
        # Pre-compile regex patterns (avoid recompiling in loop)
        regex_cache = {}
        for field_key, patterns in field_keywords.items():
            regex_cache[field_key] = patterns
        
        # Canonical names for fuzzy matching - pre-process
        canonical_map = {
            'PROVINSI': 'province',
            'KABUPATEN': 'city',
            'KOTA': 'city',
            'NIK': 'nik',
            'NAMA': 'name',
            'TEMPAT/TGL LAHIR': 'pob_dob',
            'JENIS KELAMIN': 'gender',
            'GOL. DARAH': 'blood_type',
            'ALAMAT': 'address',
            'RT/RW': 'rt_rw',
            'KEL/DESA': 'village',
            'KECAMATAN': 'district',
            'AGAMA': 'religion',
            'STATUS PERKAWINAN': 'marital_status',
            'PEKERJAAN': 'occupation',
            'KEWARGANEGARAAN': 'citizenship',
            'BERLAKU HINGGA': 'expiry_date'
        }
        canonical_list = list(canonical_map.keys())
        
        # Pre-compute cleaned canonical strings
        canonical_clean = {c: re.sub(r'[:\-\s/]', '', c) for c in canonical_list}
        
        for idx, (text, _, box) in enumerate(lines):
            bx1 = box[0] if not isinstance(box[0], (list, np.ndarray)) else box[0][0]
            # KTP labels are generally on the left side
            is_addr_region = any(kw in text.upper() for kw in ['ALAMAT', 'RT/', 'RW', 'DESA', 'KECAMATAN'])
            x_limit = img_w * 0.45 if is_addr_region else img_w * 0.4
            
            if bx1 > x_limit:
                continue

            found = False
            # Pass 1: Regex only (skip fuzzy matching for speed)
            for field_key, patterns in regex_cache.items():
                if any(p.search(text) for p in patterns):
                    label_map[idx] = field_key
                    all_label_indices.add(idx)
                    found = True
                    break
            
            # Skip Pass 2 (Fuzzy matching) for performance - keep only regex matching

        # Helper to find value for a label
        def find_value_for_label(label_idx, field_key=None, search_range=None):
            if search_range is None:
                search_range = 1 if field_key in ['gender', 'blood_type', 'religion', 'citizenship'] else 2
            
            label_text, _, label_box = lines[label_idx]
            # Assuming [x1, y1, x2, y2]
            lx1, ly1, lx2, ly2 = label_box
            label_y_mid = (ly1 + ly2) / 2
            l_height = ly2 - ly1
            
            # Helper to extract clean 16 digits from a string
            def get_nik_from_str(s):
                clean = re.sub(r'\D', '', s.replace('O', '0').replace('I', '1').replace('B', '8'))
                if len(clean) >= 16:
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
                            if nik: return nik, lines[label_idx][1], label_box
                        else:
                            return val, lines[label_idx][1], label_box
            
            # Check subsequent lines
            candidates = []
            for i in range(label_idx + 1, min(label_idx + search_range + 1, len(lines))):
                if i in all_label_indices: break
                    
                curr_text, curr_score, curr_box = lines[i]
                cx1, cy1, cx2, cy2 = curr_box
                curr_y_mid = (cy1 + cy2) / 2
                
                # Tight vertical window
                v_threshold = max(20, l_height * 0.8)
                if abs(label_y_mid - curr_y_mid) < v_threshold:
                    # If current line starts too far left, it's likely a misidentified label rather than a value
                    # (Unless it's the RT/RW which is shifted left)
                    if cx1 < img_w * 0.15 and field_key not in ['province', 'city', 'rt_rw']:
                        break
                        
                    if cx1 > lx1 + 5:
                        val = curr_text.lstrip(': ').strip()
                        if val:
                            if field_key == 'nik':
                                nik = get_nik_from_str(val)
                                if nik: return nik, curr_score, curr_box
                            
                            candidates.append({
                                'text': val,
                                'score': curr_score,
                                'bbox': curr_box
                            })
                elif curr_y_mid > label_y_mid + v_threshold:
                    # Line is too far below, stop searching to avoid sweeping up unrelated fields
                    break
            
            if candidates:
                candidates.sort(key=lambda x: x['bbox'][0])
                # Special NIK handling for merged candidates
                if field_key == 'nik':
                    merged_raw = "".join([c['text'] for c in candidates])
                    nik = get_nik_from_str(merged_raw)
                    if nik: 
                        avg_score = sum([c['score'] for c in candidates]) / len(candidates)
                        return nik, avg_score, candidates[0]['bbox']
                else:
                    merged_val = " ".join([c['text'] for c in candidates])
                    avg_score = sum([c['score'] for c in candidates]) / len(candidates)
                    return merged_val, avg_score, candidates[0]['bbox']
                        
            return None, 0.0, None

        # 2. Extract values for identified labels
        for idx, field_key in label_map.items():
            if field_key in fields: continue
            
            val, val_score, val_bbox = find_value_for_label(idx, field_key=field_key)
            if val:
                # Negative filtering for Name: don't pick up common address words
                if field_key == 'name':
                    addr_words = ['DUSUN', 'DESA', 'KELURAHAN', 'KECAMATAN', 'RT/RW', 'RT', 'RW', 'KABUPATEN', 'PROVINSI']
                    if any(word in val.upper() for word in addr_words):
                        continue
                        
                fields[field_key] = {
                    'value': val.upper(),
                    'confidence': val_score,
                    'bbox': val_bbox.tolist() if hasattr(val_bbox, 'tolist') else val_bbox
                }

        # Pre-extract Province/City if possible for cleaning
        if 'province' not in fields:
            for text, score, _ in lines[:5]:
                if 'PROVINSI' in text.upper():
                    fields['province'] = {'value': text.upper().replace('PROVINSI', '').strip(), 'confidence': score}
                    break
        
        if 'city' not in fields:
            for text, score, box in lines[:8]:
                if any(kw in text.upper() for kw in ['KABUPATEN', 'KOTA']):
                    val = text.upper().replace('KABUPATEN', '').replace('KOTA', '').strip()
                    fields['city'] = {'value': val, 'confidence': score}
                    break
                    
        # City fallback: check line below Province if still missing
        if 'city' not in fields:
            prov_line_idx = -1
            for i, (text, _, _) in enumerate(lines[:5]):
                if 'PROVINSI' in text.upper():
                    prov_line_idx = i
                    break
            
            if prov_line_idx != -1 and prov_line_idx + 1 < len(lines):
                cand_text, cand_score, cand_box = lines[prov_line_idx + 1]
                # Ensure it's not another label and it's physically close
                major_labels = ['NIK', 'NAMA', 'TEMPAT', 'JENIS', 'ALAMAT', 'RT/', 'AGAMA', 'PEKERJAAN']
                if not any(lab in cand_text.upper() for lab in major_labels):
                    if cand_box[1] < img_h * 0.25: # Cities are always near the top
                        fields['city'] = {'value': cand_text.upper().strip(), 'confidence': cand_score}

        # Sub-field logic improvement: use pivot region
        pivot_idx = -1
        if 'address' in fields:
            pivot_idx = [i for i, k in label_map.items() if k == 'address'][0]
        else:
            # Global fallback find RT/RW pattern to locate address region
            for idx, (text, _, _) in enumerate(lines):
                if re.search(r'\d{3}/\d{3}', text):
                    pivot_idx = idx - 1 if idx > 0 else 0
                    break

        if pivot_idx != -1:
            for i in range(max(0, pivot_idx), min(pivot_idx + 12, len(lines))):
                if (i in label_map and label_map[i] not in ['address', 'rt_rw', 'village', 'district']):
                    if i > pivot_idx: break # Hit another major label
                
                text = lines[i][0].upper().strip()
                score = lines[i][1]
                box = lines[i][2]
                
                if 'rt_rw' not in fields and re.search(r'\d{3}/\d{3}', text):
                    match = re.search(r'(\d{3}/\d{3})', text)
                    # Take rightmost 7 characters to avoid noise prefixes
                    val = match.group(1)
                    fields['rt_rw'] = {'value': val, 'confidence': score, 'bbox': box.tolist() if hasattr(box, 'tolist') else box}
                elif 'village' not in fields and i in label_map and label_map[i] == 'village':
                    val, s, b = find_value_for_label(i, field_key='village')
                    if val: fields['village'] = {'value': val.upper(), 'confidence': s, 'bbox': b.tolist() if hasattr(b, 'tolist') else b}
                elif 'district' not in fields and i in label_map and label_map[i] == 'district':
                    val, s, b = find_value_for_label(i, field_key='district')
                    if val: fields['district'] = {'value': val.upper(), 'confidence': s, 'bbox': b.tolist() if hasattr(b, 'tolist') else b}
                elif 'address' not in fields and i > pivot_idx and ':' not in text and (box[0] if not isinstance(box[0], (list, np.ndarray)) else box[0][0]) > img_w * 0.2:
                    # Potential multiline address value. Must NOT be purely numeric to avoid misidentifying RT/RW blocks.
                    if not re.match(r'^[\d\s/]+$', text):
                        fields['address'] = {'value': text, 'confidence': score, 'bbox': box.tolist() if hasattr(box, 'tolist') else box}
                
        # 4. POB/DOB split
        if 'pob_dob' in fields:
            raw_val = fields['pob_dob']['value']
            # Match Place and Date, allowing for comma, dot, or space separators
            match = re.search(r'^(.*?)[,\.\s]+(\d{2}[-\s][0-9OA-Z]{2}[-\s]\d{4})', raw_val)
            if match:
                pob = match.group(1).strip(',. ').strip()
                dob = match.group(2).replace(' ', '-').replace('O', '0').replace('I', '1')
                fields['place_of_birth'] = {'value': pob, 'confidence': fields['pob_dob']['confidence']}
                fields['date_of_birth'] = {'value': dob, 'confidence': fields['pob_dob']['confidence']}
            else:
                # Fallback: try splitting by multiple delimiters
                parts = re.split(r'[,.\s]', raw_val)
                # Look for a date-like part at the end
                if len(parts) >= 2:
                    pob = " ".join(parts[:-1]).strip(',. ').strip()
                    dob = parts[-1].strip()
                    fields['place_of_birth'] = {'value': pob, 'confidence': fields['pob_dob']['confidence']}
                    fields['date_of_birth'] = {'value': dob, 'confidence': fields['pob_dob']['confidence']}
            
            if 'place_of_birth' in fields and 'date_of_birth' in fields:
                del fields['pob_dob']
        
        # 5. Global cleanup and noise removal
        if 'gender' in fields:
            val = fields['gender']['value']
            # Check FEMALE first because it contains 'MALE'
            if 'PEREM' in val or 'FEMALE' in val: fields['gender']['value'] = 'PEREMPUAN' if 'PEREM' in val else 'FEMALE'
            elif 'LAKI' in val or 'MALE' in val: fields['gender']['value'] = 'LAKI-LAKI' if 'LAKI' in val else 'MALE'
            
        if 'religion' in fields:
            val = fields['religion']['value']
            # Boost: use fuzzy matching for religion
            best_religion = self._fuzzy_correct(val, self.validation_rules['religion']['values'])
            if best_religion:
                fields['religion']['value'] = best_religion

        if 'marital_status' in fields:
            val = fields['marital_status']['value']
            # Boost: use fuzzy matching for marital status
            best_status = self._fuzzy_correct(val, self.validation_rules['marital_status']['values'])
            if best_status:
                fields['marital_status']['value'] = best_status

        # 6. Hierarchical Regional Correction
        prov_id = None
        if 'province' in fields:
            best_name, pid = self.regional_matcher.match_province(fields['province']['value'])
            if best_name:
                fields['province']['value'] = best_name
                prov_id = pid
        
        city_id = None
        if 'city' in fields and prov_id:
            best_name, cid = self.regional_matcher.match_city(fields['city']['value'], prov_id)
            if best_name:
                fields['city']['value'] = best_name
                city_id = cid
        
        dist_id = None
        if 'district' in fields and city_id:
            best_name, did = self.regional_matcher.match_district(fields['district']['value'], city_id)
            if best_name:
                fields['district']['value'] = best_name
                dist_id = did
        
        if 'village' in fields and dist_id:
            best_name, vid = self.regional_matcher.match_village(fields['village']['value'], dist_id)
            if best_name:
                fields['village']['value'] = best_name

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
        """Validate extracted fields - fast version, skip fuzzy matching"""
        validation = {}
        
        # Only validate required fields for speed
        required_only = {k: v for k, v in self.validation_rules.items() if v.get('required', False)}
        
        for field_name, rules in required_only.items():
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
            
            # Pattern validation only
            if 'pattern' in rules:
                if not re.match(rules['pattern'], value):
                    validation[field_name] = {
                        'valid': False,
                        'error': 'Pattern mismatch'
                    }
                    continue
            
            # Fuzzy matching for value set validation
            if 'values' in rules:
                if value not in rules['values']:
                    # Try fuzzy matching before failing
                    best_match = self._fuzzy_correct(value, rules['values'])
                    if best_match:
                        fields[field_name]['value'] = best_match
                    else:
                        validation[field_name] = {
                            'valid': False,
                            'error': f'Invalid value: {value}'
                        }
                        continue
            
            validation[field_name] = {'valid': True}
        
        return validation

    def _levenshtein_distance(self, s1, s2):
        """Calculate the Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def _fuzzy_correct(self, text, candidates, threshold=0.6):
        """Find the best matching candidate based on Levenshtein distance"""
        if not text:
            return None
        
        text = text.upper()
        best_match = None
        max_similarity = 0
        
        for candidate in candidates:
            candidate = candidate.upper()
            distance = self._levenshtein_distance(text, candidate)
            # Normalize similarity: 1 - (distance / max_length)
            max_len = max(len(text), len(candidate))
            similarity = 1 - (distance / max_len) if max_len > 0 else 1.0
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = candidate
        
        return best_match if max_similarity >= threshold else None
    
    def _calculate_overall_confidence(self, fields):
        """Calculate overall extraction confidence"""
        if not fields:
            return 0.0
        
        confidences = [f.get('confidence', 0.0) for f in fields.values()]
        return sum(confidences) / len(confidences)

if __name__ == '__main__':
    import json
    from ocr import KTPExtractor
    extractor = KTPExtractor()
    image_path = 'images/original.jpg'
    result = extractor.extract(image_path)
    print(json.dumps(result, indent=2))
