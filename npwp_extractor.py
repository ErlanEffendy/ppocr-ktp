from rapidocr_onnxruntime import RapidOCR
import cv2
import re
import time
import numpy as np
import os
import json
from ocr import RegionalMatcher

class NPWPExtractor:
    _ocr_instance = None
    
    def __init__(self):
        if NPWPExtractor._ocr_instance is None:
            NPWPExtractor._ocr_instance = RapidOCR(is_cls=True)
        self.ocr = NPWPExtractor._ocr_instance
        self.regional_matcher = RegionalMatcher(self)
        
        self.validation_rules = {
            'npwp_number': {'required': True},
            'taxpayer_name': {'required': True},
            'address': {'required': True},
            'city': {'required': True},
            'province': {'required': True},
            'kpp': {'required': True},
            'registration_date': {'required': True}
        }
    
    def _levenshtein_distance(self, s1, s2):
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

    def extract(self, image_path):
        start_time = time.time()
        
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
            
        ocr_start = time.time()
        ocr_result, _ = self.ocr(image)
        ocr_time = time.time() - ocr_start
        
        formatted_result = {
            'rec_texts': [],
            'rec_scores': [],
            'rec_boxes': []
        }
        if ocr_result:
            for box, text, score in ocr_result:
                formatted_result['rec_texts'].append(text)
                formatted_result['rec_scores'].append(score)
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                formatted_result['rec_boxes'].append([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])
        
        extract_start = time.time()
        fields = self._extract_fields(formatted_result, image)
        validation = self._validate_fields(fields)
        extract_time = time.time() - extract_start
        
        total_time = time.time() - start_time
        
        return {
            'fields': fields,
            'validation': validation,
            'confidence_score': self._calculate_overall_confidence(fields),
            'performance': {
                'total_time': total_time,
                'preprocessing_time': 0,
                'ocr_inference_time': ocr_time,
                'extraction_time': extract_time
            }
        }
    
    def _extract_fields(self, ocr_result, image):
        fields = {}
        texts = ocr_result.get('rec_texts', [])
        scores = ocr_result.get('rec_scores', [])
        boxes = ocr_result.get('rec_boxes', [])
        
        lines = list(zip(texts, scores, boxes))
        
        name_candidate_idx = -1
        reg_date_idx = -1

        for idx, (text, score, box) in enumerate(lines):
            text_upper = text.upper()
            
            # KPP
            if ('KPP ' in text_upper or 'PENERBIT' in text_upper) and 'kpp' not in fields:
                # If it's "Penerbit", we might want to keep the label or just the value
                # "Penerbit : 805" -> "KPP 805" or just "805"
                # Let's keep the full text for now as it's more informative
                fields['kpp'] = {'value': text_upper.strip(), 'confidence': score}
            
            # NPWP Number
            # Improved regex to handle "NPWP : 03.026..."
            nik_match = re.search(r'(\d{2}[.,\s]?\d{3}[.,\s]?\d{3}[.,\s]?\d[\-\s]?\d{3}[.,\s]?\d{3})', text)
            if nik_match and 'npwp_number' not in fields:
                val = nik_match.group(1).replace(' ', '').replace(',', '.')
                clean_val_match = re.sub(r'[^\d]', '', val)
                if len(clean_val_match) >= 15:
                    cv = clean_val_match[:15]
                    val = f"{cv[0:2]}.{cv[2:5]}.{cv[5:8]}.{cv[8:9]}-{cv[9:12]}.{cv[12:15]}"
                fields['npwp_number'] = {'value': val, 'confidence': score}
                name_candidate_idx = idx + 1
            
            # Registration Date
            reg_match = re.search(r'(?:TANGGAL|TGL|TERDAFTAR)\s*[:：\-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})', text_upper)
            if reg_match and 'registration_date' not in fields:
                clean_date = reg_match.group(1).replace('-', '/')
                fields['registration_date'] = {'value': clean_date, 'confidence': score}
                reg_date_idx = idx

        # Extract taxpayer_name
        if name_candidate_idx != -1 and name_candidate_idx < len(lines):
            name_text = lines[name_candidate_idx][0]
            # Avoid labels and dates in name
            if not any(kw in name_text.upper() for kw in ['NPWP', 'PENERBIT', 'TERDAFTAR']) and not re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', name_text):
                fields['taxpayer_name'] = {'value': name_text.strip(), 'confidence': lines[name_candidate_idx][1]}

        # If we didn't find registration date explicitly, try a regex pass for just date near the end
        if reg_date_idx == -1:
            for i in range(len(lines)-1, -1, -1):
                if re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', lines[i][0]):
                    reg_date_idx = i
                    if 'registration_date' not in fields:
                        match = re.search(r'(\d{2}[/-]\d{2}[/-]\d{4})', lines[i][0])
                        if match:
                            fields['registration_date'] = {'value': match.group(1).replace('-', '/'), 'confidence': lines[i][1]}
                    break
                    
        # Address boundaries
        start_addr = name_candidate_idx + 1 if name_candidate_idx != -1 else 0
        end_addr = reg_date_idx if reg_date_idx != -1 else len(lines)
        
        address_lines = []
        for i in range(start_addr, end_addr):
            t = lines[i][0]
            t_upper = t.upper()
            if any(kw in t_upper for kw in ['NPWP16', 'KPP ', 'PENERBIT', 'TERDAFTAR', 'NPWP :']):
                continue
            if fields.get('taxpayer_name') and fields['taxpayer_name']['value'] in t:
                continue
            # Filter out very short noisy strings
            if len(t) < 4 and not re.search(r'\d', t):
                continue
            address_lines.append((t, lines[i][1]))
            
        if address_lines:
            all_text = " ".join([l[0] for l in address_lines]).upper()
            
            # Helper for substring fuzzy matching
            def match_in_text(text, items, threshold=0.6):
                best_match = None
                max_sim = 0
                match_id = None
                
                # First try exact containment (fast)
                for item in items:
                    name = item['name'].upper()
                    # Clean the name of prefixes for better KTP/NPWP matching
                    clean_name = re.sub(r'^(PROVINSI|KABUPATEN|KOTA|KAB)\b\s*', '', name).strip()
                    
                    if len(clean_name) > 3 and clean_name in text:
                        return name, item['id']
                
                # Then try fuzzy matching on words/n-grams
                # Split by space and punctuation to handle "MAKASSAR, SULAWESI"
                words = re.split(r'[\s,.\-/]+', text)
                for i in range(len(words)):
                    for j in range(i + 1, min(i + 5, len(words) + 1)):
                        chunk = " ".join(words[i:j])
                        if len(chunk) < 3: continue
                        
                        for item in items:
                            name = item['name'].upper()
                            clean_name = re.sub(r'^(PROVINSI|KABUPATEN|KOTA|KAB)\b\s*', '', name).strip()
                            
                            dist = self._levenshtein_distance(chunk, clean_name)
                            max_len = max(len(chunk), len(clean_name))
                            sim = 1 - (dist / max_len) if max_len > 0 else 1.0
                            
                            if sim > max_sim:
                                max_sim = sim
                                best_match = name
                                match_id = item['id']
                                
                return (best_match, match_id) if max_sim >= threshold else (None, None)

            # 1. Match Province
            p_name, p_id = match_in_text(all_text, self.regional_matcher.provinces)
            if p_name:
                fields['province'] = {'value': p_name, 'confidence': 0.9} # Boost confidence if matched
                
                # 2. Match City (only within province)
                self.regional_matcher.match_city("dummy", p_id) # Load cache
                cities = self.regional_matcher._city_cache.get(p_id, [])
                c_name, c_id = match_in_text(all_text, cities)
                if c_name:
                    fields['city'] = {'value': c_name, 'confidence': 0.9}
            
            # Reconstruct address from what's left
            full_addr_text = " ".join([l[0] for l in address_lines])
            
            # Remove matched province and city names and their common variants
            for field in ['province', 'city']:
                if field in fields:
                    val = fields[field]['value']
                    # Remove full name
                    full_addr_text = re.sub(re.escape(val), '', full_addr_text, flags=re.IGNORECASE)
                    # Remove bare name (without KOTA/KABUPATEN/PROVINSI)
                    bare = re.sub(r'^(PROVINSI|KABUPATEN|KOTA|KAB)\b\s*', '', val, flags=re.IGNORECASE).strip()
                    if len(bare) > 3:
                        full_addr_text = re.sub(re.escape(bare), '', full_addr_text, flags=re.IGNORECASE)

            # Final cleaning of regional keywords and OCR noise
            full_addr_text = re.sub(r'\b(KOTA|KABUPATEN|KAB|PROVINSI|ADM|DKI|PENERBIT|TERDAFTAR)\b', '', full_addr_text, flags=re.IGNORECASE)
            full_addr_text = re.sub(r'\bJAKAR\w*\b', '', full_addr_text, flags=re.IGNORECASE)
            full_addr_text = re.sub(r'\bOdjp\b', '', full_addr_text, flags=re.IGNORECASE)
            full_addr_text = re.sub(r'\b[:：\-]\b', '', full_addr_text)
            full_addr_text = re.sub(r'\s+', ' ', full_addr_text).strip(', ')
            
            if full_addr_text:
                avg_conf = sum([l[1] for l in address_lines]) / len(address_lines)
                fields['address'] = {'value': full_addr_text.strip(), 'confidence': avg_conf}

            if 'province' not in fields:
                fields['province'] = {'value': 'UNKNOWN', 'confidence': 0.0}
            if 'city' not in fields:
                fields['city'] = {'value': 'UNKNOWN', 'confidence': 0.0}

        return fields

    def _validate_fields(self, fields):
        validation = {}
        for field_name, rules in self.validation_rules.items():
            if field_name not in fields:
                validation[field_name] = {'valid': False, 'error': 'Missing field'}
                continue
            
            field_data = fields[field_name]
            value = str(field_data.get('value', ''))
            valid = True
            error = None
            
            if 'pattern' in rules:
                pattern = str(rules['pattern'])
                if not re.match(pattern, value):
                    valid = False
                    error = 'Pattern mismatch'
            
            res: dict = {'valid': valid}
            if error:
                res['error'] = error
            validation[field_name] = res
        return validation

    def _calculate_overall_confidence(self, fields):
        if not fields: return 0.0
        confidences = [f.get('confidence', 0.0) for f in fields.values()]
        return sum(confidences) / len(confidences)
