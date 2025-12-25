
import re
from ocr import KTPExtractor
import json

def test_logic():
    extractor = KTPExtractor()
    
    # Test POB/DOB Split
    print("--- Testing POB/DOB Split ---")
    raw_vals = ["FUJIAN.25-03-1977", "PEKALONGAN, 10-03-1978", "JAKARTA 27-08-2002"]
    
    for raw_val in raw_vals:
        fields = {'pob_dob': {'value': raw_val, 'confidence': 0.99}}
        
        # Logic from ocr.py
        match = re.search(r'^(.*?)[,\.\s]+(\d{2}[-\s][0-9OA-Z]{2}[-\s]\d{4})', raw_val)
        if match:
            pob = match.group(1).strip(',. ').strip()
            dob = match.group(2).replace(' ', '-').replace('O', '0').replace('I', '1')
            fields['place_of_birth'] = {'value': pob, 'confidence': 0.99}
            fields['date_of_birth'] = {'value': dob, 'confidence': 0.99}
            del fields['pob_dob']
        else:
            parts = re.split(r'[,.\s]', raw_val)
            if len(parts) >= 2:
                pob = " ".join(parts[:-1]).strip(',. ').strip()
                dob = parts[-1].strip()
                fields['place_of_birth'] = {'value': pob, 'confidence': 0.99}
                fields['date_of_birth'] = {'value': dob, 'confidence': 0.99}
                del fields['pob_dob']
        
        print(f"Raw: {raw_val} -> {fields.get('place_of_birth', {}).get('value')}, {fields.get('date_of_birth', {}).get('value')}")

    # Test Gender Mapping
    print("\n--- Testing Gender Mapping ---")
    genders = ["MALE GOL DARAH", "FEMALE", "LAKI-LAKI NOISE"]
    for g in genders:
        val = g.upper()
        # Fixed logic
        if 'PEREM' in val or 'FEMALE' in val: 
            mapped = 'PEREMPUAN' if 'PEREM' in val else 'FEMALE'
        elif 'LAKI' in val or 'MALE' in val: 
            mapped = 'LAKI-LAKI' if 'LAKI' in val else 'MALE'
        else:
            mapped = val
        print(f"Original: {g} -> Mapped: {mapped}")

    # Test Validation Rules
    print("\n--- Testing Validation Rules ---")
    test_fields = {
        'gender': {'value': 'MALE'},
        'religion': {'value': 'CHRISTIAN'},
        'marital_status': {'value': 'MARRIED'},
        'place_of_birth': {'value': 'FUJIAN'},
        'date_of_birth': {'value': '25-03-1977'}
    }
    validation = extractor._validate_fields(test_fields)
    for field, res in validation.items():
        if field in test_fields:
            print(f"{field}: {res}")

if __name__ == "__main__":
    test_logic()
