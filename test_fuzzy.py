from ocr import KTPExtractor

def test_fuzzy_matching():
    extractor = KTPExtractor()
    
    test_cases = [
        # Religion (Agama)
        ("ISL4M", extractor.validation_rules['religion']['values'], "ISLAM"),
        ("KR1STEN", extractor.validation_rules['religion']['values'], "KRISTEN"),
        ("KATOL1K", extractor.validation_rules['religion']['values'], "KATOLIK"),
        ("BUDHA", extractor.validation_rules['religion']['values'], "BUDDHA"),
        
        # Marital Status (Status Perkawinan)
        ("K4WIN", extractor.validation_rules['marital_status']['values'], "KAWIN"),
        ("BELUM KAW1N", extractor.validation_rules['marital_status']['values'], "BELUM KAWIN"),
        ("CERAI H1DUP", extractor.validation_rules['marital_status']['values'], "CERAI HIDUP"),
        ("CERAI MAT1", extractor.validation_rules['marital_status']['values'], "CERAI MATI"),
        
        # Gender
        ("LAK1-LAK1", extractor.validation_rules['gender']['values'], "LAKI-LAKI"),
        ("PEREMPU4N", extractor.validation_rules['gender']['values'], "PEREMPUAN"),
        
        # Negative cases (should not match if too different)
        ("XYZ", extractor.validation_rules['religion']['values'], None),
    ]
    
    # Regional matching tests (Using Real Address Data)
    # Real Address: Gedung Sate
    # Province: JAWA BARAT (32)
    # City: KOTA BANDUNG (3273)
    # District: BANDUNG WETAN (3273200)
    # Village: CITARUM (3273200001)
    
    # Warm up: ensure these are fetched since the test calls methods directly
    extractor.regional_matcher.match_district("BANDUNG WETAN", "3273")
    extractor.regional_matcher.match_village("CITARUM", "3273200")
    extractor.regional_matcher.match_district("GAMBIR", "3173")
    
    regional_test_cases = [
        # Province
        ("JAWA BAR4T", "JAWA BARAT"),
        
        # City (requires province ID)
        ("K0TA BANDUN6", "32", "KOTA BANDUNG"),
        
        # District (requires city ID)
        ("BANDUN6 WETA", "3273", "BANDUNG WETAN"),
        
        # Village (requires district ID)
        ("CITARUM", "3273200", "CITARUM"),
        
        # Another real address: Monas
        # Province: DKI JAKARTA (31)
        # City: KOTA JAKARTA PUSAT (3173)
        # District: GAMBIR (3173010)
        
        ("DKI JAK4RT4", "DKI JAKARTA"),
        ("JAKARTA PUS4T", "31", "KOTA JAKARTA PUSAT"), 
        ("GAMB1R", "3173", "GAMBIR"),
    ]
    
    print("Running Fuzzy Matching Tests...")
    print("-" * 50)
    
    passed = 0
    for input_text, candidates, expected in test_cases:
        result = extractor._fuzzy_correct(input_text, candidates)
        if result == expected:
            print(f"PASS: '{input_text}' -> '{result}'")
            passed += 1
        else:
            print(f"FAIL: '{input_text}' -> Expected '{expected}', got '{result}'")
            
    print("\nRunning Regional Matching Tests (Requires Internet or Cache)...")
    print("-" * 50)
    
    for case in regional_test_cases:
        if len(case) == 2: # Province
            input_text, expected = case
            result, _ = extractor.regional_matcher.match_province(input_text)
        elif len(case) == 3: # City, District, Village
            input_text, parent_id, expected = case
            if len(parent_id) == 2: # Province ID
                result, _ = extractor.regional_matcher.match_city(input_text, parent_id)
            elif len(parent_id) == 4: # City ID
                result, _ = extractor.regional_matcher.match_district(input_text, parent_id)
            elif len(parent_id) == 7: # District ID
                result, _ = extractor.regional_matcher.match_village(input_text, parent_id)
        
        if result == expected:
            print(f"PASS: '{input_text}' -> '{result}'")
            passed += 1
        else:
            print(f"FAIL: '{input_text}' -> Expected '{expected}', got '{result}'")

    total_tests = len(test_cases) + len(regional_test_cases)
    print("-" * 50)
    print(f"Results: {passed}/{total_tests} passed")

if __name__ == "__main__":
    test_fuzzy_matching()
