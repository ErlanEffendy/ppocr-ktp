import json
from npwp_extractor import NPWPExtractor
import os

def test_npwp_extraction():
    extractor = NPWPExtractor()
    image_path = 'images/npwp/NPWP.png'
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return
        
    print(f"Testing NPWP extraction on {image_path}...")
    result = extractor.extract(image_path)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_npwp_extraction()
