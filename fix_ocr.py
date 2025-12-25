
import os

with open('ocr.py', 'r') as f:
    lines = f.readlines()

# Line 482 is the end of _calculate_overall_confidence
clean_lines = lines[:482]

with open('ocr.py', 'w') as f:
    f.writelines(clean_lines)
    f.write("\nif __name__ == '__main__':\n")
    f.write("    import json\n")
    f.write("    from ocr import KTPExtractor\n")
    f.write("    extractor = KTPExtractor()\n")
    f.write("    image_path = r'C:/Users/Probuddy/.gemini/antigravity/brain/c8b818c7-b78e-4bb8-a969-d879d81fb039/uploaded_image_1766689876297.jpg'\n")
    f.write("    result = extractor.extract(image_path)\n")
    f.write("    print(json.dumps(result, indent=2))\n")
