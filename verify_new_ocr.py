import json
import os
from ocr import KTPExtractor

import cv2
import shutil

def verify_extraction(image_path):
    print(f"Verifying extraction for: {image_path}")
    
    # Create or clean output directory for crops
    output_dir = "debug_crops"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = KTPExtractor()
    original_image = cv2.imread(image_path)
    # Create a copy for visualization
    vis_image = original_image.copy()
    
    try:
        result = extractor.extract(image_path)
        print("\n--- EXTRACTION RESULTS ---")
        print(json.dumps(result['fields'], indent=2))
        
        # Save crops
        print(f"\nSaving field crops to {output_dir}/...")
        for field_name, data in result['fields'].items():
            if 'bbox' in data:
                bbox = data['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Apply same padding as in ocr.py for visual consistency: 20px horizontal, 5px vertical
                h_padding = 100
                v_padding = 20
                h, w = original_image.shape[:2]
                x1 = max(0, x1 - h_padding)
                y1 = max(0, y1 - v_padding)
                x2 = min(w, x2 + h_padding)
                y2 = min(h, y2 + v_padding)
                
                crop = original_image[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_path = os.path.join(output_dir, f"{field_name}.jpg")
                    cv2.imwrite(crop_path, crop)
                    print(f" Saved: {crop_path}")
                
                # Draw bounding box on visualization image
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add label
                label_text = f"{field_name} ({data['confidence']:.2f})"
                cv2.putText(vis_image, label_text, (x1, max(0, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save visualization image
        vis_path = os.path.join(output_dir, "visualized_extraction.jpg")
        cv2.imwrite(vis_path, vis_image)
        print(f"\nSaved full visualization to: {vis_path}")
        
        print("\n--- PERFORMANCE ---")
        print(json.dumps(result['performance'], indent=2))
        
        print(f"\nOverall Confidence: {result['confidence_score']:.4f}")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test with an existing image if available
    test_img = 'images/ktp-1.jpg'
    if os.path.exists(test_img):
        verify_extraction(test_img)
    else:
        print(f"Test image {test_img} not found.")
