import cv2
import numpy as np
import os
from ocr import KTPExtractor

def visualize_ktp_detection(image_path, output_dir='output_viz'):
    """Process image and save visualization steps"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    extractor = KTPExtractor()
    
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 1. Get YOLO results directly for detection overlay
    results = extractor.seg_model(image, verbose=False)
    viz_overlay = image.copy()
    
    if results and len(results[0]) > 0:
        result = results[0]
        # Draw Mask
        if result.masks is not None:
            mask_coords = result.masks.xy[0].astype(np.int32)
            cv2.polylines(viz_overlay, [mask_coords], True, (0, 255, 0), 3)
            
            # Find and draw corners
            contour = mask_coords.astype(np.float32)
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) != 4:
                hull = cv2.convexHull(contour)
                epsilon = 0.02 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                
            for pt in approx:
                cv2.circle(viz_overlay, tuple(pt[0].astype(int)), 10, (0, 0, 255), -1)
        
        # Draw Bbox
        if result.boxes is not None:
            box = result.boxes.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(viz_overlay, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    # 2. Get the warped image using the extractor's method
    warped_image = extractor._detect_and_warp(image)
    
    # Save results
    base_name = os.path.basename(image_path).split('.')[0]
    overlay_path = os.path.join(output_dir, f"{base_name}_detection.jpg")
    warped_path = os.path.join(output_dir, f"{base_name}_warped.jpg")
    
    cv2.imwrite(overlay_path, viz_overlay)
    cv2.imwrite(warped_path, warped_image)
    
    print(f"Visualization results saved to '{output_dir}':")
    print(f"  - Detection overlay: {overlay_path}")
    print(f"  - Warped/Cropped result: {warped_path}")

if __name__ == "__main__":
    import sys
    test_image = 'images/ktp rifan.jpg'
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    visualize_ktp_detection(test_image)
