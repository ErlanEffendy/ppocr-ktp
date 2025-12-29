# OCR Script Performance Optimization Guide

## Current Performance Issues
Your script takes 15-30s per image. The bottlenecks are:

### 1. **Model Loading (Biggest Impact - ~5-10s per call)**
- **Problem**: PaddleOCR model was reloaded for every extraction
- **Solution**: Implemented singleton pattern to reuse the same model instance
- **Impact**: **First call unchanged, subsequent calls save 5-10s each**

### 2. **Preprocessing (Heavy Operations - ~2-5s)**
- **Old**: `cv2.fastNlMeansDenoisingColored()` is extremely slow
- **New**: `cv2.bilateralFilter()` - 5-10x faster with similar results
- **Impact**: **2-5s improvement**

### 3. **Image Resizing (Optional - 1-3s)**
- **Solution**: Added optional resize if image > 1920px (adjustable)
- **Impact**: **1-3s improvement if your images are large**
- **Note**: PaddleOCR doesn't need massive resolution; 1080p is usually sufficient

### 4. **Search Range Optimization (Microseconds)**
- **Old**: Searched 4 lines for each field value
- **New**: Reduced to 1-2 lines based on field type
- **Impact**: **Minimal but helps with memory and CPU**

### 5. **Regex Pre-compilation (Microseconds)**
- **Solution**: Cache cleaned canonical strings instead of recomputing
- **Impact**: **Negligible but slightly improves loop efficiency**

---

## Implemented Changes

### Change 1: Singleton Pattern for OCR Model
```python
class KTPExtractor:
    _ocr_instance = None  # Class variable
    
    def __init__(self):
        if KTPExtractor._ocr_instance is None:
            KTPExtractor._ocr_instance = PaddleOCR(...)
        self.ocr = KTPExtractor._ocr_instance
```

### Change 2: Replace Slow Denoising
```python
# OLD (SLOW - ~3-5 seconds)
image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# NEW (FAST - ~0.5-1 second)
image = cv2.bilateralFilter(image, 5, 75, 75)
```

### Change 3: Optional Image Resizing
```python
max_dimension = 1920
if w > max_dimension or h > max_dimension:
    scale = max_dimension / max(w, h)
    image = cv2.resize(image, (int(w * scale), int(h * scale)))
```

### Change 4: Reduced Search Ranges
```python
# OLD: search_range = 2 or 4 lines
# NEW: search_range = 1 or 2 lines
```

---

## Expected Performance Improvements

| Scenario | Before | After | Improvement |
|----------|--------|-------|-------------|
| **First call** | 15-30s | 15-25s | ~10% (model loads once) |
| **Subsequent calls** | 15-30s | 5-15s | **50-66% faster!** |
| **Large images (3000+px)** | 20-35s | 8-12s | **60-70% faster!** |
| **With GPU enabled** | 10-20s | 3-5s | **75% faster!** |

---

## Advanced Optimizations (If Needed)

### 1. **Enable GPU Acceleration**
If you have CUDA-capable GPU:
```python
self.ocr = PaddleOCR(
    use_gpu=True,  # Change to True
    lang='en'
)
```
**Impact**: Could provide 3-5x speedup on inference

### 2. **Aggressive Image Resizing**
If quality is not critical:
```python
max_dimension = 1280  # Smaller = faster, less quality
```

### 3. **Skip Preprocessing**
If images are already good quality:
```python
def extract(self, image_path):
    image = cv2.imread(image_path)
    # Skip preprocessing entirely
    result = self.ocr.predict(image)
```
**Impact**: Additional 1-2s savings but may reduce accuracy

### 4. **Batch Processing**
Process multiple images at once:
```python
results = [self.ocr.predict(img) for img in batch_images]
```

### 5. **Use FastNLP Model**
PaddleOCR has faster models available with slightly reduced accuracy

---

## Testing Your Improvements

Run this to see timing breakdown:
```bash
python -c "
from ocr import KTPExtractor
import time

extractor = KTPExtractor()
img_path = 'images/ktp-1.jpg'

# Warm-up (model loads)
start = time.time()
result = extractor.extract(img_path)
print(f'First call: {time.time()-start:.2f}s')
print('Breakdown:', result['performance'])

# Second call (should be much faster)
start = time.time()
result = extractor.extract(img_path)
print(f'Second call: {time.time()-start:.2f}s')
"
```

---

## Recommendations

1. **Start with these changes** - Already implemented, test them first
2. **If still slow (>10s)**: Enable GPU or reduce max_dimension
3. **If accuracy drops**: Increase bilateral filter strength (9 instead of 5)
4. **For production**: Monitor performance and adjust preprocessing based on real KTP images

---

## Debug Denoising Quality

If bilateral filter isn't good enough, try:
```python
# Option 1: Morphological operations (fast)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Option 2: Stronger bilateral (slower but better)
image = cv2.bilateralFilter(image, 9, 100, 100)

# Option 3: Weighted between fast & slow
image = cv2.bilateralFilter(image, 7, 80, 80)
```
