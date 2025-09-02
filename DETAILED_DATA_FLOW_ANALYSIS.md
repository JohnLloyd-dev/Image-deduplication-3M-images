# 🔄 **DETAILED DATA FLOW ANALYSIS**

## **Complete Data Flow in the Accuracy-Improved Deduplication Pipeline**

### **Phase 1: Image Discovery & Loading**
```
Azure Blob Storage → AzureBlobManager → Image Path List
```

**Detailed Flow:**
1. **Azure Blob Discovery**: `AzureBlobManager.list_blobs('Image_Dedup_Project/TestEquity/CompleteImageDataset/')`
   - Uses SAS URL: `https://azwtewebsitecache.blob.core.windows.net/webvia?sp=rcwl&st=2025-05-05T17:40:16Z&se=2025-11-05T18:40:16Z&spr=https&sv=2024-11-04&sr=c&sig=6eTcYmq%2BeauVioFmi1bxh%2Bd4gDjvNdq54EufmpPSKYY%3D`
   - Filters for image extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, `.tif`, `.gif`, `.jfif`, `.pnm`, `.ppm`, `.pgm`, `.pbm`, `.heic`, `.avif`, `.ico`, `.svg`, `.raw`, `.cr2`, `.nef`, `.arw`, `.dng`, `.raf`, `.rw2`, `.pef`, `.srw`, `.orf`, `.x3f`, `.mrw`, `.mef`, `.iiq`
   - Returns: `List[str]` of image paths like `['Image_Dedup_Project/TestEquity/CompleteImageDataset/image1.jpg', ...]`

2. **Image Path Processing**: 
   - Input: `List[str]` of Azure blob paths
   - Output: Same list, ready for processing

### **Phase 2: Stage 1 - WHash Pre-Grouping**
```
Image Paths → EnhancedWHashDeduplicator → WHash Groups
```

**Detailed Flow:**
1. **Image Loading**: `EnhancedWHashDeduplicator._load_image(image_path)`
   - **Path Detection**: Checks if path starts with `Image_Dedup_Project/`
   - **Azure Loading**: If Azure path, calls `_load_azure_image(image_path)`
     - Uses `download_blob_to_memory(image_path, SAS_URL)` from `modules.azure_utils`
     - Converts bytes to numpy array: `np.frombuffer(image_data, np.uint8)`
     - Decodes with OpenCV: `cv2.imdecode(nparr, cv2.IMREAD_COLOR)`
     - Converts BGR to RGB: `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`
   - **Local Loading**: If local path, uses `cv2.imread(image_path)`

2. **Multi-Scale Hash Computation**: `_compute_multi_scale_hashes(img)`
   - **Scale Factors**: `[0.5, 0.75, 1.0, 1.25, 1.5, 2.0]` (configurable)
   - **For Each Scale**:
     - Resize image: `cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)`
     - Compute wavelet hash: `_compute_wavelet_hash(resized_img)`
       - Convert to grayscale: `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`
       - Wavelet decomposition: `pywt.wavedec2(img, 'haar', level=3)`
       - Extract approximation coefficients: `coeffs[0]`
       - Resize to hash size: `cv2.resize(approx, (8, 8), interpolation=cv2.INTER_AREA)`
       - Compute binary hash: `(approx > np.median(approx)).astype(np.uint8)`
       - Flatten: `whash.flatten()`
   - **Primary Hash Selection**: Uses hash at scale 1.0 (original size) as primary

3. **LSH Grouping**: `_lsh_grouping(whashes)`
   - **LSH Parameters**: 
     - Bands: 8 (configurable)
     - Rows per band: 3 (configurable)
     - Similarity threshold: 0.60 (configurable)
   - **Bucket Creation**: For each image hash, create LSH buckets
     - For each band: `for band in range(8)`
     - Extract band hash: `band_hash = tuple(whash[start:end])`
     - Create bucket ID: `(band, band_hash)`
   - **Union-Find Algorithm**: `_find_connected_components(whashes, lsh_buckets)`
     - Initialize parent and rank arrays
     - Union images that share LSH buckets
     - Group connected components
   - **Output**: `List[List[str]]` of WHash groups

### **Phase 3: Stage 2 - Hybrid Similarity Verification**
```
WHash Groups → HybridSimilarityCalculator → Verified Groups
```

**Detailed Flow:**
1. **Group Processing**: For each WHash group with >1 image
   - **Anchor Selection**: Use first image as anchor
   - **Candidate Verification**: For each remaining image in group

2. **Hybrid Similarity Computation**: `HybridSimilarityCalculator.compute_hybrid_similarity(img1_path, img2_path)`
   - **Cache Check**: `_get_cache_key(img1_path, img2_path)`
     - Sort paths: `tuple(sorted([img1_path, img2_path]))`
     - Generate hash: `hash(sorted_paths)`
   - **Image Loading**: Load both images using `_load_image()`
   - **Individual Similarity Measures**:

   **A. WHash Similarity**: `_compute_whash_similarity(img1, img2)`
   - Compute multi-scale hashes for both images
   - Compare hashes: `_compare_multi_scale_hashes(hashes1, hashes2)`
   - Return maximum similarity across all scale combinations
   - Weight: 0.15 (configurable)

   **B. SSIM Similarity**: `StructuralSimilarity.compute_ssim(img1, img2)`
   - **Preprocessing**: `_preprocess_image(img)`
     - Resize with aspect ratio: `_resize_with_aspect_ratio(img)`
     - Normalize illumination: `_normalize_illumination(img)`
       - Convert to LAB: `cv2.cvtColor(img, cv2.COLOR_RGB2LAB)`
       - Apply CLAHE: `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`
   - **SSIM Computation**: `_compute_ssim_scikit(img1, img2)`
     - Convert to grayscale: `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`
     - Compute SSIM: `ssim(img1, img2, data_range=img2.max() - img2.min(), win_size=11, gaussian_weights=True, sigma=1.5)`
   - Weight: 0.60 (configurable)

   **C. Color Similarity**: `_compute_color_similarity(img1, img2)`
   - Resize to 64x64: `cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)`
   - Convert to HSV: `cv2.cvtColor(img, cv2.COLOR_RGB2HSV)`
   - Compute histograms: `cv2.calcHist([hsv], [0], None, [16], [0, 180])`
   - Normalize histograms: `hist / (hist.sum() + 1e-8)`
   - Compute cosine similarity: `np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))`
   - Weighted average: `0.5 * h_similarity + 0.3 * s_similarity + 0.2 * v_similarity`
   - Weight: 0.25 (configurable)

3. **Overall Score Computation**: `_compute_weighted_score(similarity_scores)`
   - Weighted sum: `overall_score = 0.15 * whash + 0.60 * ssim + 0.25 * color`
   - Threshold check: `overall_score >= 0.65` (configurable)

4. **Caching**: `_cache_result(cache_key, result)`
   - Store result in `similarity_cache`
   - Implement LRU eviction if cache size exceeds limit

### **Phase 4: Stage 3 - Global Feature Refinement**
```
Verified Groups → Cross-Group Analysis → Refined Groups
```

**Detailed Flow:**
1. **Cross-Group Analysis**: `_global_feature_refinement(groups)`
   - **Group Comparison**: For each pair of groups
   - **Sample Comparison**: `_should_merge_groups(group1, group2)`
     - Sample up to 3 images from each group
     - Compute hybrid similarity between samples
     - If any similarity >= threshold, groups should be merged
   - **Group Merging**: Merge similar groups into single groups
   - **Output**: Refined list of groups

### **Phase 5: Stage 4 - Local Feature Verification**
```
Refined Groups → Within-Group Refinement → Final Groups
```

**Detailed Flow:**
1. **Within-Group Verification**: `_local_feature_verification(groups)`
   - **Group Processing**: For each group with >1 image
   - **Re-verification**: Use hybrid similarity to verify within-group duplicates
   - **Group Splitting**: May split groups if local verification fails
   - **Output**: Final verified groups

### **Phase 6: Report Generation**
```
Final Groups → create_report() → CSV File
```

**Detailed Flow:**
1. **Data Preparation**: `create_report(duplicate_groups, similarity_scores, output_dir)`
   - **Group Processing**: For each group in `duplicate_groups`
   - **Best Image Selection**: First image in group becomes "Best"
   - **Duplicate Processing**: Remaining images become "Duplicate"

2. **CSV Structure Creation**:
   ```python
   data = []
   for group_idx, group in enumerate(duplicate_groups):
       best_image = group[0]
       group_size = len(group)
       
       # Best image entry
       data.append({
           'Image Path': best_image,
           'Quality Score': 1.0,
           'Group ID': group_idx + 1,
           'Group Size': group_size,
           'Status': 'Best',
           'Similarity Score': 1.0
       })
       
       # Duplicate entries
       for dup_image in group[1:]:
           similarity_score = similarity_scores.get((best_image, dup_image), 0.0)
           data.append({
               'Image Path': dup_image,
               'Quality Score': 0.8,
               'Group ID': group_idx + 1,
               'Group Size': group_size,
               'Status': 'Duplicate',
               'Similarity Score': similarity_score
           })
   ```

3. **DataFrame Creation**: `pd.DataFrame(data)`
   - Sort by Group ID and Status (Best first)
   - Format scores to 3 decimal places
   - Save to CSV: `df.to_csv(report_path, index=False)`

## **Data Structures & Flow**

### **Input Data**
- **Image Paths**: `List[str]` - Azure blob paths
- **Example**: `['Image_Dedup_Project/TestEquity/CompleteImageDataset/image1.jpg', ...]`

### **Intermediate Data**
- **WHash Hashes**: `Dict[str, np.ndarray]` - Image path → binary hash
- **WHash Groups**: `List[List[str]]` - Groups from LSH clustering
- **Similarity Scores**: `Dict[Tuple[str, str], float]` - Pairwise similarity scores
- **Verified Groups**: `List[List[str]]` - Groups after hybrid verification

### **Output Data**
- **Final Groups**: `List[List[str]]` - Final duplicate groups
- **CSV Report**: Structured data with columns:
  - `Image Path`: Path to image
  - `Quality Score`: Quality assessment (1.0 for best, 0.8 for duplicates)
  - `Group ID`: Which duplicate group this image belongs to
  - `Group Size`: How many images are in this group
  - `Status`: "Best" or "Duplicate"
  - `Similarity Score`: Actual similarity score from verification

## **Performance Optimizations**

### **Caching**
- **Similarity Cache**: LRU cache for computed similarities
- **Image Cache**: Bounded cache for loaded images
- **Blob List Cache**: Persistent cache for Azure blob listings

### **Parallel Processing**
- **Multi-threading**: Concurrent image loading and processing
- **Batch Processing**: Process multiple images simultaneously
- **Rate Limiting**: Control Azure API request rate

### **Memory Management**
- **On-demand Loading**: Load images only when needed
- **Immediate Cleanup**: Delete images after processing
- **Garbage Collection**: Force GC after major operations

## **Error Handling & Fallbacks**

### **Image Loading Failures**
- **Azure Failures**: Retry with exponential backoff
- **Decode Failures**: Skip problematic images
- **Network Issues**: Rate limiting and retry logic

### **Processing Failures**
- **WHash Failures**: Fallback to single-image groups
- **SSIM Failures**: Use correlation as fallback
- **Verification Failures**: Include images by default

### **Resource Management**
- **Memory Limits**: Bounded caches and immediate cleanup
- **Timeout Handling**: 30-second timeouts for Azure operations
- **Graceful Degradation**: Continue processing despite individual failures

## **Configuration Parameters**

### **WHash Parameters**
- `hash_size`: 8 (8x8 = 64 bits)
- `wavelet_level`: 3
- `wavelet_name`: 'haar'
- `scale_factors`: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
- `lsh_bands`: 8
- `lsh_rows_per_band`: 3
- `similarity_threshold`: 0.60

### **Hybrid Similarity Parameters**
- `weights`: {'structural': 0.6, 'color': 0.25, 'whash': 0.15}
- `thresholds`: {'global': 0.65, 'structural': 0.60, 'color': 0.55, 'whash': 0.60}
- `cache_size`: 10000

### **SSIM Parameters**
- `target_size`: (256, 256)
- `use_gpu`: True (if available)
- `enable_preprocessing`: True

### **Performance Parameters**
- `max_group_size`: 1000
- `verification_threshold`: 0.65
- `max_concurrent_downloads`: 100
- `max_retries`: 3

This detailed data flow shows how the accuracy-improved deduplication pipeline processes images from Azure blob storage through multiple stages of analysis, verification, and refinement to produce a comprehensive CSV report of duplicate groups.
