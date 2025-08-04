# Image Deduplication Process

## Overview

The hierarchical deduplication system uses a five-stage approach to identify duplicate images with maximum precision and efficiency:

1. **Wavelet Hash Grouping** (Fast, Coarse)
2. **Color-Based Verification** (Fast, Perceptual)
3. **Global Feature Refinement** (Moderate, Semantic)  
4. **Local Feature Verification** (Slow, Geometric)
5. **Quality-Based Best Selection** (Organization)

## Stage 1: Wavelet Hash Grouping

**Purpose**: Fast initial grouping using perceptual hashes

**Method**:
- Uses wavelet-based perceptual hashes (32-64 bits)
- Multi-band LSH (Locality Sensitive Hashing) for efficient similarity search
- Union-Find algorithm for connected component grouping
- Threshold: `wavelet_threshold` (default: 0.8)

**Output**: Initial groups of potentially similar images

**Performance**: O(n) with LSH optimization

```python
# Example wavelet hashes
image1_hash = [1, 0, 1, 1, 0, 0, 1, 0, ...]  # 32 bits
image2_hash = [1, 0, 1, 1, 0, 0, 1, 0, ...]  # Identical -> grouped together
```

## Stage 2: Global Feature Refinement

**Purpose**: Refine wavelet groups using deep learning features

**Method**:
- Uses global features from CLIP/EfficientNet (512-2048 dimensions)
- Cosine similarity computation between feature vectors
- Connected component analysis for transitive grouping
- Threshold: `global_threshold` (default: 0.85)

**Output**: More precise groups based on semantic similarity

**Performance**: O(n²) within each wavelet group

```python
# Example global features
image1_global = [0.1, 0.5, -0.3, 0.8, ...]  # 512-dim vector
image2_global = [0.11, 0.52, -0.29, 0.82, ...]  # Similar -> cosine_sim > 0.85
```

## Stage 2: Color-Based Verification

**Purpose**: Fast perceptual filtering using color analysis

**Method**:
- Two-stage color verification process
- Dominant color distance analysis
- Average pixel difference computation
- Histogram correlation analysis
- Adaptive thresholds based on content similarity
- Threshold: `color_threshold` (default: 0.85)

**Output**: Color-verified duplicate groups

**Performance**: O(n²) with image loading overhead

```python
# Color verification combines multiple metrics:
# 1. Dominant color distance (50% weight)
# 2. Average pixel difference (30% weight)  
# 3. Histogram correlation (20% weight)

color_similarity = (0.5 * dom_sim) + (0.3 * pixel_sim) + (0.2 * hist_sim)
```

## Stage 3: Global Feature Refinement

**Purpose**: Semantic content verification using deep learning features

**Method**:
- Uses deep learning global features (ResNet/EfficientNet-based)
- Cosine similarity comparison
- Handles semantic similarity and content variations
- Threshold: `global_threshold` (default: 0.85)

**Output**: Semantically refined duplicate groups

**Performance**: O(n²) for feature comparison

```python
# Example global features
image1_global = [0.1, 0.5, -0.3, 0.8, ...]  # 512-dim vector
image2_global = [0.11, 0.52, -0.29, 0.82, ...]  # Similar -> cosine_sim > 0.85
```

## Stage 4: Local Feature Verification

**Purpose**: Final geometric verification using local keypoint features

**Method**:
- Uses local keypoint descriptors (SIFT/ORB-like features)
- Descriptor matching with distance-based similarity
- Handles geometric variations and partial occlusions
- Threshold: `local_threshold` (default: 0.75)

**Output**: Geometrically verified duplicate groups

**Performance**: O(k²) where k is the number of keypoints

```python
# Example local features
image1_local = {
    'keypoints': [[100, 200], [150, 250], ...],
    'descriptors': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], ...]
}
```

## Stage 5: Quality-Based Best Selection

**Purpose**: Select the best image from each duplicate group

**Method**:
- Quality scoring based on feature richness
- Filename analysis for quality indicators
- Feature completeness assessment
- Best image selection and organization

**Output**: Organized results with best images identified

**Performance**: O(n) linear scan

## Hierarchical Filtering

The process progressively filters candidates through 5 stages:

```
Input: 10,000 images
├── Stage 1 (Wavelet): 500 groups (5,000 potential duplicates)
├── Stage 2 (Color): 300 groups (3,000 color-verified duplicates)  
├── Stage 3 (Global): 200 groups (2,000 semantically verified duplicates)
├── Stage 4 (Local): 150 groups (1,500 geometrically verified duplicates)
└── Stage 5 (Quality): Best images selected from 150 groups

Result: 1,500 duplicates + 8,500 unique images + 150 best images
```

## Quality-Based Best Selection

For each duplicate group, the system selects the "best" image based on:

1. **Global Feature Magnitude**: Higher magnitude indicates richer features
2. **Filename Length**: Longer names often indicate higher quality versions
3. **Feature Completeness**: Images with all feature types available

```python
def select_best_image(group):
    best_score = -1
    best_image = group[0]
    
    for image in group:
        score = compute_quality_score(image)
        if score > best_score:
            best_score = score
            best_image = image
    
    return best_image
```

## Thresholds and Tuning

### Default Thresholds
- **Wavelet Threshold**: 0.8 (80% bit similarity)
- **Global Threshold**: 0.85 (85% cosine similarity)
- **Local Threshold**: 0.75 (75% descriptor match rate)

### Tuning Guidelines

**High Precision (Fewer False Positives)**:
```python
deduplicator = HierarchicalDeduplicator(
    wavelet_threshold=0.9,   # More strict
    global_threshold=0.9,    # More strict
    local_threshold=0.8      # More strict
)
```

**High Recall (Fewer False Negatives)**:
```python
deduplicator = HierarchicalDeduplicator(
    wavelet_threshold=0.7,   # More lenient
    global_threshold=0.8,    # More lenient
    local_threshold=0.7      # More lenient
)
```

## Performance Characteristics

### Time Complexity
- **Stage 1**: O(n) with LSH
- **Stage 2**: O(n² × g) where g is avg group size
- **Stage 3**: O(n² × k) where k is avg keypoints per image

### Memory Usage
- **Features**: ~10MB per 1000 images
- **Similarity Matrix**: O(n²) for large groups
- **LSH Buckets**: O(n) hash storage

### Scalability
- **Small datasets** (< 1K images): All stages fast
- **Medium datasets** (1K-10K images): Stage 2 becomes bottleneck
- **Large datasets** (> 10K images): Use batch processing

## Output Structure

### Duplicate Groups
```python
duplicate_groups = [
    ['image1.jpg', 'image1_copy.jpg', 'image1_variant.jpg'],  # Group 1: 3 duplicates
    ['image5.jpg', 'image5_copy.jpg'],                        # Group 2: 2 duplicates
    # ... more groups
]
```

### Similarity Scores
```python
similarity_scores = {
    ('image1.jpg', 'image1_copy.jpg'): 0.95,      # Very similar
    ('image1.jpg', 'image1_variant.jpg'): 0.87,   # Similar
    ('image5.jpg', 'image5_copy.jpg'): 0.92,      # Very similar
}
```

### Final Report
```csv
Image Path,Quality Score,Group ID,Group Size,Status,Avg Color Correlation,Dominant Colors
image1.jpg,95.2,1,3,Best,0.89,5
image1_copy.jpg,87.1,1,3,Duplicate,0.89,5
image1_variant.jpg,82.3,1,3,Duplicate,0.89,5
image5.jpg,91.8,2,2,Best,0.92,4
image5_copy.jpg,88.4,2,2,Duplicate,0.92,4
```

## Error Handling

The system handles various edge cases:

- **Missing Features**: Images without certain feature types are skipped for that stage
- **Invalid Features**: Corrupted or malformed features are logged and ignored
- **Empty Groups**: Single-image groups are not considered duplicates
- **Memory Limits**: Large similarity matrices are processed in batches

## Monitoring and Logging

The process provides detailed logging:

```
INFO - Step 1: Grouping images by wavelet hash...
INFO - Found 245 initial wavelet groups
INFO - Step 2: Refining groups using global features...
INFO - Step 3: Final verification using local features...
INFO - Hierarchical deduplication complete:
INFO - - Total input images: 1000
INFO - - Initial wavelet groups: 245
INFO - - Refined groups after global filtering: 89
INFO - - Final duplicate groups: 67
INFO - - Total duplicate images: 234
INFO - - Unique images (no duplicates): 766
INFO - - Deduplication efficiency: 23.4% duplicates found
```

## Best Practices

1. **Feature Quality**: Ensure high-quality feature extraction
2. **Threshold Tuning**: Adjust thresholds based on your dataset characteristics
3. **Batch Processing**: Use batching for large datasets (>10K images)
4. **Memory Management**: Monitor memory usage during processing
5. **Validation**: Manually verify results on a sample to tune thresholds
6. **Backup**: Keep original images before running deduplication

## Integration with Pipeline

The deduplication process integrates seamlessly with the main pipeline:

```python
# Extract features
features = extract_features_from_images(image_paths)

# Run deduplication
deduplicator = Deduplicator(device="cuda")
duplicate_groups, similarity_scores = deduplicator.deduplicate(
    image_paths=image_paths,
    features=features,
    output_dir="results"
)

# Generate report
report_path = deduplicator.create_report(
    duplicate_groups=duplicate_groups,
    similarity_scores=similarity_scores,
    output_dir="results"
)

# Copy organized results to Azure
copy_results_to_azure(report_path, duplicate_groups)
```

This hierarchical approach ensures both high accuracy and reasonable performance for large-scale image deduplication tasks.