# 📊 **DATA FLOW VISUALIZATION**

## **Complete Pipeline Flow**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ACCURACY-IMPROVED DEDUPLICATION PIPELINE              │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PHASE 1:      │    │   PHASE 2:      │    │   PHASE 3:      │    │   PHASE 4:      │
│   IMAGE         │    │   WHASH         │    │   HYBRID        │    │   GLOBAL        │
│   DISCOVERY     │    │   PRE-GROUPING  │    │   VERIFICATION  │    │   REFINEMENT    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PHASE 5:      │    │   PHASE 6:      │    │   PHASE 7:      │    │   PHASE 8:      │
│   LOCAL         │    │   REPORT        │    │   CSV           │    │   FINAL         │
│   VERIFICATION  │    │   GENERATION    │    │   OUTPUT        │    │   RESULTS       │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## **Detailed Phase Breakdown**

### **Phase 1: Image Discovery**
```
Azure Blob Storage
        │
        ▼
┌─────────────────┐
│ AzureBlobManager│
│ .list_blobs()   │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Image Path List │
│ List[str]       │
└─────────────────┘
```

### **Phase 2: WHash Pre-Grouping**
```
Image Path List
        │
        ▼
┌─────────────────┐
│ EnhancedWHash   │
│ Deduplicator    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Multi-Scale     │
│ Hash Computation│
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ LSH Grouping    │
│ (Union-Find)    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ WHash Groups    │
│ List[List[str]] │
└─────────────────┘
```

### **Phase 3: Hybrid Verification**
```
WHash Groups
        │
        ▼
┌─────────────────┐
│ HybridSimilarity│
│ Calculator      │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Individual      │
│ Similarity      │
│ Measures        │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ WHash + SSIM +  │
│ Color Similarity│
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Weighted        │
│ Overall Score   │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Verified Groups │
│ List[List[str]] │
└─────────────────┘
```

### **Phase 4: Global Refinement**
```
Verified Groups
        │
        ▼
┌─────────────────┐
│ Cross-Group     │
│ Analysis        │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Group Merging   │
│ (if similar)    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Refined Groups  │
│ List[List[str]] │
└─────────────────┘
```

### **Phase 5: Local Verification**
```
Refined Groups
        │
        ▼
┌─────────────────┐
│ Within-Group    │
│ Verification    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Group Splitting │
│ (if needed)     │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Final Groups    │
│ List[List[str]] │
└─────────────────┘
```

### **Phase 6: Report Generation**
```
Final Groups
        │
        ▼
┌─────────────────┐
│ create_report() │
│ Method          │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Data Structure  │
│ Preparation     │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Pandas          │
│ DataFrame       │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ CSV File        │
│ image_report.csv│
└─────────────────┘
```

## **Data Structure Flow**

### **Input Data**
```
┌─────────────────┐
│ Image Paths     │
│ List[str]       │
│                 │
│ Example:        │
│ ['Image_Dedup_  │
│  Project/.../   │
│  image1.jpg',   │
│  'Image_Dedup_  │
│  Project/.../   │
│  image2.jpg']   │
└─────────────────┘
```

### **Intermediate Data**
```
┌─────────────────┐
│ WHash Hashes    │
│ Dict[str,       │
│  np.ndarray]    │
│                 │
│ Example:        │
│ {'image1.jpg':  │
│  [1,0,1,0...],  │
│  'image2.jpg':  │
│  [1,0,1,0...]}  │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ WHash Groups    │
│ List[List[str]] │
│                 │
│ Example:        │
│ [['image1.jpg', │
│   'image2.jpg'],│
│  ['image3.jpg']]│
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Similarity      │
│ Scores          │
│ Dict[Tuple[str, │
│  str], float]   │
│                 │
│ Example:        │
│ {('image1.jpg', │
│   'image2.jpg'):│
│  0.85}          │
└─────────────────┘
```

### **Output Data**
```
┌─────────────────┐
│ Final Groups    │
│ List[List[str]] │
│                 │
│ Example:        │
│ [['image1.jpg', │
│   'image2.jpg'],│
│  ['image3.jpg']]│
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ CSV Report      │
│ image_report.csv│
│                 │
│ Columns:        │
│ - Image Path    │
│ - Quality Score │
│ - Group ID      │
│ - Group Size    │
│ - Status        │
│ - Similarity    │
│   Score         │
└─────────────────┘
```

## **Similarity Calculation Flow**

### **Hybrid Similarity Computation**
```
Image 1 ──────────────┐
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                Hybrid Similarity Calculator             │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   WHash     │  │    SSIM     │  │   Color     │     │
│  │ Similarity  │  │ Similarity  │  │ Similarity  │     │
│  │             │  │             │  │             │     │
│  │ Weight: 0.15│  │ Weight: 0.60│  │ Weight: 0.25│     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │                 │                 │          │
│         ▼                 ▼                 ▼          │
│  ┌─────────────────────────────────────────────────────┐│
│  │           Weighted Overall Score                    ││
│  │  overall = 0.15*whash + 0.60*ssim + 0.25*color     ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
Image 2 ──────────────┘
```

### **WHash Similarity Details**
```
Image ──► Multi-Scale Hashing ──► LSH Grouping ──► Similarity Score
  │              │                    │                    │
  ▼              ▼                    ▼                    ▼
┌─────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│Input│    │Scale Factors│    │Union-Find   │    │Hamming      │
│Image│    │[0.5,0.75,   │    │Algorithm    │    │Distance     │
│     │    │ 1.0,1.25,   │    │             │    │Comparison   │
│     │    │ 1.5,2.0]    │    │             │    │             │
└─────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### **SSIM Similarity Details**
```
Image ──► Preprocessing ──► SSIM Computation ──► Similarity Score
  │            │                    │                    │
  ▼            ▼                    ▼                    ▼
┌─────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│Input│    │Resize +     │    │Scikit-Image │    │Structural   │
│Image│    │CLAHE        │    │SSIM         │    │Similarity   │
│     │    │Normalization│    │Computation  │    │Score        │
└─────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### **Color Similarity Details**
```
Image ──► Color Space ──► Histogram ──► Cosine ──► Similarity Score
  │           │              │           │            │
  ▼           ▼              ▼           ▼            ▼
┌─────┐    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│Input│    │HSV Color    │ │16-bin       │ │Cosine       │ │Color        │
│Image│    │Space        │ │Histograms   │ │Similarity   │ │Similarity   │
│     │    │Conversion   │ │(H,S,V)      │ │Computation  │ │Score        │
└─────┘    └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

## **Performance Optimization Flow**

### **Caching Strategy**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Similarity    │    │   Image         │    │   Blob List     │
│   Cache         │    │   Cache         │    │   Cache         │
│   (LRU)         │    │   (Bounded)     │    │   (Persistent)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                Performance Optimization                     │
│  - Reduce redundant computations                           │
│  - Minimize Azure API calls                                │
│  - Accelerate image loading                                │
└─────────────────────────────────────────────────────────────┘
```

### **Parallel Processing**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Multi-        │    │   Batch         │    │   Rate          │
│   Threading     │    │   Processing    │    │   Limiting      │
│   (Concurrent)  │    │   (Simultaneous)│    │   (Azure API)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                Scalability & Performance                    │
│  - Handle large datasets efficiently                        │
│  - Optimize resource utilization                            │
│  - Maintain system stability                                │
└─────────────────────────────────────────────────────────────┘
```

## **Error Handling & Fallbacks**

### **Failure Recovery Flow**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Image         │    │   Processing    │    │   Verification  │
│   Loading       │    │   Failures      │    │   Failures      │
│   Failures      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Retry +       │    │   Fallback to   │    │   Include by    │
│   Rate Limiting │    │   Single Groups │    │   Default       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                Graceful Degradation                         │
│  - Continue processing despite individual failures          │
│  - Maintain data integrity                                  │
│  - Provide meaningful results                               │
└─────────────────────────────────────────────────────────────┘
```

This visualization shows the complete data flow from Azure blob storage through multiple stages of analysis, verification, and refinement to produce a comprehensive CSV report of duplicate groups. Each phase builds upon the previous one, with sophisticated error handling and performance optimizations throughout the pipeline.
