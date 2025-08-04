# Image Deduplication Project - Project Structure

## 📁 **Project Overview**

This project implements a memory-efficient 4-stage image deduplication pipeline for processing 3M+ images from Azure Blob Storage.

## 🏗️ **Directory Structure**

```
Image_Dedup_Project/
├── 📁 modules/                          # Core application modules
│   ├── feature_extraction.py           # Feature computation (wavelet, global, local)
│   ├── memory_efficient_deduplication.py # Main deduplication pipeline
│   ├── feature_cache.py                # Feature caching system
│   ├── azure_utils.py                  # Azure Blob Storage utilities
│   ├── distributed_processor.py        # Distributed processing utilities
│   └── deduplication.py                # Legacy deduplication logic
│
├── 📁 tests/                           # Test suite
│   ├── 📁 performance/                 # Performance and integration tests
│   │   ├── test_azure_image_list.py    # Azure image list testing
│   │   ├── test_large_dataset.py       # Large dataset performance
│   │   ├── test_ondemand_features.py   # On-demand feature computation
│   │   ├── test_diverse_dataset.py     # Diverse image testing
│   │   ├── test_color_verification_direct.py # Color verification testing
│   │   ├── test_small_dataset.py       # Small dataset testing
│   │   ├── test_comprehensive.py       # Comprehensive testing
│   │   ├── test_multithreading_performance.py # Multithreading tests
│   │   ├── test_memory_efficiency.py   # Memory efficiency tests
│   │   ├── test_deduplication_process.py # Process testing
│   │   ├── test_pipeline_connections.py # Pipeline connection tests
│   │   ├── test_memory_efficient_loading.py # Memory loading tests
│   │   ├── test_color_verification_fix.py # Color fix testing
│   │   ├── test_fixes.py               # General fixes testing
│   │   ├── test_pipeline.py            # Pipeline testing
│   │   └── test_gpu.py                 # GPU testing
│   │
│   ├── 📁 unit/                        # Unit tests (to be added)
│   └── 📁 existing/                    # Existing test files
│
├── 📁 docs/                            # Documentation
│   ├── 📁 technical/                   # Technical documentation
│   │   ├── README.md                   # Main project README
│   │   ├── COLOR_VERIFICATION_FIX.md   # Color verification fix details
│   │   ├── COLOR_VERIFICATION_FIX_SUMMARY.md # Fix summary
│   │   ├── AZURE_MEMORY_FIXES.md       # Azure memory optimization
│   │   ├── MEMORY_EFFICIENT_DEDUPLICATION.md # Memory efficiency guide
│   │   ├── MULTITHREADED_DEDUPLICATION.md # Multithreading guide
│   │   ├── HIERARCHICAL_DEDUPLICATION_IMPLEMENTATION.md # Implementation guide
│   │   ├── DEDUPLICATION_PROCESS.md    # Process documentation
│   │   ├── MULTITHREADING_IMPLEMENTATION_SUMMARY.md # Multithreading summary
│   │   ├── STAGE_ORDER_UPDATE_SUMMARY.md # Stage order updates
│   │   ├── SMALL_DATASET_TEST_VERIFICATION.md # Test verification
│   │   └── PIPELINE_FIXES.md           # Pipeline fixes
│   │
│   └── 📁 user_guides/                 # User guides (to be added)
│
├── 📁 scripts/                         # Utility scripts
│   └── 📁 utilities/                   # Utility scripts
│       ├── debug_azure_calls.py        # Azure debugging
│       ├── verify_fixes.py             # Fix verification
│       ├── manual_verification.py      # Manual verification
│       ├── simple_test.py              # Simple testing
│       ├── verify_color_stage.py       # Color stage verification
│       ├── download_weights.py         # Model weight download
│       └── make_clip_npy.py            # CLIP model preparation
│
├── 📁 features/                        # Feature storage
├── 📁 output/                          # Output results
├── 📁 deduplication_results/           # Deduplication results
├── 📁 test_features/                   # Test feature storage
├── 📁 env/                             # Virtual environment
├── 📁 .zencoder/                       # Zencoder configuration
├── 📁 __pycache__/                     # Python cache
│
├── 📄 main.py                          # Main entry point
├── 📄 pipeline.py                      # Pipeline orchestration
├── 📄 requirements.txt                 # Python dependencies
├── 📄 setup.py                         # Package setup
├── 📄 .gitignore                       # Git ignore rules
├── 📄 PROJECT_STRUCTURE.md             # This file
│
├── 📄 blob_cache_webvia.pkl            # Azure blob cache (108MB)
├── 📄 azure_blob_list.json             # Azure blob list (115MB)
├── 📄 class_clip.npy                   # CLIP class embeddings
├── 📄 pipeline_progress_*.json         # Pipeline progress tracking
│
└── 📁 tests/                           # Legacy test directory
```

## 🔧 **Core Components**

### **1. Memory-Efficient Deduplication Pipeline**
- **File**: `modules/memory_efficient_deduplication.py`
- **Purpose**: 4-stage deduplication with on-demand feature computation
- **Stages**: Wavelet → Color → Global → Local → Quality

### **2. Feature Extraction**
- **File**: `modules/feature_extraction.py`
- **Purpose**: Compute wavelet, global, and local features
- **Models**: EfficientNet-B7, CLIP, LoFTR

### **3. Azure Integration**
- **File**: `modules/azure_utils.py`
- **Purpose**: Azure Blob Storage operations with rate limiting
- **Features**: Concurrent downloads, caching, error handling

### **4. Feature Caching**
- **File**: `modules/feature_cache.py`
- **Purpose**: Bounded feature cache for memory efficiency
- **Features**: LRU eviction, disk persistence

## 🧪 **Testing Strategy**

### **Performance Tests** (`tests/performance/`)
- **Azure Image List Tests**: Real Azure image processing
- **Large Dataset Tests**: Scalability validation
- **Memory Efficiency Tests**: Memory usage optimization
- **Color Verification Tests**: Stage 2 optimization validation

### **Unit Tests** (`tests/unit/`)
- Individual component testing (to be implemented)
- Feature extraction validation
- Cache system testing
- Azure utility testing

## 📚 **Documentation**

### **Technical Docs** (`docs/technical/`)
- **Implementation Guides**: Detailed technical documentation
- **Fix Documentation**: Problem analysis and solutions
- **Performance Analysis**: Optimization strategies
- **Process Documentation**: Pipeline workflow details

### **User Guides** (`docs/user_guides/`)
- Setup and installation guides (to be added)
- Usage examples and tutorials (to be added)
- Troubleshooting guides (to be added)

## 🛠️ **Utility Scripts**

### **Development Utilities** (`scripts/utilities/`)
- **Debugging Tools**: Azure call debugging, verification scripts
- **Setup Scripts**: Model weight downloads, CLIP preparation
- **Testing Utilities**: Manual verification, simple testing

## 🚀 **Quick Start**

1. **Setup Environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # or env\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Run Performance Test**:
   ```bash
   python tests/performance/test_azure_image_list.py
   ```

3. **Run Full Pipeline**:
   ```bash
   python main.py
   ```

## 📊 **Performance Metrics**

### **Current Performance (Azure Test Results)**
- **Processing Rate**: 5.7 images/second
- **Memory Usage**: 1031 MB peak
- **Memory Efficiency**: 100% features freed
- **Estimated 3M Processing**: ~146 hours (6 days)
- **Azure Images Found**: 689,000+ images

### **Optimization Status**
- ✅ **On-Demand Feature Computation**: Working correctly
- ✅ **Memory-Efficient Processing**: 100% feature cleanup
- ✅ **Azure Download Optimization**: Rate-limited concurrent downloads
- ✅ **Color Verification Fix**: 66% reduction in Azure downloads
- ⚠️ **3M Scale Memory**: Needs optimization (60TB estimated)

## 🔄 **Recent Improvements**

1. **Color Verification Optimization**: 66% reduction in Azure downloads
2. **On-Demand Feature Computation**: No pre-computed features needed
3. **Memory Management**: 100% feature cleanup after processing
4. **Azure Integration**: Robust error handling and rate limiting
5. **Project Organization**: Clean, maintainable structure

## 📈 **Next Steps**

1. **Memory Optimization**: Reduce memory per image for 3M scale
2. **GPU Acceleration**: Implement GPU processing for faster computation
3. **Parallel Processing**: Multi-node distributed processing
4. **Unit Tests**: Comprehensive component testing
5. **User Documentation**: Setup and usage guides 