# ColorOptimizedDeduplicator Testing Guide

This directory contains comprehensive test suites for validating the `ColorOptimizedDeduplicator` implementation.

## Test Files

### 1. `test_color_optimization_comprehensive.py`
**Full test suite** that validates all implemented features:
- **15 test methods** covering all aspects of the implementation
- **Comprehensive validation** of all new features
- **Mock-based testing** for Azure and external dependencies
- **Resource management** and cleanup testing

### 2. `test_color_optimization_simple.py`
**Quick validation script** for core functionality:
- **3 main test categories** for rapid validation
- **Minimal dependencies** and fast execution
- **Basic functionality** verification
- **Perfect for CI/CD** and quick checks

## Running the Tests

### Prerequisites
```bash
# Install required dependencies
pip install -r requirements.txt

# Ensure you have the required packages
pip install opencv-python numpy scikit-learn
```

### Quick Test (Recommended for first run)
```bash
# Run the simple test suite
python tests/performance/test_color_optimization_simple.py
```

### Comprehensive Test Suite
```bash
# Run the full test suite
python tests/performance/test_color_optimization_comprehensive.py

# Or use pytest (if available)
python -m pytest tests/performance/test_color_optimization_comprehensive.py -v
```

### Individual Test Categories
```bash
# Test only basic functionality
python -c "
from tests.performance.test_color_optimization_simple import test_basic_functionality
test_basic_functionality()
"

# Test only feature extraction
python -c "
from tests.performance.test_color_optimization_simple import test_feature_extraction
test_feature_extraction()
"

# Test only adaptive thresholds
python -c "
from tests.performance.test_color_optimization_simple import test_adaptive_thresholds
test_adaptive_thresholds()
"
```

## What the Tests Validate

### ✅ Core Functionality
- **Initialization**: Proper parameter handling and defaults
- **Configuration**: Custom parameter validation
- **Context Manager**: Resource acquisition and release
- **Memory Management**: Proper cleanup and resource tracking

### ✅ Feature Extraction
- **Unified Interface**: Single method for multiple feature types
- **Image Loading**: Local and Azure image handling
- **Color Features**: Compact color feature extraction
- **Fallback Methods**: Basic feature extraction when models unavailable

### ✅ Advanced Features
- **Adaptive Thresholds**: Dynamic threshold calculation
- **Color Pre-grouping**: Color-based image clustering
- **Parallel Processing**: Multi-threading for large datasets
- **Caching Strategy**: Intelligent feature caching

### ✅ Integration
- **Pipeline Integration**: Full deduplication pipeline testing
- **Quality Selection**: Best image selection algorithms
- **Error Handling**: Robust error handling and recovery
- **Performance Metrics**: Comprehensive statistics and monitoring

## Test Output

### Successful Test Run
```
🚀 Starting simple ColorOptimizedDeduplicator tests...

==================================================
Running: Basic Functionality
==================================================
🧪 Testing basic ColorOptimizedDeduplicator functionality...
Testing basic initialization...
✅ Basic initialization passed
Testing context manager...
✅ Context manager passed
Testing configuration validation...
✅ Configuration validation passed
Testing memory stats initialization...
✅ Memory stats initialization passed
Testing resource cleanup...
✅ Resource cleanup passed
🎉 All basic functionality tests passed!
✅ Basic Functionality PASSED

==================================================
Running: Feature Extraction
==================================================
🧪 Testing feature extraction functionality...
Testing image loading...
✅ Image loading passed
Testing color feature extraction...
✅ Color feature extraction passed
Testing basic global feature extraction...
✅ Basic global feature extraction passed
Testing basic local feature extraction...
✅ Basic local feature extraction passed
Testing basic quality assessment...
✅ Basic quality assessment passed
🎉 All feature extraction tests passed!
✅ Feature Extraction PASSED

==================================================
Running: Adaptive Thresholds
==================================================
🧪 Testing adaptive threshold functionality...
Testing adaptive thresholding enabled...
✅ Adaptive thresholding enabled passed
Testing adaptive thresholding disabled...
✅ Adaptive thresholding disabled passed
🎉 All adaptive threshold tests passed!
✅ Adaptive Thresholds PASSED

==================================================
TEST SUMMARY
==================================================
Total tests: 3
Passed: 3
Failed: 0
Success rate: 100.0%
🎉 ALL TESTS PASSED! ColorOptimizedDeduplicator is working correctly.
```

### Failed Test Example
```
❌ Test failed: AssertionError: Expected 2000, got 1000
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure you're in the project root directory
cd /path/to/dev_6_23_original

# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. Missing Dependencies
```bash
# Install missing packages
pip install opencv-python numpy scikit-learn

# For GPU support (optional)
pip install torch torchvision
```

#### 3. Permission Issues
```bash
# Ensure write permissions for test directories
chmod 755 tests/performance/
```

#### 4. Memory Issues
```bash
# Reduce test image sizes for low-memory systems
# Edit test files to use smaller image dimensions (e.g., 16x16 instead of 64x64)
```

### Debug Mode
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG

# Run with more detailed output
python tests/performance/test_color_optimization_simple.py 2>&1 | tee test_output.log
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Test ColorOptimizedDeduplicator

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install opencv-python numpy scikit-learn
    
    - name: Run tests
      run: |
        python tests/performance/test_color_optimization_simple.py
        python tests/performance/test_color_optimization_comprehensive.py
```

## Performance Testing

### Large Dataset Testing
```bash
# Test with larger datasets (adjust parameters as needed)
python -c "
from modules.color_optimized_deduplicator import ColorOptimizedDeduplicator
from modules.feature_cache import BoundedFeatureCache

# Create large test dataset
import numpy as np
test_images = [f'/tmp/test_img_{i}.jpg' for i in range(1000)]

# Test performance
feature_cache = BoundedFeatureCache(max_size=10000)
dedup = ColorOptimizedDeduplicator(
    feature_cache=feature_cache,
    color_clusters=100,
    batch_size=100,
    parallel_processing=True,
    max_workers=4
)

# Run performance test
import time
start_time = time.time()
with dedup as d:
    groups = d._stage0_color_pre_grouping(test_images[:100])
end_time = time.time()

print(f'Processed 100 images in {end_time - start_time:.2f} seconds')
print(f'Created {len(groups)} color groups')
"
```

## Contributing

When adding new features to `ColorOptimizedDeduplicator`:

1. **Add tests** to both test files
2. **Update this README** with new test descriptions
3. **Ensure backward compatibility** with existing tests
4. **Test edge cases** and error conditions
5. **Validate performance** with realistic datasets

## Support

If you encounter issues with the tests:

1. Check the **Troubleshooting** section above
2. Verify all **dependencies** are installed
3. Ensure you're running from the **project root**
4. Check **file permissions** and paths
5. Review the **test output** for specific error messages

For additional help, refer to the main project documentation or create an issue in the project repository.
