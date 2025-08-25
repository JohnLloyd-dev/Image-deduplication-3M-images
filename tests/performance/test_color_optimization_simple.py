#!/usr/bin/env python3
"""
Simple Test Script for ColorOptimizedDeduplicator

This script provides a quick way to test the core functionality
of the ColorOptimizedDeduplicator without running the full test suite.

Usage:
    python tests/performance/test_color_optimization_simple.py
"""

import os
import sys
import tempfile
import shutil
import logging
import numpy as np
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.color_optimized_deduplicator import ColorOptimizedDeduplicator
from modules.feature_cache import BoundedFeatureCache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic functionality of ColorOptimizedDeduplicator"""
    logger.info("🧪 Testing basic ColorOptimizedDeduplicator functionality...")
    
    # Create temporary test directory
    test_dir = tempfile.mkdtemp(prefix="simple_test_")
    try:
        # Initialize feature cache
        feature_cache = BoundedFeatureCache(max_size=100)
        
        # Test 1: Basic initialization
        logger.info("Testing basic initialization...")
        dedup = ColorOptimizedDeduplicator(
            feature_cache=feature_cache,
            color_clusters=5,  # Small number for testing
            batch_size=3
        )
        
        assert dedup.color_clusters == 5
        assert dedup.batch_size == 3
        assert dedup.adaptive_thresholding == True
        assert dedup.parallel_processing == True
        logger.info("✅ Basic initialization passed")
        
        # Test 2: Context manager
        logger.info("Testing context manager...")
        with dedup as d:
            assert d is dedup
            assert d.color_groups is not None
            assert d.color_model is not None
        logger.info("✅ Context manager passed")
        
        # Test 3: Configuration validation
        logger.info("Testing configuration validation...")
        custom_dedup = ColorOptimizedDeduplicator(
            feature_cache=feature_cache,
            color_clusters=10,
            batch_size=5,
            adaptive_thresholding=False,
            parallel_processing=False
        )
        
        assert custom_dedup.color_clusters == 10
        assert custom_dedup.batch_size == 5
        assert custom_dedup.adaptive_thresholding == False
        assert custom_dedup.parallel_processing == False
        logger.info("✅ Configuration validation passed")
        
        # Test 4: Memory stats initialization
        logger.info("Testing memory stats initialization...")
        assert 'color_groups_created' in dedup.memory_stats
        assert 'color_group_sizes' in dedup.memory_stats
        assert 'cache_hits' in dedup.memory_stats
        assert 'cache_misses' in dedup.memory_stats
        logger.info("✅ Memory stats initialization passed")
        
        # Test 5: Resource cleanup
        logger.info("Testing resource cleanup...")
        dedup.release()
        assert dedup.color_groups is None
        assert dedup.color_model is None
        assert len(dedup.color_features_cache) == 0
        logger.info("✅ Resource cleanup passed")
        
        logger.info("🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    finally:
        # Clean up
        try:
            shutil.rmtree(test_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up test directory: {e}")

def test_feature_extraction():
    """Test feature extraction functionality"""
    logger.info("🧪 Testing feature extraction functionality...")
    
    try:
        # Create temporary test directory
        test_dir = tempfile.mkdtemp(prefix="feature_test_")
        
        # Create a simple test image
        img_data = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img_path = os.path.join(test_dir, "test_image.jpg")
        import cv2
        cv2.imwrite(img_path, img_data)
        
        # Initialize deduplicator
        feature_cache = BoundedFeatureCache(max_size=100)
        dedup = ColorOptimizedDeduplicator(feature_cache=feature_cache)
        
        # Test image loading
        logger.info("Testing image loading...")
        loaded_img = dedup._load_image_efficiently(img_path)
        assert loaded_img is not None
        assert loaded_img.shape == (32, 32, 3)
        assert loaded_img.dtype == np.uint8
        logger.info("✅ Image loading passed")
        
        # Test color feature extraction
        logger.info("Testing color feature extraction...")
        color_features = dedup._extract_compact_color_features(img_path)
        assert color_features is not None
        assert len(color_features) > 0
        assert color_features.dtype == np.float32
        logger.info("✅ Color feature extraction passed")
        
        # Test basic global feature extraction
        logger.info("Testing basic global feature extraction...")
        global_features = dedup._basic_global_feature_extraction(img_path)
        assert global_features is not None
        assert len(global_features) == 9  # 9 basic global features
        logger.info("✅ Basic global feature extraction passed")
        
        # Test basic local feature extraction
        logger.info("Testing basic local feature extraction...")
        local_features = dedup._basic_local_feature_extraction(img_path)
        assert local_features is not None
        assert len(local_features) == 5  # 5 basic local features
        logger.info("✅ Basic local feature extraction passed")
        
        # Test basic quality assessment
        logger.info("Testing basic quality assessment...")
        quality_score = dedup._basic_quality_assessment(img_path)
        assert quality_score is not None
        assert 0 <= quality_score <= 100
        logger.info("✅ Basic quality assessment passed")
        
        # Cleanup
        dedup.release()
        shutil.rmtree(test_dir)
        
        logger.info("🎉 All feature extraction tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature extraction test failed: {e}")
        return False

def test_adaptive_thresholds():
    """Test adaptive threshold functionality"""
    logger.info("🧪 Testing adaptive threshold functionality...")
    
    try:
        # Create temporary test directory
        test_dir = tempfile.mkdtemp(prefix="threshold_test_")
        
        # Create test images
        img_paths = []
        for i in range(5):
            img_data = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            img_path = os.path.join(test_dir, f"test_image_{i}.jpg")
            import cv2
            cv2.imwrite(img_path, img_data)
            img_paths.append(img_path)
        
        # Initialize deduplicator
        feature_cache = BoundedFeatureCache(max_size=100)
        dedup = ColorOptimizedDeduplicator(feature_cache=feature_cache)
        
        # Test adaptive thresholding enabled
        logger.info("Testing adaptive thresholding enabled...")
        thresholds = dedup._calculate_adaptive_thresholds(img_paths)
        assert 'global' in thresholds
        assert 'local' in thresholds
        assert 'wavelet' in thresholds
        
        # Verify threshold ranges
        assert 0.5 <= thresholds['global'] <= 0.7
        assert 0.55 <= thresholds['local'] <= 0.65
        assert 0.6 <= thresholds['wavelet'] <= 0.75
        logger.info("✅ Adaptive thresholding enabled passed")
        
        # Test adaptive thresholding disabled
        logger.info("Testing adaptive thresholding disabled...")
        dedup.adaptive_thresholding = False
        default_thresholds = dedup._calculate_adaptive_thresholds(img_paths)
        assert default_thresholds['global'] == 0.7
        assert default_thresholds['local'] == 0.65
        assert default_thresholds['wavelet'] == 0.75
        logger.info("✅ Adaptive thresholding disabled passed")
        
        # Cleanup
        dedup.release()
        shutil.rmtree(test_dir)
        
        logger.info("🎉 All adaptive threshold tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Adaptive threshold test failed: {e}")
        return False

def main():
    """Run all simple tests"""
    logger.info("🚀 Starting simple ColorOptimizedDeduplicator tests...")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Feature Extraction", test_feature_extraction),
        ("Adaptive Thresholds", test_adaptive_thresholds)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} PASSED")
        else:
            logger.error(f"❌ {test_name} FAILED")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"TEST SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {total - passed}")
    logger.info(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! ColorOptimizedDeduplicator is working correctly.")
        return True
    else:
        logger.error("⚠️  Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
