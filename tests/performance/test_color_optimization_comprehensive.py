#!/usr/bin/env python3
"""
Comprehensive Test Suite for ColorOptimizedDeduplicator

This test suite validates all the key improvements implemented in the
ColorOptimizedDeduplicator, including:

1. Unified feature extraction interface
2. Proper Azure image loading
3. Adaptive threshold calculation
4. Parallel processing for large datasets
5. Enhanced caching strategy
6. Better progress reporting
7. Comprehensive resource management
8. Configurable parameters

Usage:
    python -m pytest tests/performance/test_color_optimization_comprehensive.py -v
    python tests/performance/test_color_optimization_comprehensive.py
"""

import os
import sys
import tempfile
import shutil
import time
import logging
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.color_optimized_deduplicator import ColorOptimizedDeduplicator, create_color_optimized_deduplicator
from modules.feature_cache import BoundedFeatureCache

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestColorOptimizedDeduplicator:
    """Comprehensive test suite for ColorOptimizedDeduplicator"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.test_dir = tempfile.mkdtemp(prefix="color_opt_test_")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test images
        self.test_images = self._create_test_images()
        
        # Initialize feature cache
        self.feature_cache = BoundedFeatureCache(max_size=1000)
        
        # Mock Azure utilities
        self.azure_patcher = patch('modules.color_optimized_deduplicator.download_blob_to_memory')
        self.mock_download = self.azure_patcher.start()
        self.mock_download.return_value = self._create_mock_image_data()
        
        # Mock feature extractor
        self.mock_feature_extractor = Mock()
        self.mock_feature_extractor.extract_global_features.return_value = np.random.rand(512)
        self.mock_feature_extractor.extract_local_features.return_value = np.random.rand(256)
        self.mock_feature_extractor.extract_wavelet_features.return_value = np.random.rand(128)
        self.mock_feature_extractor.extract_quality_score.return_value = 85.5
        
        logger.info(f"Test setup complete. Test directory: {self.test_dir}")
    
    def teardown_method(self):
        """Clean up test fixtures after each test method"""
        self.azure_patcher.stop()
        
        # Clean up test directory
        try:
            shutil.rmtree(self.test_dir)
            logger.info(f"Test directory cleaned up: {self.test_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up test directory: {e}")
    
    def _create_test_images(self, num_images=10):
        """Create test image files for testing"""
        test_images = []
        
        for i in range(num_images):
            # Create a simple test image using numpy
            img_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            
            # Add some color variations to test color grouping
            if i < 3:
                # Red-dominant images
                img_data[:, :, 0] = np.random.randint(200, 255, (64, 64))
                img_data[:, :, 1:] = np.random.randint(0, 100, (64, 64, 2))
            elif i < 6:
                # Green-dominant images
                img_data[:, :, 1] = np.random.randint(200, 255, (64, 64))
                img_data[:, :, [0, 2]] = np.random.randint(0, 100, (64, 64, 2))
            else:
                # Blue-dominant images
                img_data[:, :, 2] = np.random.randint(200, 255, (64, 64))
                img_data[:, :, :2] = np.random.randint(0, 100, (64, 64, 2))
            
            # Save image
            img_path = os.path.join(self.test_dir, f"test_image_{i:02d}.jpg")
            import cv2
            cv2.imwrite(img_path, img_data)
            test_images.append(img_path)
        
        logger.info(f"Created {len(test_images)} test images")
        return test_images
    
    def _create_mock_image_data(self):
        """Create mock image data for Azure testing"""
        # Create a simple 64x64 RGB image as bytes
        img_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        import cv2
        _, buffer = cv2.imencode('.jpg', img_data)
        return buffer.tobytes()
    
    def test_initialization_and_configuration(self):
        """Test proper initialization and configuration"""
        logger.info("Testing initialization and configuration...")
        
        # Test with default parameters
        dedup = ColorOptimizedDeduplicator(feature_cache=self.feature_cache)
        
        assert dedup.color_clusters == 2000
        assert dedup.batch_size == 1000
        assert dedup.color_tolerance == 0.8
        assert dedup.min_group_size == 2
        assert dedup.max_group_size == 1000
        assert dedup.adaptive_thresholding == True
        assert dedup.parallel_processing == True
        assert dedup.max_workers > 0
        assert dedup.chunk_size == 1000
        
        # Test with custom parameters
        custom_params = {
            'color_clusters': 1000,
            'batch_size': 500,
            'color_tolerance': 0.9,
            'min_group_size': 5,
            'max_group_size': 500,
            'adaptive_thresholding': False,
            'parallel_processing': False,
            'max_workers': 4,
            'chunk_size': 500
        }
        
        dedup_custom = ColorOptimizedDeduplicator(
            feature_cache=self.feature_cache,
            **custom_params
        )
        
        for key, value in custom_params.items():
            assert getattr(dedup_custom, key) == value
        
        logger.info("✅ Initialization and configuration tests passed")
    
    def test_context_manager_functionality(self):
        """Test context manager entry and exit"""
        logger.info("Testing context manager functionality...")
        
        dedup = ColorOptimizedDeduplicator(feature_cache=self.feature_cache)
        
        # Test context manager
        with dedup as d:
            assert d is dedup
            assert d.color_groups is not None
            assert d.color_model is not None
        
        # Test that resources are released
        assert dedup.color_groups is None
        assert dedup.color_model is None
        
        logger.info("✅ Context manager functionality tests passed")
    
    def test_unified_feature_extraction(self):
        """Test unified feature extraction interface"""
        logger.info("Testing unified feature extraction interface...")
        
        dedup = ColorOptimizedDeduplicator(feature_cache=self.feature_cache)
        dedup.feature_extractor = self.mock_feature_extractor
        
        # Test extracting multiple feature types
        features = dedup.extract_features(self.test_images[0], ['color', 'global', 'local'])
        
        assert 'color' in features
        assert 'global' in features
        assert 'local' in features
        
        # Test that features are cached
        cached_features = dedup.extract_features(self.test_images[0], ['color', 'global'])
        assert 'color' in cached_features
        assert 'global' in cached_features
        
        # Verify cache statistics
        assert dedup.memory_stats['cache_hits'] > 0
        assert dedup.memory_stats['cache_misses'] > 0
        
        logger.info("✅ Unified feature extraction tests passed")
    
    def test_image_loading_functionality(self):
        """Test image loading for both local and Azure images"""
        logger.info("Testing image loading functionality...")
        
        dedup = ColorOptimizedDeduplicator(feature_cache=self.feature_cache)
        
        # Test local image loading
        local_img = dedup._load_image_efficiently(self.test_images[0])
        assert local_img is not None
        assert local_img.shape == (64, 64, 3)
        assert local_img.dtype == np.uint8
        
        # Test Azure image loading
        azure_url = "https://storage.blob.core.windows.net/container/test_image.jpg"
        azure_img = dedup._load_azure_image(azure_url)
        assert azure_img is not None
        assert azure_img.shape == (64, 64, 3)
        assert azure_img.dtype == np.uint8
        
        # Test Azure URL detection
        assert dedup._load_image_efficiently(azure_url) is not None
        
        logger.info("✅ Image loading functionality tests passed")
    
    def test_color_feature_extraction(self):
        """Test compact color feature extraction"""
        logger.info("Testing color feature extraction...")
        
        dedup = ColorOptimizedDeduplicator(feature_cache=self.feature_cache)
        
        # Test color feature extraction
        color_features = dedup._extract_compact_color_features(self.test_images[0])
        assert color_features is not None
        assert len(color_features) > 0
        assert color_features.dtype == np.float32
        
        # Test array-based extraction (for parallel processing)
        img_array = dedup._load_image_efficiently(self.test_images[0])
        array_features = dedup._extract_compact_color_features_from_array(img_array)
        assert array_features is not None
        assert len(array_features) > 0
        
        logger.info("✅ Color feature extraction tests passed")
    
    def test_adaptive_threshold_calculation(self):
        """Test adaptive threshold calculation"""
        logger.info("Testing adaptive threshold calculation...")
        
        dedup = ColorOptimizedDeduplicator(feature_cache=self.feature_cache)
        
        # Test with adaptive thresholding enabled
        thresholds = dedup._calculate_adaptive_thresholds(self.test_images[:5])
        assert 'global' in thresholds
        assert 'local' in thresholds
        assert 'wavelet' in thresholds
        
        # Verify threshold ranges
        assert 0.5 <= thresholds['global'] <= 0.7
        assert 0.55 <= thresholds['local'] <= 0.65
        assert 0.6 <= thresholds['wavelet'] <= 0.75
        
        # Test with adaptive thresholding disabled
        dedup.adaptive_thresholding = False
        default_thresholds = dedup._calculate_adaptive_thresholds(self.test_images[:5])
        assert default_thresholds['global'] == 0.7
        assert default_thresholds['local'] == 0.65
        assert default_thresholds['wavelet'] == 0.75
        
        logger.info("✅ Adaptive threshold calculation tests passed")
    
    def test_color_pre_grouping(self):
        """Test color-based pre-grouping functionality"""
        logger.info("Testing color pre-grouping functionality...")
        
        dedup = ColorOptimizedDeduplicator(
            feature_cache=self.feature_cache,
            color_clusters=3,  # Small number for testing
            batch_size=5
        )
        
        # Test sequential color pre-grouping
        color_groups = dedup._stage0_color_pre_grouping(self.test_images)
        assert len(color_groups) > 0
        
        # Verify group sizes
        for group in color_groups:
            assert len(group) >= dedup.min_group_size
            assert len(group) <= dedup.max_group_size
        
        # Test color group creation
        cluster_labels = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
        groups = dedup._create_color_groups(self.test_images, cluster_labels)
        assert len(groups) > 0
        
        logger.info("✅ Color pre-grouping tests passed")
    
    def test_parallel_processing(self):
        """Test parallel processing functionality"""
        logger.info("Testing parallel processing functionality...")
        
        dedup = ColorOptimizedDeduplicator(
            feature_cache=self.feature_cache,
            parallel_processing=True,
            max_workers=2,
            chunk_size=3
        )
        
        # Test parallel color pre-grouping
        if len(self.test_images) >= dedup.chunk_size:
            parallel_groups = dedup._stage0_color_pre_grouping_parallel(self.test_images)
            assert len(parallel_groups) > 0
            
            # Verify parallel processing time is tracked
            assert dedup.memory_stats['parallel_processing_time'] > 0
        
        logger.info("✅ Parallel processing tests passed")
    
    def test_feature_extraction_fallbacks(self):
        """Test fallback feature extraction methods"""
        logger.info("Testing feature extraction fallbacks...")
        
        dedup = ColorOptimizedDeduplicator(feature_cache=self.feature_cache)
        
        # Test global feature extraction fallback
        global_features = dedup._basic_global_feature_extraction(self.test_images[0])
        assert global_features is not None
        assert len(global_features) == 9  # 9 basic global features
        
        # Test local feature extraction fallback
        local_features = dedup._basic_local_feature_extraction(self.test_images[0])
        assert local_features is not None
        assert len(local_features) == 5  # 5 basic local features
        
        # Test quality assessment fallback
        quality_score = dedup._basic_quality_assessment(self.test_images[0])
        assert quality_score is not None
        assert 0 <= quality_score <= 100
        
        logger.info("✅ Feature extraction fallback tests passed")
    
    def test_quality_based_selection(self):
        """Test quality-based image selection"""
        logger.info("Testing quality-based image selection...")
        
        dedup = ColorOptimizedDeduplicator(feature_cache=self.feature_cache)
        dedup.feature_extractor = self.mock_feature_extractor
        
        # Test single group selection
        test_group = self.test_images[:3]
        best_group = dedup._select_best_images_from_group(test_group)
        assert len(best_group) == 1
        assert best_group[0] in test_group
        
        # Test multiple groups selection
        test_groups = [self.test_images[:2], self.test_images[2:4], self.test_images[4:6]]
        final_groups = dedup._select_best_images_from_groups(test_groups)
        assert len(final_groups) == len(test_groups)
        
        for group in final_groups:
            assert len(group) == 1
        
        logger.info("✅ Quality-based selection tests passed")
    
    def test_deduplication_pipeline_integration(self):
        """Test full deduplication pipeline integration"""
        logger.info("Testing deduplication pipeline integration...")
        
        dedup = ColorOptimizedDeduplicator(
            feature_cache=self.feature_cache,
            color_clusters=3,
            batch_size=5
        )
        
        # Mock the parent class methods
        dedup._stage1_wavelet_grouping = Mock(return_value=[self.test_images])
        
        # Test deduplication within color group
        similarity_scores = {}
        results, scores = dedup._deduplicate_within_color_group(
            self.test_images[:5], 
            similarity_scores
        )
        
        assert len(results) > 0
        assert isinstance(scores, dict)
        
        logger.info("✅ Deduplication pipeline integration tests passed")
    
    def test_memory_management(self):
        """Test memory management and cleanup"""
        logger.info("Testing memory management and cleanup...")
        
        dedup = ColorOptimizedDeduplicator(feature_cache=self.feature_cache)
        
        # Add some data to memory stats
        dedup.memory_stats['test_value'] = 123
        
        # Test release method
        dedup.release()
        
        # Verify cleanup
        assert dedup.color_groups is None
        assert dedup.color_model is None
        assert len(dedup.color_features_cache) == 0
        assert len(dedup.memory_stats) == 0
        
        logger.info("✅ Memory management tests passed")
    
    def test_factory_function(self):
        """Test factory function for creating deduplicator instances"""
        logger.info("Testing factory function...")
        
        # Test factory function
        dedup = create_color_optimized_deduplicator(
            feature_cache=self.feature_cache,
            color_clusters=1000,
            batch_size=500
        )
        
        assert isinstance(dedup, ColorOptimizedDeduplicator)
        assert dedup.color_clusters == 1000
        assert dedup.batch_size == 500
        assert dedup.feature_cache == self.feature_cache
        
        logger.info("✅ Factory function tests passed")
    
    def test_error_handling(self):
        """Test error handling and robustness"""
        logger.info("Testing error handling and robustness...")
        
        dedup = ColorOptimizedDeduplicator(feature_cache=self.feature_cache)
        
        # Test with non-existent image
        non_existent_img = "/path/to/nonexistent/image.jpg"
        result = dedup._load_image_efficiently(non_existent_img)
        assert result is None
        
        # Test with invalid Azure URL
        invalid_azure_url = "https://invalid-url.com/image.jpg"
        result = dedup._load_azure_image(invalid_azure_url)
        assert result is None
        
        # Test feature extraction with invalid image
        result = dedup._extract_compact_color_features(non_existent_img)
        assert result is None
        
        logger.info("✅ Error handling tests passed")
    
    def test_performance_metrics(self):
        """Test performance metrics and statistics"""
        logger.info("Testing performance metrics and statistics...")
        
        dedup = ColorOptimizedDeduplicator(feature_cache=self.feature_cache)
        
        # Simulate some processing
        dedup.memory_stats['color_groups_created'] = 5
        dedup.memory_stats['color_group_sizes'] = [10, 15, 8, 12, 20]
        dedup.memory_stats['cache_hits'] = 50
        dedup.memory_stats['cache_misses'] = 10
        
        # Test statistics calculation
        stats = dedup.get_color_optimization_stats()
        
        assert 'avg_group_size' in stats
        assert 'max_group_size' in stats
        assert 'min_group_size' in stats
        assert 'cache_efficiency' in stats
        
        assert stats['avg_group_size'] == 13.0
        assert stats['max_group_size'] == 20
        assert stats['min_group_size'] == 8
        assert abs(stats['cache_efficiency'] - 0.833) < 0.001
        
        logger.info("✅ Performance metrics tests passed")

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    logger.info("🚀 Starting comprehensive ColorOptimizedDeduplicator tests...")
    
    test_instance = TestColorOptimizedDeduplicator()
    
    # List of all test methods
    test_methods = [
        'test_initialization_and_configuration',
        'test_context_manager_functionality',
        'test_unified_feature_extraction',
        'test_image_loading_functionality',
        'test_color_feature_extraction',
        'test_adaptive_threshold_calculation',
        'test_color_pre_grouping',
        'test_parallel_processing',
        'test_feature_extraction_fallbacks',
        'test_quality_based_selection',
        'test_deduplication_pipeline_integration',
        'test_memory_management',
        'test_factory_function',
        'test_error_handling',
        'test_performance_metrics'
    ]
    
    passed_tests = 0
    total_tests = len(test_methods)
    
    for test_method in test_methods:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {test_method}")
            logger.info(f"{'='*60}")
            
            # Setup
            test_instance.setup_method()
            
            # Run test
            getattr(test_instance, test_method)()
            
            # Teardown
            test_instance.teardown_method()
            
            passed_tests += 1
            logger.info(f"✅ {test_method} PASSED")
            
        except Exception as e:
            logger.error(f"❌ {test_method} FAILED: {e}")
            test_instance.teardown_method()
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! ColorOptimizedDeduplicator is ready for production.")
    else:
        logger.error("⚠️  Some tests failed. Please review the implementation.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
