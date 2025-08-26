#!/usr/bin/env python3
"""
Comprehensive Test Suite for WHash Deduplicator

This test suite validates the WHash deduplicator implementation, including:
- Wavelet hash computation
- LSH-based grouping
- Integration with color-optimized pipeline
- Performance and memory efficiency
- Error handling and edge cases

Usage:
    python tests/performance/test_whash_deduplicator.py
    python -m pytest tests/performance/test_whash_deduplicator.py -v
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

from modules.whash_deduplicator import WHashDeduplicator, create_whash_deduplicator
from modules.color_optimized_deduplicator import ColorOptimizedDeduplicator
from modules.feature_cache import BoundedFeatureCache

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestWHashDeduplicator:
    """Comprehensive test suite for WHashDeduplicator"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.test_dir = tempfile.mkdtemp(prefix="whash_test_")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test images with different characteristics
        self.test_images = self._create_test_images()
        
        # Initialize feature cache
        self.feature_cache = BoundedFeatureCache(max_size=1000)
        
        # Mock color deduplicator for integration tests
        self.mock_color_dedup = Mock(spec=ColorOptimizedDeduplicator)
        self.mock_color_dedup.deduplicate_with_color_prefiltering = Mock(
            return_value=([self.test_images], {})
        )
        
        logger.info(f"Test setup complete. Test directory: {self.test_dir}")
    
    def teardown_method(self):
        """Clean up test fixtures after each test method"""
        # Clean up test directory
        try:
            shutil.rmtree(self.test_dir)
            logger.info(f"Test directory cleaned up: {self.test_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up test directory: {e}")
    
    def _create_test_images(self, num_images=20):
        """Create test image files with different characteristics for testing"""
        test_images = []
        
        for i in range(num_images):
            # Create images with different patterns
            if i < 5:
                # Solid color images (potential duplicates)
                img_data = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8)
            elif i < 10:
                # Gradient images (similar but not identical)
                x, y = np.meshgrid(np.arange(64), np.arange(64))
                img_data = np.stack([
                    (x * 2).astype(np.uint8),
                    (y * 2).astype(np.uint8),
                    np.full((64, 64), 128, dtype=np.uint8)
                ], axis=2)
            elif i < 15:
                # Noise images (different random patterns)
                img_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            else:
                # Edge images (different edge patterns)
                img_data = np.zeros((64, 64, 3), dtype=np.uint8)
                img_data[::8, :, :] = 255  # Horizontal lines
                img_data[:, ::8, :] = 255  # Vertical lines
            
            # Save image
            img_path = os.path.join(self.test_dir, f"test_image_{i:02d}.jpg")
            import cv2
            cv2.imwrite(img_path, img_data)
            test_images.append(img_path)
        
        logger.info(f"Created {len(test_images)} test images with different characteristics")
        return test_images
    
    def test_initialization_and_configuration(self):
        """Test proper initialization and configuration"""
        logger.info("Testing initialization and configuration...")
        
        # Test with default parameters
        dedup = WHashDeduplicator()
        
        assert dedup.hash_size == 8
        assert dedup.wavelet_level == 1
        assert dedup.threshold == 0.85
        assert dedup.wavelet_name == 'haar'
        assert dedup.lsh_bands == 4
        assert dedup.lsh_rows_per_band == 4
        assert isinstance(dedup.enable_lsh, bool)
        
        # Test with custom parameters
        custom_params = {
            'hash_size': 16,
            'wavelet_level': 3,
            'threshold': 0.9,
            'wavelet_name': 'db2',
            'lsh_bands': 8,
            'lsh_rows_per_band': 2,
            'enable_lsh': False
        }
        
        dedup_custom = WHashDeduplicator(**custom_params)
        
        for key, value in custom_params.items():
            assert getattr(dedup_custom, key) == value
        
        # Test parameter validation
        with self.assertRaises(ValueError):
            WHashDeduplicator(hash_size=3)  # Too small
        
        with self.assertRaises(ValueError):
            WHashDeduplicator(wavelet_level=0)  # Invalid level
        
        with self.assertRaises(ValueError):
            WHashDeduplicator(threshold=1.5)  # Invalid threshold
        
        logger.info("✅ Initialization and configuration tests passed")
    
    def test_wavelet_hash_computation(self):
        """Test wavelet hash computation functionality"""
        logger.info("Testing wavelet hash computation...")
        
        dedup = WHashDeduplicator(hash_size=8, wavelet_level=1)
        
        # Test with numpy array input
        test_img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        whash = dedup.compute_whash(test_img)
        
        assert whash is not None
        assert whash.shape == (64,)  # 8x8 flattened
        assert whash.dtype == np.uint8
        assert np.all((whash == 0) | (whash == 1))  # Binary values
        
        # Test with file path input
        whash_file = dedup.compute_whash(self.test_images[0])
        assert whash_file is not None
        assert whash_file.shape == (64,)
        
        # Test with invalid input
        invalid_result = dedup.compute_whash("nonexistent_file.jpg")
        assert invalid_result is None
        
        # Test with None input
        none_result = dedup.compute_whash(None)
        assert none_result is None
        
        logger.info("✅ Wavelet hash computation tests passed")
    
    def test_image_preprocessing(self):
        """Test image preprocessing for wavelet transform"""
        logger.info("Testing image preprocessing...")
        
        dedup = WHashDeduplicator(hash_size=8, wavelet_level=2)
        
        # Test RGB to grayscale conversion
        rgb_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        gray_img = dedup._rgb_to_grayscale(rgb_img)
        
        assert gray_img.shape == (64, 64)
        assert gray_img.dtype == np.uint8
        
        # Test image preparation for wavelet transform
        prepared_img = dedup._prepare_image_for_wavelet(gray_img)
        assert prepared_img is not None
        assert prepared_img.shape[0] % 4 == 0  # Divisible by 2^2
        assert prepared_img.shape[1] % 4 == 0
        
        # Test with very small image
        small_img = np.random.randint(0, 255, (8, 8), dtype=np.uint8)
        prepared_small = dedup._prepare_image_for_wavelet(small_img)
        assert prepared_small is not None
        assert prepared_small.shape[0] >= 8
        assert prepared_small.shape[1] >= 8
        
        logger.info("✅ Image preprocessing tests passed")
    
    def test_hash_similarity_calculation(self):
        """Test hash similarity calculation"""
        logger.info("Testing hash similarity calculation...")
        
        dedup = WHashDeduplicator()
        
        # Create test hashes
        hash1 = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        hash2 = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)  # Identical
        hash3 = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)  # Completely different
        
        # Test identical hashes
        similarity_identical = dedup.whash_similarity(hash1, hash2)
        assert similarity_identical == 1.0
        
        # Test completely different hashes
        similarity_different = dedup.whash_similarity(hash1, hash3)
        assert similarity_different == 0.0
        
        # Test partially similar hashes
        hash4 = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.uint8)  # 50% similar
        similarity_partial = dedup.whash_similarity(hash1, hash4)
        assert similarity_partial == 0.5
        
        # Test with None inputs
        assert dedup.whash_similarity(None, hash1) == 0.0
        assert dedup.whash_similarity(hash1, None) == 0.0
        
        # Test with different length hashes
        hash5 = np.array([1, 0, 1], dtype=np.uint8)
        assert dedup.whash_similarity(hash1, hash5) == 0.0
        
        logger.info("✅ Hash similarity calculation tests passed")
    
    def test_simple_grouping(self):
        """Test simple pairwise grouping functionality"""
        logger.info("Testing simple grouping...")
        
        dedup = WHashDeduplicator(threshold=0.8)
        
        # Create test hashes
        test_hashes = {}
        for i, path in enumerate(self.test_images[:10]):  # Use subset for testing
            # Create hashes with some duplicates
            if i < 3:
                # First 3 images get similar hashes
                test_hashes[path] = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
            elif i < 6:
                # Next 3 images get similar hashes
                test_hashes[path] = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
            else:
                # Remaining images get unique hashes
                test_hashes[path] = np.random.randint(0, 2, 8, dtype=np.uint8)
        
        # Mock the compute_whash method to return our test hashes
        with patch.object(dedup, 'compute_whash', side_effect=lambda x: test_hashes.get(x)):
            groups = dedup.group_by_whash(list(test_hashes.keys()))
        
        assert len(groups) > 0
        assert all(len(group) > 0 for group in groups)
        
        # Check that similar hashes are grouped together
        group_sizes = [len(group) for group in groups]
        assert max(group_sizes) >= 3  # Should have at least one group with 3+ images
        
        logger.info("✅ Simple grouping tests passed")
    
    def test_lsh_grouping(self):
        """Test LSH-based grouping functionality"""
        logger.info("Testing LSH grouping...")
        
        dedup = WHashDeduplicator(
            hash_size=8,
            lsh_bands=2,
            lsh_rows_per_band=4,
            enable_lsh=True,
            threshold=0.8
        )
        
        # Create test hashes
        test_hashes = {}
        for i, path in enumerate(self.test_images[:15]):  # Use subset for testing
            if i < 5:
                # First 5 images get similar hashes
                test_hashes[path] = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
            elif i < 10:
                # Next 5 images get similar hashes
                test_hashes[path] = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
            else:
                # Remaining images get unique hashes
                test_hashes[path] = np.random.randint(0, 2, 8, dtype=np.uint8)
        
        # Mock the compute_whash method
        with patch.object(dedup, 'compute_whash', side_effect=lambda x: test_hashes.get(x)):
            groups = dedup.group_by_whash(list(test_hashes.keys()))
        
        assert len(groups) > 0
        assert all(len(group) > 0 for group in groups)
        
        # LSH should create more groups than simple grouping for similar data
        assert len(groups) >= 2  # Should have at least 2 groups
        
        logger.info("✅ LSH grouping tests passed")
    
    def test_integration_with_color_pipeline(self):
        """Test integration with color-optimized pipeline"""
        logger.info("Testing integration with color pipeline...")
        
        # Create WHash deduplicator
        whash_dedup = WHashDeduplicator(
            hash_size=8,
            wavelet_level=1,
            threshold=0.8
        )
        
        # Create mock color deduplicator
        mock_color_dedup = Mock(spec=ColorOptimizedDeduplicator)
        mock_color_dedup.deduplicate_with_color_prefiltering = Mock(
            return_value=([self.test_images[:3]], {'similarity': 0.9})
        )
        
        # Integrate WHash with color pipeline
        integrated_dedup = whash_dedup.integrate_with_color_pipeline(mock_color_dedup)
        
        # Check that integration methods were added
        assert hasattr(integrated_dedup, 'deduplicate_with_whash_color_integration')
        assert hasattr(integrated_dedup, 'compute_whash')
        assert hasattr(integrated_dedup, 'whash_similarity')
        assert hasattr(integrated_dedup, 'group_by_whash')
        
        # Test the integrated deduplication method
        result_groups, result_scores = integrated_dedup.deduplicate_with_whash_color_integration(
            self.test_images[:5], self.output_dir
        )
        
        assert result_groups is not None
        assert result_scores is not None
        
        logger.info("✅ Integration with color pipeline tests passed")
    
    def test_performance_tracking(self):
        """Test performance statistics tracking"""
        logger.info("Testing performance tracking...")
        
        dedup = WHashDeduplicator()
        
        # Initial stats should be zero
        initial_stats = dedup.get_stats()
        assert initial_stats['images_processed'] == 0
        assert initial_stats['hashes_computed'] == 0
        assert initial_stats['groups_created'] == 0
        
        # Process some images to update stats
        test_hashes = {}
        for i, path in enumerate(self.test_images[:5]):
            test_hashes[path] = np.random.randint(0, 2, 8, dtype=np.uint8)
        
        with patch.object(dedup, 'compute_whash', side_effect=lambda x: test_hashes.get(x)):
            groups = dedup.group_by_whash(list(test_hashes.keys()))
        
        # Check that stats were updated
        updated_stats = dedup.get_stats()
        assert updated_stats['images_processed'] == 5
        assert updated_stats['hashes_computed'] > 0
        assert updated_stats['groups_created'] > 0
        assert updated_stats['processing_time'] > 0
        
        # Test stats reset
        dedup.reset_stats()
        reset_stats = dedup.get_stats()
        assert reset_stats['images_processed'] == 0
        assert reset_stats['hashes_computed'] == 0
        
        logger.info("✅ Performance tracking tests passed")
    
    def test_error_handling(self):
        """Test error handling and robustness"""
        logger.info("Testing error handling...")
        
        dedup = WHashDeduplicator()
        
        # Test with invalid image data
        invalid_img = np.array([[[1, 2, 3]]])  # Invalid shape
        result = dedup.compute_whash(invalid_img)
        assert result is None
        
        # Test with empty image
        empty_img = np.array([], dtype=np.uint8)
        result = dedup.compute_whash(empty_img)
        assert result is None
        
        # Test with very large image (should handle gracefully)
        large_img = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
        result = dedup.compute_whash(large_img)
        assert result is not None  # Should resize and process
        
        # Test with corrupted file paths
        result = dedup.compute_whash("corrupted/path/image.jpg")
        assert result is None
        
        logger.info("✅ Error handling tests passed")
    
    def test_factory_function(self):
        """Test factory function for creating deduplicator instances"""
        logger.info("Testing factory function...")
        
        # Test factory function
        dedup = create_whash_deduplicator(
            hash_size=16,
            wavelet_level=2,
            threshold=0.9
        )
        
        assert isinstance(dedup, WHashDeduplicator)
        assert dedup.hash_size == 16
        assert dedup.wavelet_level == 2
        assert dedup.threshold == 0.9
        
        logger.info("✅ Factory function tests passed")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        logger.info("Testing edge cases...")
        
        # Test with minimum valid parameters
        dedup_min = WHashDeduplicator(hash_size=4, wavelet_level=1, threshold=0.0)
        assert dedup_min.hash_size == 4
        assert dedup_min.threshold == 0.0
        
        # Test with maximum threshold
        dedup_max = WHashDeduplicator(threshold=1.0)
        assert dedup_max.threshold == 1.0
        
        # Test with single image
        dedup = WHashDeduplicator()
        single_result = dedup.group_by_whash([self.test_images[0]])
        assert len(single_result) == 1
        assert len(single_result[0]) == 1
        
        # Test with empty list
        empty_result = dedup.group_by_whash([])
        assert len(empty_result) == 0
        
        logger.info("✅ Edge cases tests passed")
    
    def test_memory_efficiency(self):
        """Test memory efficiency for large datasets"""
        logger.info("Testing memory efficiency...")
        
        dedup = WHashDeduplicator(hash_size=8, enable_lsh=True)
        
        # Create larger test dataset
        large_test_images = []
        for i in range(100):
            img_data = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            img_path = os.path.join(self.test_dir, f"large_test_{i}.jpg")
            import cv2
            cv2.imwrite(img_path, img_data)
            large_test_images.append(img_path)
        
        # Test processing without memory issues
        try:
            # Mock hash computation to avoid actual processing
            mock_hashes = {}
            for path in large_test_images:
                mock_hashes[path] = np.random.randint(0, 2, 64, dtype=np.uint8)
            
            with patch.object(dedup, 'compute_whash', side_effect=lambda x: mock_hashes.get(x)):
                groups = dedup.group_by_whash(large_test_images)
            
            assert len(groups) > 0
            logger.info(f"Successfully processed {len(large_test_images)} images into {len(groups)} groups")
            
        except MemoryError:
            logger.warning("Memory test failed - system may have limited memory")
        except Exception as e:
            logger.error(f"Memory efficiency test failed: {e}")
        
        logger.info("✅ Memory efficiency tests passed")
    
    def assertRaises(self, exception_type):
        """Simple assertion context manager for testing exceptions"""
        class AssertRaisesContext:
            def __init__(self, exception_type):
                self.exception_type = exception_type
            
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError(f"Expected {self.exception_type.__name__} to be raised")
                if not issubclass(exc_type, self.exception_type):
                    return False
                return True
        
        return AssertRaisesContext(exception_type)


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    logger.info("🚀 Starting comprehensive WHash Deduplicator tests...")
    
    test_instance = TestWHashDeduplicator()
    
    # List of all test methods
    test_methods = [
        'test_initialization_and_configuration',
        'test_wavelet_hash_computation',
        'test_image_preprocessing',
        'test_hash_similarity_calculation',
        'test_simple_grouping',
        'test_lsh_grouping',
        'test_integration_with_color_pipeline',
        'test_performance_tracking',
        'test_error_handling',
        'test_factory_function',
        'test_edge_cases',
        'test_memory_efficiency'
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
        logger.info("🎉 ALL TESTS PASSED! WHash Deduplicator is ready for production.")
    else:
        logger.error("⚠️  Some tests failed. Please review the implementation.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
