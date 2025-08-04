#!/usr/bin/env python3
"""
Test script to verify the hierarchical deduplication process works correctly.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.feature_cache import BoundedFeatureCache
from modules.deduplication import HierarchicalDeduplicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_features() -> Dict[str, Dict]:
    """Create test features that should form duplicate groups."""
    
    # Create similar features for duplicates
    base_global_1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    base_global_2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    
    base_local_1 = {
        'keypoints': np.array([[100, 200], [150, 250]]),
        'descriptors': np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    }
    base_local_2 = {
        'keypoints': np.array([[300, 400], [350, 450]]),
        'descriptors': np.array([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]])
    }
    
    base_wavelet_1 = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 4)  # 32 bits
    base_wavelet_2 = np.array([0, 1, 0, 0, 1, 1, 0, 1] * 4)  # 32 bits
    
    features = {
        # Group 1: Very similar images (should be duplicates)
        'image1.jpg': {
            'global': base_global_1,
            'local': base_local_1,
            'wavelet': base_wavelet_1,
            'color_features': np.array([0.8, 0.2, 0.1, 0.9, 0.3])  # Color histogram
        },
        'image1_copy.jpg': {
            'global': base_global_1 + np.random.normal(0, 0.01, 5),  # Very similar
            'local': {
                'keypoints': base_local_1['keypoints'] + np.random.normal(0, 1, (2, 2)),
                'descriptors': base_local_1['descriptors'] + np.random.normal(0, 0.01, (2, 3))
            },
            'wavelet': base_wavelet_1,  # Identical wavelet
            'color_features': np.array([0.81, 0.21, 0.11, 0.89, 0.31])  # Very similar colors
        },
        'image1_variant.jpg': {
            'global': base_global_1 + np.random.normal(0, 0.02, 5),  # Very similar
            'local': {
                'keypoints': base_local_1['keypoints'] + np.random.normal(0, 2, (2, 2)),
                'descriptors': base_local_1['descriptors'] + np.random.normal(0, 0.02, (2, 3))
            },
            'wavelet': base_wavelet_1,  # Identical wavelet
            'color_features': np.array([0.82, 0.22, 0.12, 0.88, 0.32])  # Very similar colors
        },
        
        # Group 2: Another set of similar images
        'image2.jpg': {
            'global': base_global_2,
            'local': base_local_2,
            'wavelet': base_wavelet_2,
            'color_features': np.array([0.1, 0.9, 0.8, 0.2, 0.7])  # Different colors
        },
        'image2_copy.jpg': {
            'global': base_global_2 + np.random.normal(0, 0.01, 5),  # Very similar
            'local': {
                'keypoints': base_local_2['keypoints'] + np.random.normal(0, 1, (2, 2)),
                'descriptors': base_local_2['descriptors'] + np.random.normal(0, 0.01, (2, 3))
            },
            'wavelet': base_wavelet_2,  # Identical wavelet
            'color_features': np.array([0.11, 0.89, 0.81, 0.21, 0.71])  # Very similar colors
        },
        
        # Unique image (should not be in any duplicate group)
        'unique_image.jpg': {
            'global': np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
            'local': {
                'keypoints': np.array([[500, 600], [550, 650]]),
                'descriptors': np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
            },
            'wavelet': np.array([1, 1, 0, 0, 1, 0, 1, 1] * 4),  # Different pattern
            'color_features': np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # Neutral colors
        }
    }
    
    return features

def test_hierarchical_deduplication():
    """Test the complete hierarchical deduplication process."""
    logger.info("Testing hierarchical deduplication process...")
    
    try:
        # Create test data
        features = create_test_features()
        image_paths = list(features.keys())
        
        logger.info(f"Created test data with {len(image_paths)} images:")
        for path in image_paths:
            logger.info(f"  - {path}")
        
        # Create feature cache and populate it
        cache = BoundedFeatureCache(cache_dir="test_dedup_cache", max_size=100)
        for path, feat in features.items():
            cache.put_features(path, feat)
        
        # Create deduplicator with appropriate thresholds
        deduplicator = HierarchicalDeduplicator(
            feature_cache=cache,
            global_threshold=0.85,   # Standard threshold for global similarity
            local_threshold=0.75,    # Standard threshold for local similarity
            wavelet_threshold=0.8,   # Standard threshold for wavelet similarity
            device="cpu"
        )
        
        # Run deduplication
        logger.info("Running hierarchical deduplication...")
        duplicate_groups, similarity_scores = deduplicator.deduplicate(
            image_paths=image_paths,
            features=features,
            output_dir="test_dedup_output"
        )
        
        # Analyze results
        logger.info(f"\n{'='*50}")
        logger.info("DEDUPLICATION RESULTS")
        logger.info(f"{'='*50}")
        
        logger.info(f"Found {len(duplicate_groups)} duplicate groups:")
        
        total_duplicates = 0
        for i, group in enumerate(duplicate_groups):
            logger.info(f"  Group {i+1}: {len(group)} images")
            for img in group:
                logger.info(f"    - {img}")
            total_duplicates += len(group)
        
        logger.info(f"\nTotal images in duplicate groups: {total_duplicates}")
        logger.info(f"Unique images: {len(image_paths) - total_duplicates}")
        
        # Verify expected results
        expected_groups = 2  # Should find 2 duplicate groups
        expected_group1_size = 3  # image1.jpg, image1_copy.jpg, image1_variant.jpg
        expected_group2_size = 2  # image2.jpg, image2_copy.jpg
        
        success = True
        
        if len(duplicate_groups) != expected_groups:
            logger.error(f"Expected {expected_groups} groups, got {len(duplicate_groups)}")
            success = False
        
        # Check if we have the right group sizes
        group_sizes = sorted([len(group) for group in duplicate_groups], reverse=True)
        expected_sizes = sorted([expected_group1_size, expected_group2_size], reverse=True)
        
        if group_sizes != expected_sizes:
            logger.error(f"Expected group sizes {expected_sizes}, got {group_sizes}")
            success = False
        
        # Check similarity scores
        logger.info(f"\nSimilarity scores computed: {len(similarity_scores)}")
        for (img1, img2), score in similarity_scores.items():
            logger.info(f"  {img1} <-> {img2}: {score:.3f}")
        
        if success:
            logger.info("✅ Hierarchical deduplication test PASSED")
            logger.info("   - Correctly identified duplicate groups")
            logger.info("   - Proper similarity scoring")
            logger.info("   - Expected group sizes")
        else:
            logger.error("❌ Hierarchical deduplication test FAILED")
        
        # Clean up
        cache.clear()
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Hierarchical deduplication test ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_similarity_methods():
    """Test individual similarity computation methods."""
    logger.info("Testing individual similarity methods...")
    
    try:
        cache = BoundedFeatureCache(cache_dir="test_sim_cache", max_size=10)
        deduplicator = HierarchicalDeduplicator(feature_cache=cache, device="cpu")
        
        # Test global similarity
        feat1 = np.array([1.0, 2.0, 3.0])
        feat2 = np.array([1.1, 2.1, 3.1])  # Very similar
        feat3 = np.array([10.0, 20.0, 30.0])  # Different
        
        sim_similar = deduplicator.compute_global_similarity(feat1, feat2)
        sim_different = deduplicator.compute_global_similarity(feat1, feat3)
        
        logger.info(f"Global similarity (similar): {sim_similar:.3f}")
        logger.info(f"Global similarity (different): {sim_different:.3f}")
        
        # Test wavelet similarity
        hash1 = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        hash2 = np.array([1, 0, 1, 1, 0, 0, 1, 0])  # Identical
        hash3 = np.array([0, 1, 0, 0, 1, 1, 0, 1])  # Different
        
        wav_identical = deduplicator.compute_wavelet_similarity(hash1, hash2)
        wav_different = deduplicator.compute_wavelet_similarity(hash1, hash3)
        
        logger.info(f"Wavelet similarity (identical): {wav_identical:.3f}")
        logger.info(f"Wavelet similarity (different): {wav_different:.3f}")
        
        # Test local similarity
        local1 = {
            'descriptors': np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        }
        local2 = {
            'descriptors': np.array([[0.11, 0.21, 0.31], [0.41, 0.51, 0.61]])  # Similar
        }
        local3 = {
            'descriptors': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Different
        }
        
        local_similar = deduplicator._compute_local_similarity(local1, local2)
        local_different = deduplicator._compute_local_similarity(local1, local3)
        
        logger.info(f"Local similarity (similar): {local_similar:.3f}")
        logger.info(f"Local similarity (different): {local_different:.3f}")
        
        # Verify expectations
        success = True
        if sim_similar <= sim_different:
            logger.error("Global similarity: similar should be > different")
            success = False
        
        if wav_identical != 1.0:
            logger.error("Wavelet similarity: identical should be 1.0")
            success = False
        
        if wav_identical <= wav_different:
            logger.error("Wavelet similarity: identical should be > different")
            success = False
        
        if local_similar <= local_different:
            logger.error("Local similarity: similar should be > different")
            success = False
        
        if success:
            logger.info("✅ Individual similarity methods test PASSED")
        else:
            logger.error("❌ Individual similarity methods test FAILED")
        
        cache.clear()
        return success
        
    except Exception as e:
        logger.error(f"❌ Individual similarity methods test ERROR: {str(e)}")
        return False

def test_edge_cases():
    """Test edge cases in hierarchical deduplication."""
    logger.info("Testing edge cases...")
    
    try:
        cache = BoundedFeatureCache(cache_dir="test_edge_cache", max_size=50)
        deduplicator = HierarchicalDeduplicator(feature_cache=cache, device="cpu")
        
        # Test 1: Empty input
        logger.info("Test 1: Empty input")
        duplicate_groups, similarity_scores = deduplicator.deduplicate(
            image_paths=[],
            features={},
            output_dir="test_edge_output"
        )
        if len(duplicate_groups) != 0:
            logger.error("Empty input should return no groups")
            return False
        
        # Test 2: Single image
        logger.info("Test 2: Single image")
        single_features = {
            'single.jpg': {
                'global': np.array([1.0, 2.0, 3.0]),
                'local': {'descriptors': np.array([[0.1, 0.2, 0.3]])},
                'wavelet': np.array([1, 0, 1, 0] * 8)
            }
        }
        cache.put_features('single.jpg', single_features['single.jpg'])
        
        duplicate_groups, similarity_scores = deduplicator.deduplicate(
            image_paths=['single.jpg'],
            features=single_features,
            output_dir="test_edge_output"
        )
        if len(duplicate_groups) != 0:
            logger.error("Single image should return no duplicate groups")
            return False
        
        # Test 3: Images with missing features
        logger.info("Test 3: Images with missing features")
        incomplete_features = {
            'img1.jpg': {
                'global': np.array([1.0, 2.0, 3.0]),
                'local': None,  # Missing local features
                'wavelet': np.array([1, 0, 1, 0] * 8)
            },
            'img2.jpg': {
                'global': None,  # Missing global features
                'local': {'descriptors': np.array([[0.1, 0.2, 0.3]])},
                'wavelet': np.array([1, 0, 1, 0] * 8)
            },
            'img3.jpg': {
                'global': np.array([1.0, 2.0, 3.0]),
                'local': {'descriptors': np.array([[0.1, 0.2, 0.3]])},
                'wavelet': None  # Missing wavelet features
            }
        }
        
        for path, feat in incomplete_features.items():
            cache.put_features(path, feat)
        
        duplicate_groups, similarity_scores = deduplicator.deduplicate(
            image_paths=list(incomplete_features.keys()),
            features=incomplete_features,
            output_dir="test_edge_output"
        )
        
        # Should handle missing features gracefully
        logger.info(f"Handled incomplete features: {len(duplicate_groups)} groups found")
        
        # Test 4: Very similar but not identical features
        logger.info("Test 4: Very similar features")
        base_global = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        similar_features = {
            'similar1.jpg': {
                'global': base_global,
                'local': {'descriptors': np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])},
                'wavelet': np.array([1, 0, 1, 1, 0, 0, 1, 0] * 4)
            },
            'similar2.jpg': {
                'global': base_global + 0.001,  # Very similar
                'local': {'descriptors': np.array([[0.101, 0.201, 0.301], [0.401, 0.501, 0.601]])},
                'wavelet': np.array([1, 0, 1, 1, 0, 0, 1, 0] * 4)  # Identical
            }
        }
        
        for path, feat in similar_features.items():
            cache.put_features(path, feat)
        
        duplicate_groups, similarity_scores = deduplicator.deduplicate(
            image_paths=list(similar_features.keys()),
            features=similar_features,
            output_dir="test_edge_output"
        )
        
        # Should find these as duplicates
        if len(duplicate_groups) != 1 or len(duplicate_groups[0]) != 2:
            logger.error(f"Expected 1 group with 2 images, got {len(duplicate_groups)} groups")
            return False
        
        logger.info("✅ Edge cases test PASSED")
        cache.clear()
        return True
        
    except Exception as e:
        logger.error(f"❌ Edge cases test ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all deduplication tests."""
    logger.info("Starting deduplication process tests...")
    
    tests = [
        ("Individual Similarity Methods", test_individual_similarity_methods),
        ("Edge Cases", test_edge_cases),
        ("Hierarchical Deduplication Process", test_hierarchical_deduplication),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} test ERROR: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("DEDUPLICATION TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All deduplication tests passed! The process works correctly.")
        return True
    else:
        logger.error("❌ Some deduplication tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)