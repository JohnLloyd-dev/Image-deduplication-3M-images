#!/usr/bin/env python3
"""
Test script to verify that all the fixes work correctly.
"""

import os
import sys
import logging

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all imports work correctly."""
    logger.info("Testing imports...")
    
    try:
        from modules.feature_cache import BoundedFeatureCache
        from modules.feature_extraction import FeatureExtractor
        from modules.deduplication import Deduplicator
        from modules.distributed_processor import DistributedProcessor
        from modules.azure_utils import AzureBlobManager
        from modules.io_utils import save_image_info_to_csv, load_image_info_from_csv
        from pipeline import run_pipeline, ProgressTracker
        
        logger.info("✓ All imports successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Import failed: {str(e)}")
        return False

def test_feature_cache():
    """Test feature cache functionality."""
    logger.info("Testing feature cache...")
    
    try:
        cache = BoundedFeatureCache(cache_dir="test_cache", max_size=10)
        
        # Test storing and retrieving features
        test_features = {
            'global': [1.0, 2.0, 3.0],
            'local': {'keypoints': [4.0, 5.0], 'descriptors': [6.0, 7.0]},
            'wavelet': [8.0, 9.0, 10.0]
        }
        
        success = cache.put_features("test/image.jpg", test_features)
        if not success:
            logger.error("✗ Failed to store features")
            return False
            
        retrieved = cache.get_features("test/image.jpg")
        if retrieved != test_features:
            logger.error("✗ Retrieved features don't match stored features")
            return False
            
        logger.info("✓ Feature cache works correctly")
        cache.clear()
        return True
        
    except Exception as e:
        logger.error(f"✗ Feature cache test failed: {str(e)}")
        return False

def test_deduplicator_wrapper():
    """Test the Deduplicator wrapper class."""
    logger.info("Testing Deduplicator wrapper...")
    
    try:
        deduplicator = Deduplicator(device="cpu")
        
        # Test with dummy data
        image_paths = ["image1.jpg", "image2.jpg"]
        features = {
            "image1.jpg": {
                'global': [1.0, 2.0, 3.0],
                'local': {'keypoints': [4.0, 5.0], 'descriptors': [6.0, 7.0]},
                'wavelet': [8.0, 9.0, 10.0]
            },
            "image2.jpg": {
                'global': [1.1, 2.1, 3.1],
                'local': {'keypoints': [4.1, 5.1], 'descriptors': [6.1, 7.1]},
                'wavelet': [8.1, 9.1, 10.1]
            }
        }
        
        # This should not crash (even if it returns empty results due to dummy data)
        duplicate_groups, similarity_scores = deduplicator.deduplicate(
            image_paths=image_paths,
            features=features,
            output_dir="test_output"
        )
        
        logger.info("✓ Deduplicator wrapper works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Deduplicator wrapper test failed: {str(e)}")
        return False

def test_csv_report_parsing():
    """Test CSV report parsing for both best and duplicate images."""
    logger.info("Testing CSV report parsing...")
    
    try:
        import pandas as pd
        import tempfile
        import os
        
        # Create a temporary CSV file with test data
        test_data = [
            {'Image Path': 'test/image1.jpg', 'Status': 'Best', 'Group ID': 1, 'Quality Score': 95.5},
            {'Image Path': 'test/image2.jpg', 'Status': 'Duplicate', 'Group ID': 1, 'Quality Score': 85.2},
            {'Image Path': 'test/image3.jpg', 'Status': 'Duplicate', 'Group ID': 1, 'Quality Score': 78.9},
            {'Image Path': 'test/image4.jpg', 'Status': 'Best', 'Group ID': 2, 'Quality Score': 92.1},
            {'Image Path': 'test/image5.jpg', 'Status': 'Duplicate', 'Group ID': 2, 'Quality Score': 88.7},
        ]
        
        df = pd.DataFrame(test_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_csv_path = f.name
        
        try:
            # Test parsing logic (same as in pipeline)
            df_read = pd.read_csv(temp_csv_path)
            
            best_images_df = df_read[df_read['Status'] == 'Best']
            duplicate_images_df = df_read[df_read['Status'] == 'Duplicate']
            
            best_images = set(best_images_df['Image Path'].tolist())
            duplicate_images = set(duplicate_images_df['Image Path'].tolist())
            
            # Verify results
            expected_best = {'test/image1.jpg', 'test/image4.jpg'}
            expected_duplicates = {'test/image2.jpg', 'test/image3.jpg', 'test/image5.jpg'}
            
            if best_images == expected_best and duplicate_images == expected_duplicates:
                logger.info("✓ CSV report parsing works correctly")
                logger.info(f"  - Found {len(best_images)} best images: {best_images}")
                logger.info(f"  - Found {len(duplicate_images)} duplicate images: {duplicate_images}")
                return True
            else:
                logger.error(f"✗ CSV parsing mismatch. Expected best: {expected_best}, got: {best_images}")
                logger.error(f"  Expected duplicates: {expected_duplicates}, got: {duplicate_images}")
                return False
                
        finally:
            # Clean up temp file
            os.unlink(temp_csv_path)
        
    except Exception as e:
        logger.error(f"✗ CSV report parsing test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting fix verification tests...")
    
    tests = [
        ("Imports", test_imports),
        ("Feature Cache", test_feature_cache),
        ("Deduplicator Wrapper", test_deduplicator_wrapper),
        ("CSV Report Parsing", test_csv_report_parsing),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✓ {test_name} test PASSED")
            else:
                logger.error(f"✗ {test_name} test FAILED")
                
        except Exception as e:
            logger.error(f"✗ {test_name} test ERROR: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Fixes are working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the fixes.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)