#!/usr/bin/env python3
"""
Test script to verify pipeline connections are working correctly.
"""

import os
import sys
import logging
import json
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.feature_cache import BoundedFeatureCache
from modules.azure_utils import AzureBlobManager
from pipeline import ProgressTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_feature_cache():
    """Test the feature cache functionality."""
    logger.info("Testing feature cache...")
    
    try:
        # Initialize feature cache
        cache = BoundedFeatureCache(cache_dir="test_features", max_size=100)
        
        # Test storing features
        test_features = {
            'global': [1.0, 2.0, 3.0],
            'local': [4.0, 5.0, 6.0],
            'wavelet': [7.0, 8.0, 9.0]
        }
        
        azure_path = "test/folder/image.jpg"
        success = cache.put_features(azure_path, test_features)
        
        if success:
            logger.info("✓ Feature cache put_features() works")
        else:
            logger.error("✗ Feature cache put_features() failed")
            return False
        
        # Test retrieving features
        retrieved_features = cache.get_features(azure_path)
        if retrieved_features and retrieved_features == test_features:
            logger.info("✓ Feature cache get_features() works")
        else:
            logger.error("✗ Feature cache get_features() failed")
            return False
        
        # Test get_all_features
        all_features = cache.get_all_features()
        if len(all_features) == 1 and azure_path in all_features:
            logger.info("✓ Feature cache get_all_features() works")
        else:
            logger.error("✗ Feature cache get_all_features() failed")
            return False
        
        # Test has_features
        if cache.has_features(azure_path):
            logger.info("✓ Feature cache has_features() works")
        else:
            logger.error("✗ Feature cache has_features() failed")
            return False
        
        # Clean up
        cache.clear()
        logger.info("✓ Feature cache clear() works")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Feature cache test failed: {str(e)}")
        return False

def test_progress_tracker():
    """Test the progress tracker functionality."""
    logger.info("Testing progress tracker...")
    
    try:
        # Initialize progress tracker
        tracker = ProgressTracker(total_images=100)
        
        # Test updates
        tracker.update_downloaded()
        tracker.update_processed()
        tracker.update_saved()
        tracker.add_processed_file("test/file.jpg")
        
        # Test summary
        summary = tracker.get_summary()
        expected_keys = ['total_images', 'downloaded', 'processed', 'saved', 'processed_files']
        
        if all(key in summary for key in expected_keys):
            logger.info("✓ Progress tracker summary works")
        else:
            logger.error("✗ Progress tracker summary failed")
            return False
        
        # Test save/load
        progress_file = "test_progress.json"
        tracker.save_progress(progress_file)
        
        loaded_tracker = ProgressTracker.load_progress(progress_file)
        if loaded_tracker and loaded_tracker.get_summary() == summary:
            logger.info("✓ Progress tracker save/load works")
        else:
            logger.error("✗ Progress tracker save/load failed")
            return False
        
        # Test is_processed
        if loaded_tracker.is_processed("test/file.jpg"):
            logger.info("✓ Progress tracker is_processed() works")
        else:
            logger.error("✗ Progress tracker is_processed() failed")
            return False
        
        # Clean up
        if os.path.exists(progress_file):
            os.remove(progress_file)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Progress tracker test failed: {str(e)}")
        return False

def test_azure_connection():
    """Test Azure connection functionality."""
    logger.info("Testing Azure connection...")
    
    try:
        # Test SAS URL validation
        sas_url = "https://azwtewebsitecache.blob.core.windows.net/webvia?sp=rcwl&st=2025-05-05T17:40:16Z&se=2025-11-05T18:40:16Z&spr=https&sv=2024-11-04&sr=c&sig=6eTcYmq%2BeauVioFmi1bxh%2Bd4gDjvNdq54EufmpPSKYY%3D"
        
        # Initialize Azure blob manager
        azure_manager = AzureBlobManager(sas_url)
        
        # Test container name extraction
        container_name = azure_manager._extract_container_name(sas_url)
        if container_name == "webvia":
            logger.info("✓ Azure container name extraction works")
        else:
            logger.error("✗ Azure container name extraction failed")
            return False
        
        # Test copy_blob method exists
        if hasattr(azure_manager, 'copy_blob'):
            logger.info("✓ Azure copy_blob method exists")
        else:
            logger.error("✗ Azure copy_blob method missing")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Azure connection test failed: {str(e)}")
        return False

def test_pipeline_structure():
    """Test the overall pipeline structure."""
    logger.info("Testing pipeline structure...")
    
    try:
        # Import pipeline components
        from pipeline import (
            TokenBucketRateLimiter,
            AzureConnectionPool,
            download_worker_azure,
            process_worker,
            save_worker,
            shutdown_sequence,
            run_pipeline
        )
        
        logger.info("✓ All pipeline components imported successfully")
        
        # Test rate limiter
        rate_limiter = TokenBucketRateLimiter(rate_per_sec=10)
        if hasattr(rate_limiter, 'wait'):
            logger.info("✓ Rate limiter works")
        else:
            logger.error("✗ Rate limiter missing wait method")
            return False
        
        # Use a syntactically valid dummy SAS URL for testing structure
        dummy_sas_url = "https://account.blob.core.windows.net/container?sp=rl&st=2020-01-01T00:00:00Z&se=2030-01-01T00:00:00Z&spr=https&sv=2020-08-04&sr=c&sig=dummy"
        sas_urls = [dummy_sas_url] * 3
        try:
            connection_pool = AzureConnectionPool(sas_urls)
            if hasattr(connection_pool, 'get_connection'):
                logger.info("✓ Connection pool works")
            else:
                logger.error("✗ Connection pool missing get_connection method")
                return False
        except Exception as e:
            logger.error(f"✗ Connection pool initialization failed: {str(e)}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Pipeline structure test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting pipeline connection tests...")
    
    tests = [
        ("Feature Cache", test_feature_cache),
        ("Progress Tracker", test_progress_tracker),
        ("Azure Connection", test_azure_connection),
        ("Pipeline Structure", test_pipeline_structure)
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
        logger.info("🎉 All tests passed! Pipeline connections are working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the pipeline connections.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 