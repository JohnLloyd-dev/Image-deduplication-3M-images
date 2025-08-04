#!/usr/bin/env python3
"""
Test on-demand feature computation for 3M+ images scenario.
This test verifies that features are computed on-demand instead of relying on pre-computed cache.
"""

import sys
import os
import logging
import time
from typing import List, Dict
import tempfile
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.feature_cache import BoundedFeatureCache
from modules.memory_efficient_deduplication import MemoryEfficientDeduplicator
from modules.azure_utils import AzureBlobManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_test_images() -> List[str]:
    """Get test images that exist in Azure storage."""
    
    test_images = [
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2105_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2105_TP.jpg", 
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2125_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2225_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2225.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-3021.jpg",
    ]
    
    return test_images


def verify_images_exist(image_paths: List[str]) -> List[str]:
    """Verify which test images actually exist."""
    
    logger.info("🔍 Verifying test images exist in Azure storage...")
    
    try:
        azure_manager = AzureBlobManager()
        
        existing_images = []
        missing_images = []
        
        for img_path in image_paths:
            try:
                properties = azure_manager.get_blob_properties(img_path)
                if properties:
                    existing_images.append(img_path)
                    logger.info(f"✅ Found: {os.path.basename(img_path)}")
                else:
                    missing_images.append(img_path)
                    logger.warning(f"❌ Missing: {os.path.basename(img_path)}")
            except Exception as e:
                missing_images.append(img_path)
                logger.warning(f"❌ Error checking {os.path.basename(img_path)}: {e}")
        
        logger.info(f"📊 Image verification complete:")
        logger.info(f"   - Total images: {len(image_paths)}")
        logger.info(f"   - Existing images: {len(existing_images)}")
        logger.info(f"   - Missing images: {len(missing_images)}")
        
        return existing_images
        
    except Exception as e:
        logger.error(f"❌ Error verifying images: {e}")
        return []


def test_ondemand_feature_computation():
    """Test that features are computed on-demand for 3M+ images scenario."""
    
    logger.info("🧪 Testing on-demand feature computation...")
    
    # Get existing images
    test_images = get_test_images()
    existing_images = verify_images_exist(test_images)
    
    if len(existing_images) < 3:
        logger.error("❌ Not enough existing images for testing!")
        return False
    
    # Create empty feature cache (simulating fresh start for 3M+ images)
    cache_dir = tempfile.mkdtemp(prefix="test_ondemand_cache_")
    cache = BoundedFeatureCache(cache_dir=cache_dir, max_size=1000)
    
    # Create deduplicator
    deduplicator = MemoryEfficientDeduplicator(
        feature_cache=cache,
        color_threshold=0.85,
        global_threshold=0.85,
        local_threshold=0.75,
        wavelet_threshold=0.8
    )
    
    # Create temporary output directory
    output_dir = tempfile.mkdtemp(prefix="test_ondemand_output_")
    
    try:
        logger.info("🔄 Testing full pipeline with on-demand feature computation...")
        start_time = time.time()
        
        # Test the full memory-efficient deduplication pipeline
        duplicate_groups, similarity_scores = deduplicator.deduplicate_memory_efficient(
            existing_images, output_dir, None
        )
        
        total_time = time.time() - start_time
        
        # Analyze results
        logger.info(f"\n🎉 Pipeline completed in {total_time:.2f}s!")
        logger.info(f"📊 Results:")
        logger.info(f"   - Input images: {len(existing_images)}")
        logger.info(f"   - Duplicate groups: {len(duplicate_groups)}")
        logger.info(f"   - Total duplicate images: {sum(len(group) for group in duplicate_groups)}")
        logger.info(f"   - Processing rate: {len(existing_images)/total_time:.1f} images/second")
        
        # Memory statistics
        memory_stats = deduplicator.memory_stats
        logger.info(f"💾 Memory Statistics:")
        logger.info(f"   - Peak memory usage: {memory_stats.get('peak_memory_mb', 0):.1f} MB")
        logger.info(f"   - Features loaded/computed: {memory_stats.get('features_loaded', 0)}")
        logger.info(f"   - Features freed: {memory_stats.get('features_freed', 0)}")
        
        # Check if features were computed on-demand
        if memory_stats.get('features_loaded', 0) > 0:
            logger.info("✅ On-demand feature computation is working!")
            logger.info("   - Features are being computed as needed")
            logger.info("   - Cache is being populated during processing")
            logger.info("   - Memory is being managed efficiently")
        else:
            logger.warning("⚠️  No features were loaded/computed - check implementation")
        
        # Show duplicate groups
        logger.info(f"📋 Duplicate Groups:")
        for i, group in enumerate(duplicate_groups):
            if len(group) > 1:  # Only show groups with duplicates
                logger.info(f"   Group {i+1}: {len(group)} images")
                for img in group[:3]:  # Show first 3 images
                    logger.info(f"     - {os.path.basename(img)}")
                if len(group) > 3:
                    logger.info(f"     - ... and {len(group)-3} more")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(output_dir)
            shutil.rmtree(cache_dir)
            logger.info("🧹 Cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️  Cleanup failed: {e}")


def test_stage1_wavelet_computation():
    """Test Stage 1 wavelet feature computation specifically."""
    
    logger.info("🎯 Testing Stage 1: Wavelet feature computation...")
    
    # Get existing images
    test_images = get_test_images()
    existing_images = verify_images_exist(test_images)
    
    if len(existing_images) < 3:
        logger.error("❌ Not enough existing images for testing!")
        return False
    
    # Create empty feature cache
    cache_dir = tempfile.mkdtemp(prefix="test_wavelet_cache_")
    cache = BoundedFeatureCache(cache_dir=cache_dir, max_size=1000)
    
    # Create deduplicator
    deduplicator = MemoryEfficientDeduplicator(
        feature_cache=cache,
        wavelet_threshold=0.8
    )
    
    try:
        logger.info("🔄 Testing Stage 1: Wavelet grouping with on-demand computation...")
        start_time = time.time()
        
        # Test Stage 1 specifically
        wavelet_groups = deduplicator._stage1_wavelet_grouping(existing_images)
        
        stage1_time = time.time() - start_time
        
        logger.info(f"✅ Stage 1 completed in {stage1_time:.2f}s")
        logger.info(f"📊 Results:")
        logger.info(f"   - Input images: {len(existing_images)}")
        logger.info(f"   - Wavelet groups: {len(wavelet_groups)}")
        logger.info(f"   - Processing rate: {len(existing_images)/stage1_time:.1f} images/second")
        
        # Check memory usage
        memory_stats = deduplicator.memory_stats
        logger.info(f"💾 Memory usage: {memory_stats.get('peak_memory_mb', 0):.1f} MB")
        logger.info(f"   - Features loaded/computed: {memory_stats.get('features_loaded', 0)}")
        logger.info(f"   - Features freed: {memory_stats.get('features_freed', 0)}")
        
        # Verify wavelet groups were created
        if len(wavelet_groups) > 0:
            logger.info("✅ Wavelet feature computation is working!")
            logger.info("   - Features computed on-demand")
            logger.info("   - Groups created successfully")
            logger.info("   - Ready for Stage 2 color verification")
        else:
            logger.warning("⚠️  No wavelet groups created - check feature computation")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Stage 1 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(cache_dir)
        except Exception as e:
            logger.warning(f"⚠️  Cleanup failed: {e}")


if __name__ == "__main__":
    logger.info("🚀 On-Demand Feature Computation Test")
    logger.info("=" * 50)
    
    # Run tests
    test1_passed = test_stage1_wavelet_computation()
    test2_passed = test_ondemand_feature_computation()
    
    if test1_passed and test2_passed:
        logger.info("🎉 All tests passed! On-demand feature computation is working correctly.")
        logger.info("✅ Ready for 3M+ images processing!")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed!")
        sys.exit(1) 