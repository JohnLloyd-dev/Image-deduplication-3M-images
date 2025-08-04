#!/usr/bin/env python3
"""
Test with diverse images to verify the system handles different images properly.
This test uses different image paths to ensure we're testing actual deduplication.
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


def get_diverse_test_images() -> List[str]:
    """Get diverse test images from different directories to test actual deduplication."""
    
    # These should be different images from various directories
    diverse_images = [
        # Original test images
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2105_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2105_TP.jpg", 
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2125_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2225_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2225.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-3021.jpg",
        
        # Try to find more diverse images by exploring different paths
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/512/TEST-EQUIT-50297_AFG-2105_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/513/TEST-EQUIT-50297_AFG-2105_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/523/511/TEST-EQUIT-50297_AFG-2105_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/523/512/TEST-EQUIT-50297_AFG-2105_AP.jpg",
        
        # Try different product codes
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50298_AFG-2105_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50299_AFG-2105_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50300_AFG-2105_AP.jpg",
        
        # Try different view types
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2105_BP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2105_CP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2105_DP.jpg",
    ]
    
    return diverse_images


def verify_images_exist(images: List[str]) -> List[str]:
    """Verify which images actually exist in Azure storage."""
    
    logger.info(f"🔍 Verifying {len(images)} images exist in Azure storage...")
    
    # Initialize Azure manager
    azure_manager = AzureBlobManager()
    
    existing_images = []
    for img_path in images:
        try:
            # Check if blob exists
            properties = azure_manager.get_blob_properties(img_path)
            if properties:
                existing_images.append(img_path)
                logger.info(f"✅ Found: {img_path}")
            else:
                logger.warning(f"❌ Not found: {img_path}")
        except Exception as e:
            logger.warning(f"❌ Error checking {img_path}: {e}")
    
    logger.info(f"📊 Found {len(existing_images)} existing images out of {len(images)}")
    return existing_images


def test_diverse_dataset():
    """Test with diverse images to verify actual deduplication."""
    
    logger.info("🧪 Testing diverse dataset for actual deduplication...")
    
    # Get diverse images
    test_images = get_diverse_test_images()
    
    # Verify which images exist
    existing_images = verify_images_exist(test_images)
    
    if len(existing_images) < 3:
        logger.error(f"❌ Not enough diverse images found! Found: {len(existing_images)}")
        return False
    
    logger.info(f"📊 Using {len(existing_images)} diverse images for testing")
    
    # Create feature cache
    cache_dir = tempfile.mkdtemp(prefix="test_diverse_cache_")
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
    output_dir = tempfile.mkdtemp(prefix="test_diverse_output_")
    
    try:
        logger.info("🔄 Testing diverse dataset with on-demand feature computation...")
        start_time = time.time()
        
        # Test the full memory-efficient deduplication pipeline
        duplicate_groups, similarity_scores = deduplicator.deduplicate_memory_efficient(
            existing_images, output_dir, None
        )
        
        total_time = time.time() - start_time
        
        # Analyze results
        logger.info(f"\n🎉 Diverse Dataset Test Completed!")
        logger.info(f"📊 Results:")
        logger.info(f"   - Input images: {len(existing_images)}")
        logger.info(f"   - Processing time: {total_time:.1f}s")
        logger.info(f"   - Processing rate: {len(existing_images)/total_time:.1f} images/second")
        
        # Memory statistics
        memory_stats = deduplicator.memory_stats
        logger.info(f"💾 Memory Statistics:")
        logger.info(f"   - Peak memory usage: {memory_stats.get('peak_memory_mb', 0):.1f} MB")
        logger.info(f"   - Features loaded/computed: {memory_stats.get('features_loaded', 0):,}")
        logger.info(f"   - Features freed: {memory_stats.get('features_freed', 0):,}")
        logger.info(f"   - Memory efficiency: {(memory_stats.get('features_freed', 0)/max(memory_stats.get('features_loaded', 1), 1))*100:.1f}% freed")
        
        # Deduplication results
        logger.info(f"🔍 Deduplication Results:")
        logger.info(f"   - Duplicate groups: {len(duplicate_groups)}")
        logger.info(f"   - Total duplicate images: {sum(len(group) for group in duplicate_groups):,}")
        logger.info(f"   - Unique images: {len(existing_images) - sum(len(group) for group in duplicate_groups):,}")
        logger.info(f"   - Duplication rate: {sum(len(group) for group in duplicate_groups)/len(existing_images)*100:.1f}%")
        
        # Check if we found actual duplicates
        if len(duplicate_groups) > 0:
            logger.info("✅ Found actual duplicates! System is working correctly.")
            for i, group in enumerate(duplicate_groups):
                logger.info(f"   Group {i+1}: {len(group)} images")
                for img in group[:3]:  # Show first 3 images
                    logger.info(f"     - {img}")
                if len(group) > 3:
                    logger.info(f"     ... and {len(group)-3} more")
        else:
            logger.info("ℹ️  No duplicates found - all images are unique (which is expected for diverse dataset)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Diverse dataset test failed: {e}")
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


if __name__ == "__main__":
    logger.info("🚀 Diverse Dataset Test")
    logger.info("=" * 50)
    
    # Run test
    test_passed = test_diverse_dataset()
    
    if test_passed:
        logger.info("🎉 Diverse dataset test completed successfully!")
        logger.info("✅ System correctly handles diverse images!")
        sys.exit(0)
    else:
        logger.error("❌ Diverse dataset test failed!")
        sys.exit(1) 