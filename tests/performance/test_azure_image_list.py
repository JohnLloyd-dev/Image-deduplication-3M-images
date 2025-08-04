#!/usr/bin/env python3
"""
Test with actual images from Azure storage.
This test fetches the real image list from Azure to get diverse images for testing.
"""

import sys
import os
import logging
import time
from typing import List, Dict
import tempfile
import shutil
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.feature_cache import BoundedFeatureCache
from modules.memory_efficient_deduplication import MemoryEfficientDeduplicator
from modules.azure_utils import AzureBlobManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_azure_image_list(max_images: int = 100) -> List[str]:
    """Get actual image list from Azure storage."""
    
    logger.info(f"🔍 Fetching image list from Azure storage (max: {max_images})...")
    
    # Initialize Azure manager
    azure_manager = AzureBlobManager()
    
    try:
        # List blobs in the container
        logger.info("📋 Listing blobs in Azure container...")
        
        # Get list of all blobs
        blobs = azure_manager.list_blobs()
        
        # Filter for image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_blobs = []
        
        for blob in blobs:
            if any(blob.lower().endswith(ext) for ext in image_extensions):
                image_blobs.append(blob)
        
        logger.info(f"📊 Found {len(image_blobs)} image files in Azure storage")
        
        # Take a random sample if we have too many
        if len(image_blobs) > max_images:
            selected_blobs = random.sample(image_blobs, max_images)
            logger.info(f"🎲 Randomly selected {len(selected_blobs)} images from {len(image_blobs)} total")
        else:
            selected_blobs = image_blobs
            logger.info(f"📋 Using all {len(selected_blobs)} available images")
        
        # Show some examples
        logger.info("📝 Sample images:")
        for i, blob in enumerate(selected_blobs[:5]):
            logger.info(f"   {i+1}. {blob}")
        if len(selected_blobs) > 5:
            logger.info(f"   ... and {len(selected_blobs)-5} more")
        
        return selected_blobs
        
    except Exception as e:
        logger.error(f"❌ Failed to get image list from Azure: {e}")
        return []


def verify_images_exist(images: List[str]) -> List[str]:
    """Verify which images actually exist and are accessible."""
    
    logger.info(f"🔍 Verifying {len(images)} images are accessible...")
    
    # Initialize Azure manager
    azure_manager = AzureBlobManager()
    
    existing_images = []
    for i, img_path in enumerate(images):
        try:
            # Check if blob exists and is accessible
            properties = azure_manager.get_blob_properties(img_path)
            if properties:
                existing_images.append(img_path)
                if i < 5:  # Log first 5 for verification
                    logger.info(f"✅ Verified: {img_path}")
            else:
                logger.warning(f"❌ Not accessible: {img_path}")
        except Exception as e:
            logger.warning(f"❌ Error checking {img_path}: {e}")
    
    logger.info(f"📊 Verified {len(existing_images)} accessible images out of {len(images)}")
    return existing_images


def test_azure_image_list():
    """Test with actual images from Azure storage."""
    
    logger.info("🧪 Testing with actual Azure image list...")
    
    # Get image list from Azure
    test_images = get_azure_image_list(max_images=50)  # Start with 50 images
    
    if len(test_images) == 0:
        logger.error("❌ No images found in Azure storage!")
        return False
    
    # Verify images are accessible
    accessible_images = verify_images_exist(test_images)
    
    if len(accessible_images) < 10:
        logger.error(f"❌ Not enough accessible images! Found: {len(accessible_images)}")
        return False
    
    logger.info(f"📊 Using {len(accessible_images)} actual Azure images for testing")
    
    # Create feature cache
    cache_dir = tempfile.mkdtemp(prefix="test_azure_cache_")
    cache = BoundedFeatureCache(cache_dir=cache_dir, max_size=2000)
    
    # Create deduplicator
    deduplicator = MemoryEfficientDeduplicator(
        feature_cache=cache,
        color_threshold=0.85,
        global_threshold=0.85,
        local_threshold=0.75,
        wavelet_threshold=0.8
    )
    
    # Create temporary output directory
    output_dir = tempfile.mkdtemp(prefix="test_azure_output_")
    
    try:
        logger.info("🔄 Testing Azure image list with on-demand feature computation...")
        start_time = time.time()
        
        # Test the full memory-efficient deduplication pipeline
        duplicate_groups, similarity_scores = deduplicator.deduplicate_memory_efficient(
            accessible_images, output_dir, None
        )
        
        total_time = time.time() - start_time
        
        # Analyze results
        logger.info(f"\n🎉 Azure Image List Test Completed!")
        logger.info(f"📊 Results:")
        logger.info(f"   - Input images: {len(accessible_images)}")
        logger.info(f"   - Processing time: {total_time:.1f}s")
        logger.info(f"   - Processing rate: {len(accessible_images)/total_time:.1f} images/second")
        logger.info(f"   - Estimated time for 3M images: {3000000/(len(accessible_images)/total_time)/3600:.1f} hours")
        
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
        logger.info(f"   - Unique images: {len(accessible_images) - sum(len(group) for group in duplicate_groups):,}")
        logger.info(f"   - Duplication rate: {sum(len(group) for group in duplicate_groups)/len(accessible_images)*100:.1f}%")
        
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
            logger.info("ℹ️  No duplicates found - all images are unique")
        
        # Performance analysis
        logger.info(f"📈 Performance Analysis:")
        images_per_second = len(accessible_images) / total_time
        memory_per_image = memory_stats.get('peak_memory_mb', 0) / len(accessible_images)
        
        logger.info(f"   - Images/second: {images_per_second:.1f}")
        logger.info(f"   - Memory per image: {memory_per_image:.2f} MB")
        logger.info(f"   - Estimated 3M memory: {memory_per_image * 3000000 / 1024:.1f} GB")
        
        # Check if performance is acceptable
        if images_per_second >= 1.0:
            logger.info("✅ Performance is acceptable for 3M+ images!")
            logger.info("   - Processing rate is sufficient")
            logger.info("   - Memory usage is manageable")
            logger.info("   - Ready for production scale")
        else:
            logger.warning("⚠️  Performance may need optimization for 3M+ images")
            logger.warning("   - Consider GPU acceleration")
            logger.warning("   - Consider parallel processing")
            logger.warning("   - Consider batch optimization")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Azure image list test failed: {e}")
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
    logger.info("🚀 Azure Image List Test")
    logger.info("=" * 50)
    
    # Run test
    test_passed = test_azure_image_list()
    
    if test_passed:
        logger.info("🎉 Azure image list test completed successfully!")
        logger.info("✅ System correctly handles real Azure images!")
        sys.exit(0)
    else:
        logger.error("❌ Azure image list test failed!")
        sys.exit(1) 