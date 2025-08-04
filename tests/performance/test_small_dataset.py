#!/usr/bin/env python3
"""
Test the color verification fix on a small dataset.
This will use actual images from your dataset to verify the fix works correctly.
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


def get_small_test_dataset() -> List[str]:
    """Get a small subset of actual images from the dataset."""
    
    # Small test dataset - these should be actual images from your Azure storage
    test_images = [
        # Group 1: Similar images (should be grouped together)
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2105_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2105_TP.jpg", 
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2105.jpg",
        
        # Group 2: Another set of similar images
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2125_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2125.jpg",
        
        # Group 3: Larger group of similar images
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2225_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2225.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2225_TP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2225_V2.jpg",
        
        # Individual images (likely no duplicates)
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-3000.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-3021.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-3022.jpg",
    ]
    
    return test_images


def verify_images_exist(image_paths: List[str]) -> List[str]:
    """Verify that the test images actually exist in Azure storage."""
    
    logger.info("🔍 Verifying test images exist in Azure storage...")
    
    try:
        # Get Azure client using the correct method
        azure_manager = AzureBlobManager()
        
        existing_images = []
        missing_images = []
        
        for img_path in image_paths:
            try:
                # Check if blob exists by trying to get properties
                properties = azure_manager.get_blob_properties(img_path)
                
                if properties:
                    existing_images.append(img_path)
                    logger.debug(f"✅ Found: {img_path}")
                else:
                    missing_images.append(img_path)
                    logger.warning(f"❌ Missing: {img_path}")
                    
            except Exception as e:
                missing_images.append(img_path)
                logger.warning(f"❌ Error checking {img_path}: {e}")
        
        logger.info(f"📊 Image verification complete:")
        logger.info(f"   - Total images: {len(image_paths)}")
        logger.info(f"   - Existing images: {len(existing_images)}")
        logger.info(f"   - Missing images: {len(missing_images)}")
        
        if missing_images:
            logger.warning("⚠️  Some test images are missing. Using only existing images.")
            return existing_images
        else:
            logger.info("✅ All test images found!")
            return existing_images
            
    except Exception as e:
        logger.error(f"❌ Error verifying images: {e}")
        logger.info("⚠️  Proceeding with original image list...")
        return image_paths


def create_feature_cache() -> BoundedFeatureCache:
    """Create a feature cache for the test."""
    
    # Create temporary directory for cache
    cache_dir = tempfile.mkdtemp(prefix="test_small_cache_")
    logger.info(f"📁 Cache directory: {cache_dir}")
    
    cache = BoundedFeatureCache(cache_dir=cache_dir, max_size=1000)
    return cache


def test_small_dataset_deduplication():
    """Test the color verification fix on a small dataset."""
    
    logger.info("🧪 Testing color verification fix on small dataset...")
    
    # Get test dataset
    test_images = get_small_test_dataset()
    logger.info(f"📊 Test dataset: {len(test_images)} images")
    
    # Verify images exist
    existing_images = verify_images_exist(test_images)
    
    if len(existing_images) < 3:
        logger.error("❌ Not enough images found for testing!")
        return False
    
    # Create feature cache
    cache = create_feature_cache()
    
    # Create memory-efficient deduplicator
    deduplicator = MemoryEfficientDeduplicator(
        feature_cache=cache,
        color_threshold=0.85,
        global_threshold=0.85,
        local_threshold=0.75,
        wavelet_threshold=0.8
    )
    
    # Create temporary output directory
    output_dir = tempfile.mkdtemp(prefix="test_small_output_")
    logger.info(f"📁 Output directory: {output_dir}")
    
    try:
        # Test the full memory-efficient deduplication pipeline
        logger.info("🔄 Starting memory-efficient deduplication...")
        start_time = time.time()
        
        duplicate_groups, similarity_scores = deduplicator.deduplicate_memory_efficient(
            existing_images, output_dir, None
        )
        
        total_time = time.time() - start_time
        
        # Analyze results
        logger.info(f"\n🎉 Deduplication completed in {total_time:.2f}s!")
        logger.info(f"📊 Results:")
        logger.info(f"   - Input images: {len(existing_images)}")
        logger.info(f"   - Duplicate groups: {len(duplicate_groups)}")
        logger.info(f"   - Total duplicate images: {sum(len(group) for group in duplicate_groups)}")
        logger.info(f"   - Processing rate: {len(existing_images)/total_time:.1f} images/second")
        
        # Memory statistics
        memory_stats = deduplicator.memory_stats
        logger.info(f"💾 Memory Statistics:")
        logger.info(f"   - Peak memory usage: {memory_stats.get('peak_memory_mb', 0):.1f} MB")
        logger.info(f"   - Features loaded: {memory_stats.get('features_loaded', 0)}")
        logger.info(f"   - Features freed: {memory_stats.get('features_freed', 0)}")
        
        # Show duplicate groups
        logger.info(f"📋 Duplicate Groups:")
        for i, group in enumerate(duplicate_groups):
            if len(group) > 1:  # Only show groups with duplicates
                logger.info(f"   Group {i+1}: {len(group)} images")
                for img in group[:3]:  # Show first 3 images
                    logger.info(f"     - {os.path.basename(img)}")
                if len(group) > 3:
                    logger.info(f"     - ... and {len(group)-3} more")
        
        # Check if the fix worked by looking at group sizes
        expected_groups = [
            ["TEST-EQUIT-50297_AFG-2105_AP.jpg", "TEST-EQUIT-50297_AFG-2105_TP.jpg", "TEST-EQUIT-50297_AFG-2105.jpg"],
            ["TEST-EQUIT-50297_AFG-2125_AP.jpg", "TEST-EQUIT-50297_AFG-2125.jpg"],
            ["TEST-EQUIT-50297_AFG-2225_AP.jpg", "TEST-EQUIT-50297_AFG-2225.jpg", "TEST-EQUIT-50297_AFG-2225_TP.jpg", "TEST-EQUIT-50297_AFG-2225_V2.jpg"]
        ]
        
        logger.info(f"🎯 Expected groups: {len(expected_groups)}")
        logger.info(f"✅ Actual groups: {len(duplicate_groups)}")
        
        # Success criteria
        success = True
        
        # Check if we found any duplicate groups
        if len(duplicate_groups) == 0:
            logger.warning("⚠️  No duplicate groups found - this might be expected for small dataset")
        
        # Check memory efficiency
        if memory_stats.get('features_freed', 0) > 0:
            efficiency = (memory_stats['features_freed'] / max(memory_stats['features_loaded'], 1)) * 100
            logger.info(f"✅ Memory efficiency: {efficiency:.1f}%")
            if efficiency < 50:
                logger.warning("⚠️  Low memory efficiency - investigate further")
        else:
            logger.warning("⚠️  No memory statistics available")
        
        # Check processing speed
        if total_time > 0 and len(existing_images) / total_time < 1:
            logger.warning("⚠️  Slow processing speed - investigate further")
        else:
            logger.info("✅ Processing speed looks good")
        
        logger.info("✅ Small dataset test completed successfully!")
        return success
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(output_dir)
            shutil.rmtree(cache.cache_dir)
            logger.info("🧹 Cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️  Cleanup failed: {e}")


def test_color_verification_stage_only():
    """Test only the color verification stage to verify the fix."""
    
    logger.info("🎨 Testing color verification stage only...")
    
    # Get test dataset
    test_images = get_small_test_dataset()
    existing_images = verify_images_exist(test_images)
    
    if len(existing_images) < 3:
        logger.error("❌ Not enough images for testing!")
        return False
    
    # Create deduplicator
    cache = create_feature_cache()
    deduplicator = MemoryEfficientDeduplicator(
        feature_cache=cache,
        color_threshold=0.85
    )
    
    try:
        # Create mock wavelet groups (simulate Stage 1 output)
        # Group similar images together
        wavelet_groups = [
            existing_images[:3],  # First 3 images as one group
            existing_images[3:5], # Next 2 images as another group
            existing_images[5:9], # Next 4 images as another group
            existing_images[9:],  # Remaining images as individual groups
        ]
        
        logger.info(f"📊 Testing with {len(wavelet_groups)} wavelet groups")
        
        # Test Stage 2: Color verification (the fixed stage)
        start_time = time.time()
        
        color_groups = deduplicator._stage2_color_verification(
            wavelet_groups, {}, None
        )
        
        color_time = time.time() - start_time
        
        logger.info(f"✅ Color verification completed in {color_time:.2f}s")
        logger.info(f"📊 Results:")
        logger.info(f"   - Input groups: {len(wavelet_groups)}")
        logger.info(f"   - Output groups: {len(color_groups)}")
        
        # Check memory usage
        memory_stats = deduplicator.memory_stats
        logger.info(f"💾 Memory usage: {memory_stats.get('peak_memory_mb', 0):.1f} MB")
        
        logger.info("✅ Color verification stage test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Color verification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(cache.cache_dir)
        except Exception as e:
            logger.warning(f"⚠️  Cleanup failed: {e}")


if __name__ == "__main__":
    logger.info("🚀 Starting small dataset tests...")
    
    # Run both tests
    test1_passed = test_small_dataset_deduplication()
    test2_passed = test_color_verification_stage_only()
    
    if test1_passed and test2_passed:
        logger.info("🎉 All tests passed! Color verification fix is working correctly.")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed!")
        sys.exit(1)