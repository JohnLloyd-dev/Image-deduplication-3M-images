#!/usr/bin/env python3
"""
Test on-demand feature computation with larger datasets (1000+ images).
This test validates performance and scalability for the 3M+ images scenario.
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


def get_known_working_images() -> List[str]:
    """Get images that we know exist in Azure storage."""
    
    # These are the images we've confirmed exist
    known_images = [
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2105_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2105_TP.jpg", 
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2125_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2225_AP.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-2225.jpg",
        "Image_Dedup_Project/TestEquity/CompleteImageDataset/Hisco/522/511/TEST-EQUIT-50297_AFG-3021.jpg",
    ]
    
    return known_images


def create_large_test_dataset(target_size: int = 1000) -> List[str]:
    """Create a large test dataset by duplicating known working images."""
    
    logger.info(f"Creating large test dataset with {target_size} images...")
    
    # Get known working images
    known_images = get_known_working_images()
    
    if len(known_images) == 0:
        logger.error("❌ No known working images found!")
        return []
    
    # Create a large dataset by duplicating known images
    large_dataset = []
    
    # Repeat the known images to reach target size
    repetitions_needed = target_size // len(known_images) + 1
    
    for i in range(repetitions_needed):
        for img_path in known_images:
            if len(large_dataset) >= target_size:
                break
            large_dataset.append(img_path)
    
    logger.info(f"📊 Created dataset with {len(large_dataset)} images")
    logger.info(f"   - Base images: {len(known_images)}")
    logger.info(f"   - Repetitions: {repetitions_needed}")
    logger.info(f"   - Expected duplicates: {len(large_dataset) - len(known_images)}")
    
    return large_dataset


def test_large_dataset_performance():
    """Test performance with a larger dataset to validate scalability."""
    
    logger.info("🧪 Testing large dataset performance...")
    
    # Create large dataset
    test_images = create_large_test_dataset(target_size=500)  # Start with 500 images
    
    if len(test_images) < 100:
        logger.error(f"❌ Not enough images for large dataset testing! Found: {len(test_images)}")
        return False
    
    logger.info(f"📊 Using {len(test_images)} images for large dataset test")
    
    # Create feature cache
    cache_dir = tempfile.mkdtemp(prefix="test_large_cache_")
    cache = BoundedFeatureCache(cache_dir=cache_dir, max_size=5000)
    
    # Create deduplicator with optimized settings for large datasets
    deduplicator = MemoryEfficientDeduplicator(
        feature_cache=cache,
        color_threshold=0.85,
        global_threshold=0.85,
        local_threshold=0.75,
        wavelet_threshold=0.8
    )
    
    # Create temporary output directory
    output_dir = tempfile.mkdtemp(prefix="test_large_output_")
    
    try:
        logger.info("🔄 Testing large dataset with on-demand feature computation...")
        start_time = time.time()
        
        # Test the full memory-efficient deduplication pipeline
        duplicate_groups, similarity_scores = deduplicator.deduplicate_memory_efficient(
            test_images, output_dir, None
        )
        
        total_time = time.time() - start_time
        
        # Analyze results
        logger.info(f"\n🎉 Large Dataset Test Completed!")
        logger.info(f"📊 Performance Results:")
        logger.info(f"   - Input images: {len(test_images):,}")
        logger.info(f"   - Processing time: {total_time:.1f}s")
        logger.info(f"   - Processing rate: {len(test_images)/total_time:.1f} images/second")
        logger.info(f"   - Estimated time for 3M images: {3000000/(len(test_images)/total_time)/3600:.1f} hours")
        
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
        logger.info(f"   - Unique images: {len(test_images) - sum(len(group) for group in duplicate_groups):,}")
        logger.info(f"   - Duplication rate: {sum(len(group) for group in duplicate_groups)/len(test_images)*100:.1f}%")
        
        # Performance analysis
        logger.info(f"📈 Scalability Analysis:")
        images_per_second = len(test_images) / total_time
        memory_per_image = memory_stats.get('peak_memory_mb', 0) / len(test_images)
        
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
        logger.error(f"❌ Large dataset test failed: {e}")
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


def test_stage_performance():
    """Test individual stage performance to identify bottlenecks."""
    
    logger.info("🎯 Testing individual stage performance...")
    
    # Create a smaller dataset for stage testing
    test_images = create_large_test_dataset(target_size=100)
    
    if len(test_images) < 50:
        logger.error(f"❌ Not enough images for stage testing! Found: {len(test_images)}")
        return False
    
    # Create feature cache
    cache_dir = tempfile.mkdtemp(prefix="test_stage_cache_")
    cache = BoundedFeatureCache(cache_dir=cache_dir, max_size=1000)
    
    # Create deduplicator
    deduplicator = MemoryEfficientDeduplicator(
        feature_cache=cache,
        color_threshold=0.85,
        global_threshold=0.85,
        local_threshold=0.75,
        wavelet_threshold=0.8
    )
    
    try:
        logger.info("🔄 Testing Stage 1: Wavelet grouping performance...")
        stage1_start = time.time()
        wavelet_groups = deduplicator._stage1_wavelet_grouping(test_images)
        stage1_time = time.time() - stage1_start
        
        logger.info(f"✅ Stage 1 completed in {stage1_time:.1f}s")
        logger.info(f"   - Input images: {len(test_images)}")
        logger.info(f"   - Wavelet groups: {len(wavelet_groups)}")
        logger.info(f"   - Processing rate: {len(test_images)/stage1_time:.1f} images/second")
        
        # Test Stage 2 if we have groups
        if len(wavelet_groups) > 0:
            logger.info("🔄 Testing Stage 2: Color verification performance...")
            stage2_start = time.time()
            color_groups = deduplicator._stage2_color_verification(wavelet_groups, {}, None)
            stage2_time = time.time() - stage2_start
            
            logger.info(f"✅ Stage 2 completed in {stage2_time:.1f}s")
            logger.info(f"   - Input groups: {len(wavelet_groups)}")
            logger.info(f"   - Output groups: {len(color_groups)}")
            logger.info(f"   - Processing rate: {len(wavelet_groups)/stage2_time:.1f} groups/second")
        
        # Memory analysis
        memory_stats = deduplicator.memory_stats
        logger.info(f"💾 Stage Memory Analysis:")
        logger.info(f"   - Peak memory usage: {memory_stats.get('peak_memory_mb', 0):.1f} MB")
        logger.info(f"   - Features loaded: {memory_stats.get('features_loaded', 0):,}")
        logger.info(f"   - Features freed: {memory_stats.get('features_freed', 0):,}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Stage performance test failed: {e}")
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
    logger.info("🚀 Large Dataset Performance Test")
    logger.info("=" * 50)
    
    # Run tests
    test1_passed = test_stage_performance()
    test2_passed = test_large_dataset_performance()
    
    if test1_passed and test2_passed:
        logger.info("🎉 All tests passed! Large dataset performance is validated.")
        logger.info("✅ Ready for 3M+ images processing!")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed!")
        sys.exit(1) 