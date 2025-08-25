#!/usr/bin/env python3
"""
Quick test to verify the fixes work without memory issues.
"""

import os
import sys
import logging

# Fix OpenMP conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.azure_utils import AzureBlobManager
from modules.memory_efficient_deduplication import create_memory_efficient_deduplicator
from modules.feature_cache import BoundedFeatureCache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_quick_fix():
    """Quick test to verify the fixes work."""
    
    try:
        logger.info("🚀 Testing quick fix...")
        
        # 1. Test Azure connection
        logger.info("📡 Testing Azure connection...")
        azure_manager = AzureBlobManager()
        logger.info("✅ Azure connection successful")
        
        # 2. Test image listing
        logger.info("📋 Testing image listing...")
        image_paths = azure_manager.list_blobs('Image_Dedup_Project/TestEquity/CompleteImageDataset/')
        logger.info(f"✅ Found {len(image_paths)} images in target directory")
        
        # 3. Test deduplicator creation
        logger.info("🔧 Testing deduplicator creation...")
        deduplicator = create_memory_efficient_deduplicator(
            feature_cache=BoundedFeatureCache(max_size=50)  # Reduced cache size
        )
        logger.info("✅ Deduplicator created successfully")
        
        # 4. Test with very small sample (10 images)
        logger.info("🧪 Testing with 10 images...")
        test_images = image_paths[:10]
        logger.info(f"📊 Testing with {len(test_images)} images")
        
        # 5. Test deduplication pipeline
        logger.info("🔄 Testing deduplication pipeline...")
        final_groups, similarity_scores = deduplicator.deduplicate_memory_efficient(
            image_paths=test_images,
            output_dir="./test_output"
        )
        logger.info(f"✅ Deduplication completed: {len(final_groups)} groups")
        
        # 6. Verify results
        total_images = sum(len(group) for group in final_groups)
        logger.info(f"📈 Results: {len(test_images)} original → {total_images} final")
        
        if total_images == len(test_images):
            logger.info("✅ All images preserved - Fix is working!")
            return True
        else:
            missing = len(test_images) - total_images
            logger.error(f"❌ {missing} images still missing!")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_quick_fix()
    if success:
        print("✅ Quick fix is working correctly!")
    else:
        print("❌ Quick fix still has issues!") 