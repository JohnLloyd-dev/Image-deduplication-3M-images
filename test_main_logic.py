#!/usr/bin/env python3
"""
Quick test to verify main logic is working with new target directory.
"""

import os
import sys
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.azure_utils import AzureBlobManager
from modules.memory_efficient_deduplication import create_memory_efficient_deduplicator
from modules.feature_cache import BoundedFeatureCache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_main_logic():
    """Quick test of main logic with new target directory."""
    
    try:
        logger.info("🚀 Testing main logic with new target directory...")
        
        # 1. Test Azure connection
        logger.info("📡 Testing Azure connection...")
        azure_manager = AzureBlobManager()
        logger.info("✅ Azure connection successful")
        
        # 2. Test image listing
        logger.info("📋 Testing image listing...")
        image_paths = azure_manager.list_blobs('Image_Dedup_Project/TestEquity/CompleteImageDataset/')
        logger.info(f"✅ Found {len(image_paths)} images in new target directory")
        
        # 3. Test deduplicator creation
        logger.info("🔧 Testing deduplicator creation...")
        deduplicator = create_memory_efficient_deduplicator(
            feature_cache=BoundedFeatureCache(max_size=100)
        )
        logger.info("✅ Deduplicator created successfully")
        
        # 4. Test with small sample
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
        
        # 6. Test report generation
        logger.info("📊 Testing report generation...")
        report_path = deduplicator.create_report(final_groups, similarity_scores, "./test_output")
        logger.info(f"✅ Report generated: {report_path}")
        
        # 7. Verify results
        total_images = sum(len(group) for group in final_groups)
        logger.info(f"📈 Results: {len(test_images)} original → {total_images} final")
        
        if total_images == len(test_images):
            logger.info("✅ All images preserved - Main logic working perfectly!")
        else:
            logger.warning(f"⚠️  {len(test_images) - total_images} images missing")
        
        logger.info("🎉 Main logic test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_main_logic()
    if success:
        print("✅ Main logic is working correctly!")
    else:
        print("❌ Main logic has issues!") 