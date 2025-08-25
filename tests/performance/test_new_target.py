#!/usr/bin/env python3
"""
Test script to run the main deduplication logic with the new target directory:
Image_Dedup_Project/TestEquity/CompleteImageDataset/
"""

import os
import sys
import logging
import tempfile
import shutil
from typing import List

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.memory_efficient_deduplication import create_memory_efficient_deduplicator
from modules.feature_cache import BoundedFeatureCache
from modules.azure_utils import AzureBlobManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_new_target_directory():
    """Test the main deduplication logic with the new target directory."""
    
    # Create a temporary directory for the report
    temp_dir = tempfile.mkdtemp()
    report_path = os.path.join(temp_dir, "new_target_report.csv")
    
    try:
        logger.info("🚀 Starting deduplication with new target directory...")
        logger.info("📁 Target Directory: Image_Dedup_Project/TestEquity/CompleteImageDataset/")
        
        # Initialize Azure Blob Manager
        azure_manager = AzureBlobManager()
        
        # Get images from the new target directory
        logger.info("📋 Fetching images from new target directory...")
        image_paths = azure_manager.list_blobs('Image_Dedup_Project/TestEquity/CompleteImageDataset/')
        
        if not image_paths:
            logger.error("❌ No images found in the new target directory!")
            return
        
        # Limit to first 50 images for testing
        test_images = image_paths[:50]
        logger.info(f"📊 Testing with {len(test_images)} images from new target directory")
        
        # Create deduplicator
        deduplicator = create_memory_efficient_deduplicator(
            feature_cache=BoundedFeatureCache(max_size=1000)
        )
        
        # Run the complete deduplication pipeline
        logger.info("🔄 Running deduplication pipeline...")
        final_groups, similarity_scores = deduplicator.deduplicate_memory_efficient(
            image_paths=test_images,
            output_dir=temp_dir
        )
        
        # Generate the final report
        logger.info("📊 Generating final report...")
        report_path = deduplicator.create_report(final_groups, similarity_scores, temp_dir)
        
        logger.info(f"✅ Deduplication completed successfully!")
        logger.info(f"📄 Report saved to: {report_path}")
        
        # Analyze results
        logger.info("🔍 Analyzing results...")
        
        # Count images in final groups
        total_images_in_groups = sum(len(group) for group in final_groups)
        logger.info(f"📈 Final Results:")
        logger.info(f"   - Original images: {len(test_images)}")
        logger.info(f"   - Final groups: {len(final_groups)}")
        logger.info(f"   - Total images in groups: {total_images_in_groups}")
        logger.info(f"   - Missing images: {len(test_images) - total_images_in_groups}")
        
        if total_images_in_groups == len(test_images):
            logger.info("✅ All images preserved through deduplication")
        else:
            logger.warning(f"⚠️  {len(test_images) - total_images_in_groups} images missing from final report")
        
        # Check report content
        import pandas as pd
        if os.path.exists(report_path):
            df = pd.read_csv(report_path)
            reported_images = set(df['Image Path'].tolist())
            
            logger.info(f"📋 Report Analysis:")
            logger.info(f"   - Report images: {len(reported_images)}")
            logger.info(f"   - Best images in report: {len(df[df['Status'] == 'Best'])}")
            logger.info(f"   - Duplicate images in report: {len(df[df['Status'] == 'Duplicate'])}")
            
            missing_images = set(test_images) - reported_images
            if missing_images:
                logger.error(f"❌ {len(missing_images)} images missing from report!")
                for img in list(missing_images)[:5]:  # Show first 5 missing
                    logger.error(f"   - Missing: {img}")
            else:
                logger.info("✅ All images present in report")
        
        logger.info("🎉 New target directory test completed successfully!")
        
    except Exception as e:
        logger.exception(f"❌ Test failed with error: {e}")
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"🧹 Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    test_new_target_directory() 