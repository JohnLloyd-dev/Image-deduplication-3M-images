#!/usr/bin/env python3
"""
Main entry point for the Memory-Efficient Image Deduplication Pipeline.
Uses the optimized deduplication logic for 3M+ images.
"""

import os
import sys
import logging
import tempfile
import shutil
import gc
from typing import List

# Fix OpenMP conflicts (Intel vs LLVM)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'  # Increased for full dataset
os.environ['LOKY_MAX_CPU_COUNT'] = '8'  # Increased CPU usage for full dataset

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.memory_efficient_deduplication import create_memory_efficient_deduplicator
from modules.color_optimized_deduplicator import create_color_optimized_deduplicator
from modules.whash_deduplicator import create_whash_deduplicator
from modules.feature_cache import BoundedFeatureCache
from modules.azure_utils import AzureBlobManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to see quality score computation and cache status
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the optimized image deduplication pipeline."""
    
    # Create a temporary directory for the report
    temp_dir = tempfile.mkdtemp()
    report_path = os.path.join(temp_dir, "deduplication_report.csv")
    
    try:
        logger.info("🚀 Starting Memory-Efficient Image Deduplication Pipeline...")
        logger.info("📁 Target Directory: Image_Dedup_Project/TestEquity/CompleteImageDataset/")
        
        # Initialize Azure Blob Manager with reduced connections
        azure_manager = AzureBlobManager()
        
        # Get images from the target directory
        logger.info("📋 Fetching images from Azure target directory...")
        image_paths = azure_manager.list_blobs('Image_Dedup_Project/TestEquity/CompleteImageDataset/')
        
        if not image_paths:
            logger.error("❌ No images found in the target directory!")
            return
        
        logger.info(f"✅ Found {len(image_paths)} images in target directory")
        
        # Process ALL images in the dataset
        all_images = image_paths
        logger.info(f"📊 Processing ALL {len(all_images)} images from the dataset")
        
        # Create integrated WHash-Color deduplicator for optimal performance
        logger.info("🔧 Creating integrated WHash-Color deduplicator...")
        
        # Create color-optimized deduplicator
        color_deduplicator = create_color_optimized_deduplicator(
            feature_cache=BoundedFeatureCache(max_size=2000),  # Increased for full dataset
            color_clusters=2000,
            parallel_processing=True,
            max_workers=8
        )
        
        # Create WHash deduplicator and integrate it
        whash_deduplicator = create_whash_deduplicator(
            hash_size=8,
            wavelet_level=2,
            threshold=0.85,
            enable_lsh=True,
            lsh_bands=4,
            lsh_rows_per_band=4
        )
        
        # Use the integrated deduplicator directly
        deduplicator = color_deduplicator
        whash_deduplicator = whash_deduplicator  # Keep reference for integration
        
        logger.info("✅ WHash-Color deduplicators created!")
        
        # Force garbage collection before starting
        gc.collect()
        
        # Run the complete integrated deduplication pipeline
        logger.info("🔄 Running WHash-Color integrated deduplication pipeline...")
        final_groups, similarity_scores = deduplicator.deduplicate_with_whash_integration(
            image_paths=all_images,
            output_dir=temp_dir,
            whash_deduplicator=whash_deduplicator
        )
        
        # Force garbage collection after pipeline
        gc.collect()
        
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
        logger.info(f"   - Original images: {len(all_images)}")
        logger.info(f"   - Final groups: {len(final_groups)}")
        logger.info(f"   - Total images in groups: {total_images_in_groups}")
        logger.info(f"   - Missing images: {len(all_images) - total_images_in_groups}")
        
        if total_images_in_groups == len(all_images):
            logger.info("✅ All images preserved through deduplication")
        else:
            logger.warning(f"⚠️  {len(all_images) - total_images_in_groups} images missing from final report")
        
        # Check report content
        import pandas as pd
        if os.path.exists(report_path):
            df = pd.read_csv(report_path)
            reported_images = set(df['Image Path'].tolist())
            
            logger.info(f"📋 Report Analysis:")
            logger.info(f"   - Report images: {len(reported_images)}")
            logger.info(f"   - Best images in report: {len(df[df['Status'] == 'Best'])}")
            logger.info(f"   - Duplicate images in report: {len(df[df['Status'] == 'Duplicate'])}")
            
            missing_images = set(all_images) - reported_images
            if missing_images:
                logger.error(f"❌ {len(missing_images)} images missing from report!")
                for img in list(missing_images)[:5]:  # Show first 5 missing
                    logger.error(f"   - Missing: {img}")
            else:
                logger.info("✅ All images present in report")
        
        logger.info("🎉 Memory-Efficient Deduplication Pipeline completed successfully!")
        
        # Copy report to a permanent location
        permanent_report_path = os.path.join(".", "deduplication_report.csv")
        shutil.copy2(report_path, permanent_report_path)
        logger.info(f"📄 Permanent report saved to: {permanent_report_path}")
        
        return {
            'success': True,
            'original_images': len(all_images),
            'final_groups': len(final_groups),
            'total_images_in_groups': total_images_in_groups,
            'missing_images': len(all_images) - total_images_in_groups,
            'report_path': permanent_report_path
        }
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Process interrupted by user")
        return {
            'success': False,
            'error': 'Process interrupted by user'
        }
    except Exception as e:
        logger.exception(f"❌ Pipeline failed with error: {e}")
        return {
            'success': False,
            'error': str(e)
        }
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"🧹 Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    results = main()
    if results['success']:
        print(f"✅ Pipeline complete. Processed {results['original_images']} images.")
        print(f"📊 Results: {results['final_groups']} groups, {results['total_images_in_groups']} images preserved")
        print(f"📄 Report: {results['report_path']}")
    else:
        print(f"❌ Pipeline failed: {results['error']}") 