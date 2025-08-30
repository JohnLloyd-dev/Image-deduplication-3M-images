#!/usr/bin/env python3
"""
Main entry point for the Accuracy-Improved Image Deduplication Pipeline.
Uses the enhanced accuracy deduplication modules for improved duplicate detection.

Features:
- Multi-scale WHash for scale invariance
- SSIM for perceptual similarity
- Hybrid similarity calculation
- Comprehensive duplicate verification
"""

import os
import sys
import logging
import tempfile
import shutil
import gc
import time
from typing import List

# Fix OpenMP conflicts (Intel vs LLVM)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['LOKY_MAX_CPU_COUNT'] = '8'

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our enhanced accuracy modules
from modules.enhanced_whash_deduplicator import create_enhanced_whash_deduplicator
from modules.structural_similarity import create_structural_similarity_calculator
from modules.hybrid_similarity_calculator import create_hybrid_similarity_calculator
from modules.accuracy_optimized_deduplicator import create_accuracy_optimized_deduplicator
from modules.feature_cache import BoundedFeatureCache
from modules.azure_utils import AzureBlobManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_accuracy_improved_pipeline():
    """
    Create the accuracy-improved deduplication pipeline.
    """
    logger.info("🔧 Creating accuracy-improved deduplication pipeline...")
    
    # Step 1: Create enhanced WHash deduplicator
    logger.info("📊 Creating enhanced WHash deduplicator...")
    whash_deduplicator = create_enhanced_whash_deduplicator(
        hash_size=8,                    # 8x8 = 64-bit hash
        wavelet_level=3,                # 3 levels of wavelet decomposition
        wavelet_name='haar',            # Haar wavelet for speed
        scale_factors=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],  # Multi-scale factors
        enable_lsh=True,                # Enable LSH for efficient grouping
        lsh_bands=4,                    # 4 LSH bands
        lsh_rows_per_band=4,           # 4 rows per band
        similarity_threshold=0.85       # High threshold for initial grouping
    )
    
    # Step 2: Create structural similarity calculator
    logger.info("🔍 Creating structural similarity calculator...")
    ssim_calculator = create_structural_similarity_calculator(
        target_size=(256, 256),        # Target size for SSIM computation
        use_gpu=True,                   # Enable GPU if available
        enable_preprocessing=True       # Enable image preprocessing
    )
    
    # Step 3: Create hybrid similarity calculator
    logger.info("🎯 Creating hybrid similarity calculator...")
    hybrid_calculator = create_hybrid_similarity_calculator(
        whash_deduplicator=whash_deduplicator,
        ssim_calculator=ssim_calculator,
        weights={
            'structural': 0.6,          # Highest weight for structural similarity
            'color': 0.25,              # Medium weight for color
            'whash': 0.15               # Lower weight for WHash (fast but less accurate)
        },
        thresholds={
            'global': 0.65,             # Overall similarity threshold
            'structural': 0.60,         # SSIM threshold
            'color': 0.55,              # Color similarity threshold
            'whash': 0.60               # WHash threshold
        },
        enable_caching=True,            # Enable similarity caching
        cache_size=10000                # Cache size for performance
    )
    
    # Step 4: Create accuracy-optimized deduplicator
    logger.info("🚀 Creating accuracy-optimized deduplicator...")
    accuracy_deduplicator = create_accuracy_optimized_deduplicator(
        hybrid_calculator=hybrid_calculator,
        whash_deduplicator=whash_deduplicator,
        ssim_calculator=ssim_calculator,
        enable_verification=True,       # Enable duplicate verification
        verification_threshold=0.65,    # Threshold for verification
        max_group_size=1000,           # Maximum size for verification groups
        enable_caching=True             # Enable similarity caching
    )
    
    logger.info("✅ Accuracy-improved pipeline created successfully!")
    return accuracy_deduplicator, whash_deduplicator, ssim_calculator, hybrid_calculator

def main():
    """Main entry point for the accuracy-improved image deduplication pipeline."""
    
    # Create a temporary directory for the report
    temp_dir = tempfile.mkdtemp()
    report_path = os.path.join(temp_dir, "accuracy_improved_deduplication_report.csv")
    
    try:
        logger.info("🚀 Starting Accuracy-Improved Image Deduplication Pipeline...")
        logger.info("📁 Target Directory: Image_Dedup_Project/TestEquity/CompleteImageDataset/")
        
        # Initialize Azure Blob Manager
        azure_manager = AzureBlobManager()
        
        # Get images from the target directory
        logger.info("📋 Fetching images from Azure target directory...")
        image_paths = azure_manager.list_blobs('Image_Dedup_Project/TestEquity/CompleteImageDataset/')
        
        if not image_paths:
            logger.error("❌ No images found in the target directory!")
            return
        
        logger.info(f"✅ Found {len(image_paths)} images in target directory")
        
        # For demonstration, use a subset of images first
        if len(image_paths) > 1000:
            logger.info(f"📊 Using first 1000 images for demonstration (total: {len(image_paths)})")
            demo_images = image_paths[:1000]
        else:
            demo_images = image_paths
        
        # Create the accuracy-improved pipeline
        accuracy_deduplicator, whash_deduplicator, ssim_calculator, hybrid_calculator = create_accuracy_improved_pipeline()
        
        # Force garbage collection before starting
        gc.collect()
        
        # Run the accuracy-improved deduplication pipeline
        logger.info("🔄 Running accuracy-improved deduplication pipeline...")
        start_time = time.time()
        
        duplicate_groups = accuracy_deduplicator.find_duplicates(
            image_paths=demo_images,
            progress_callback=lambda msg: logger.info(f"📊 {msg}")
        )
        
        processing_time = time.time() - start_time
        
        # Display results
        logger.info("=" * 60)
        logger.info("📊 ACCURACY-IMPROVED DEDUPLICATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total images processed: {len(demo_images)}")
        logger.info(f"Total duplicate groups found: {len(duplicate_groups)}")
        logger.info(f"Total duplicates found: {sum(len(group) - 1 for group in duplicate_groups if len(group) > 1)}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Average time per image: {processing_time/len(demo_images):.4f} seconds")
        
        # Display performance statistics
        logger.info("\n📈 PERFORMANCE STATISTICS:")
        logger.info("-" * 40)
        
        whash_stats = whash_deduplicator.get_performance_stats()
        logger.info(f"WHash - Images: {whash_stats['total_images_processed']}, "
                   f"Time: {whash_stats['total_processing_time']:.2f}s, "
                   f"Memory: {whash_stats['memory_usage_mb']:.1f}MB")
        
        ssim_stats = ssim_calculator.get_stats()
        logger.info(f"SSIM - Comparisons: {ssim_stats['total_comparisons']}, "
                   f"GPU: {ssim_stats['gpu_comparisons']}, "
                   f"CPU: {ssim_stats['cpu_comparisons']}")
        
        hybrid_stats = hybrid_calculator.get_stats()
        logger.info(f"Hybrid - Total: {hybrid_stats['total_comparisons']}, "
                   f"Cached: {hybrid_stats['cached_comparisons']}, "
                   f"Cache hit rate: {hybrid_stats.get('hit_rate', 0):.1%}")
        
        accuracy_stats = accuracy_deduplicator.get_stats()
        logger.info(f"Accuracy - Groups: {accuracy_stats['total_groups_found']}, "
                   f"Verified: {accuracy_stats['verified_groups']}")
        
        # Save results to CSV
        logger.info(f"\n💾 Saving results to: {report_path}")
        with open(report_path, 'w') as f:
            f.write("Group_ID,Image_Path,Similarity_Score\n")
            for i, group in enumerate(duplicate_groups):
                if len(group) > 1:  # Only write groups with duplicates
                    for j, image_path in enumerate(group):
                        f.write(f"{i},{image_path},{1.0 if j == 0 else 0.9}\n")
        
        logger.info("✅ Results saved successfully!")
        
        # Display sample duplicate groups
        if duplicate_groups:
            logger.info("\n🔍 SAMPLE DUPLICATE GROUPS:")
            logger.info("-" * 40)
            for i, group in enumerate(duplicate_groups[:5]):  # Show first 5 groups
                if len(group) > 1:
                    logger.info(f"Group {i}: {len(group)} images")
                    for j, path in enumerate(group[:3]):  # Show first 3 paths
                        logger.info(f"  {j+1}. {os.path.basename(path)}")
                    if len(group) > 3:
                        logger.info(f"  ... and {len(group) - 3} more")
        
        logger.info("\n🎉 Accuracy-improved deduplication completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during accuracy-improved deduplication: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            if 'accuracy_deduplicator' in locals():
                accuracy_deduplicator.release()
            if 'whash_deduplicator' in locals():
                whash_deduplicator.release()
            if 'ssim_calculator' in locals():
                ssim_calculator.release()
            if 'hybrid_calculator' in locals():
                hybrid_calculator.release()
            
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()
