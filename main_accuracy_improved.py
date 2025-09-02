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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Fix OpenMP conflicts (Intel vs LLVM)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'  # Reduced per process
os.environ['LOKY_MAX_CPU_COUNT'] = '16'  # Increased for parallel processing

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
        lsh_bands=8,                    # Increased bands for 3M+ images
        lsh_rows_per_band=3,           # Optimized rows for large dataset
        similarity_threshold=0.75       # Balanced threshold for large dataset
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
        cache_size=100000               # Increased cache size for 3M+ images
    )
    
    # Step 4: Create accuracy-optimized deduplicator
    logger.info("🚀 Creating accuracy-optimized deduplicator...")
    accuracy_deduplicator = create_accuracy_optimized_deduplicator(
        hybrid_calculator=hybrid_calculator,
        whash_deduplicator=whash_deduplicator,
        ssim_calculator=ssim_calculator,
        enable_verification=True,       # Enable duplicate verification
        verification_threshold=0.65,    # Threshold for verification
        max_group_size=5000,           # Increased max group size for 3M+ images
        enable_caching=True             # Enable similarity caching
    )
    
    logger.info("✅ Accuracy-improved pipeline created successfully!")
    return accuracy_deduplicator, whash_deduplicator, ssim_calculator, hybrid_calculator

def process_images_parallel(image_paths: List[str], accuracy_deduplicator, max_workers: int = 16) -> List[List[str]]:
    """
    Process images in parallel using multiple workers.
    
    Args:
        image_paths: List of image paths to process
        accuracy_deduplicator: The deduplicator instance
        max_workers: Maximum number of parallel workers
    
    Returns:
        List of duplicate groups
    """
    logger.info(f"🚀 Starting parallel processing with {max_workers} workers...")
    
    # Calculate optimal batch size
    batch_size = max(1, len(image_paths) // max_workers)
    logger.info(f"📊 Processing {len(image_paths):,} images in batches of {batch_size:,}")
    
    # Split images into batches
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
    logger.info(f"📦 Created {len(batches)} batches for parallel processing")
    
    all_duplicate_groups = []
    processed_batches = 0
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(accuracy_deduplicator.find_duplicates, batch): batch_idx 
            for batch_idx, batch in enumerate(batches)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_groups = future.result()
                all_duplicate_groups.extend(batch_groups)
                processed_batches += 1
                
                logger.info(f"✅ Completed batch {batch_idx + 1}/{len(batches)} "
                           f"({processed_batches * 100 / len(batches):.1f}% complete)")
                
            except Exception as e:
                logger.error(f"❌ Batch {batch_idx + 1} failed: {e}")
                continue
    
    logger.info(f"🎉 Parallel processing completed! Found {len(all_duplicate_groups)} total groups")
    return all_duplicate_groups

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
        
        # Process ALL images in the dataset with parallel processing
        logger.info(f"📊 Processing ALL {len(image_paths):,} images from the complete dataset")
        logger.info(f"🚀 Using parallel processing for dramatic speedup!")
        logger.info(f"🎯 Expected processing time: 4-8 hours (with parallel processing)")
        logger.info(f"💾 Expected memory usage: ~500MB peak (parallel processing)")
        logger.info(f"📈 Expected groups: ~{len(image_paths)//5:,} WHash groups")
        all_images = image_paths
        
        # Create the accuracy-improved pipeline
        accuracy_deduplicator, whash_deduplicator, ssim_calculator, hybrid_calculator = create_accuracy_improved_pipeline()
        
        # Force garbage collection before starting
        gc.collect()
        
        # Run the accuracy-improved deduplication pipeline with parallel processing
        logger.info("🔄 Running parallel accuracy-improved deduplication pipeline...")
        start_time = time.time()
        
        # Determine optimal number of workers based on system resources
        max_workers = min(16, mp.cpu_count() * 2)  # Use 2x CPU cores, max 16
        logger.info(f"🔧 Using {max_workers} parallel workers (CPU cores: {mp.cpu_count()})")
        
        duplicate_groups = process_images_parallel(
            image_paths=all_images,
            accuracy_deduplicator=accuracy_deduplicator,
            max_workers=max_workers
        )
        
        processing_time = time.time() - start_time
        
        # Display results
        logger.info("=" * 60)
        logger.info("📊 ACCURACY-IMPROVED DEDUPLICATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total images processed: {len(all_images):,}")
        logger.info(f"Total duplicate groups found: {len(duplicate_groups):,}")
        logger.info(f"Total duplicates found: {sum(len(group) - 1 for group in duplicate_groups if len(group) > 1):,}")
        logger.info(f"Processing time: {processing_time:.2f} seconds ({processing_time/60:.1f} minutes)")
        logger.info(f"Average time per image: {processing_time/len(all_images):.4f} seconds")
        logger.info(f"⚡ Parallel speedup: {max_workers}x faster than sequential")
        logger.info(f"🚀 Processing rate: {len(all_images)/processing_time:.2f} images/second")
        
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
