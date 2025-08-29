#!/usr/bin/env python3
"""
Accuracy Improvement Example Script

This script demonstrates how to use the enhanced accuracy deduplication system
to detect same-content images across different sizes and formats.

Features demonstrated:
- Multi-scale WHash for scale invariance
- SSIM for perceptual similarity
- Hybrid similarity calculation
- Comprehensive duplicate verification
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_dataset():
    """
    Create a test dataset with different image sizes and formats.
    In a real scenario, you would use your actual image paths.
    """
    # Example image paths - replace with your actual paths
    test_images = [
        # Local images
        "test_images/image1.jpg",
        "test_images/image2.png",
        "test_images/image3.tiff",
        
        # Azure blob URLs (example)
        "https://yourstorage.blob.core.windows.net/container/image1.jpg",
        "https://yourstorage.blob.core.windows.net/container/image2.jpg",
        "https://yourstorage.blob.core.windows.net/container/image3.jpg",
        
        # Add more paths as needed
    ]
    
    # Filter out non-existent local files for demo
    existing_images = []
    for path in test_images:
        if path.startswith('http') or os.path.exists(path):
            existing_images.append(path)
    
    logger.info(f"Test dataset created with {len(existing_images)} images")
    return existing_images

def run_accuracy_improved_deduplication():
    """
    Run the accuracy-improved deduplication pipeline.
    """
    try:
        # Import our enhanced modules
        from modules.enhanced_whash_deduplicator import create_enhanced_whash_deduplicator
        from modules.structural_similarity import create_structural_similarity_calculator
        from modules.hybrid_similarity_calculator import create_hybrid_similarity_calculator
        from modules.accuracy_optimized_deduplicator import create_accuracy_optimized_deduplicator
        
        logger.info("🚀 Starting accuracy-improved deduplication pipeline...")
        
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
                'structural': 0.6,          # Highest weight for SSIM
                'color': 0.25,              # Medium weight for color
                'whash': 0.15               # Lower weight for WHash
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
        logger.info("⚡ Creating accuracy-optimized deduplicator...")
        accuracy_deduplicator = create_accuracy_optimized_deduplicator(
            hybrid_calculator=hybrid_calculator,
            whash_deduplicator=whash_deduplicator,
            ssim_calculator=ssim_calculator,
            enable_verification=True,       # Enable duplicate verification
            verification_threshold=0.65,    # Threshold for verification
            max_group_size=1000,           # Maximum group size for verification
            enable_caching=True             # Enable caching
        )
        
        # Step 5: Create test dataset
        logger.info("📁 Creating test dataset...")
        test_images = create_test_dataset()
        
        if not test_images:
            logger.warning("No test images available. Please update the test dataset function.")
            return
        
        # Step 6: Run deduplication
        logger.info(f"🔄 Running accuracy-improved deduplication on {len(test_images)} images...")
        
        def progress_callback(message):
            logger.info(f"Progress: {message}")
        
        start_time = time.time()
        
        # Find duplicates with enhanced accuracy
        duplicate_groups = accuracy_deduplicator.find_duplicates(
            test_images, 
            progress_callback=progress_callback
        )
        
        processing_time = time.time() - start_time
        
        # Step 7: Analyze results
        logger.info("📊 Analyzing deduplication results...")
        
        total_groups = len(duplicate_groups)
        total_duplicates = sum(len(group) - 1 for group in duplicate_groups if len(group) > 1)
        single_images = len([group for group in duplicate_groups if len(group) == 1])
        duplicate_groups_count = len([group for group in duplicate_groups if len(group) > 1])
        
        logger.info("=" * 60)
        logger.info("🎉 DEDUPLICATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total images processed: {len(test_images)}")
        logger.info(f"Total groups found: {total_groups}")
        logger.info(f"Single images (no duplicates): {single_images}")
        logger.info(f"Duplicate groups: {duplicate_groups_count}")
        logger.info(f"Total duplicates found: {total_duplicates}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Average time per image: {processing_time/len(test_images):.3f} seconds")
        logger.info("=" * 60)
        
        # Display detailed results
        for i, group in enumerate(duplicate_groups):
            if len(group) > 1:
                logger.info(f"Duplicate Group {i+1}: {len(group)} images")
                for j, image_path in enumerate(group):
                    logger.info(f"  {j+1}. {image_path}")
                logger.info("")
        
        # Step 8: Performance statistics
        logger.info("📈 PERFORMANCE STATISTICS")
        logger.info("=" * 60)
        
        # WHash statistics
        whash_stats = whash_deduplicator.get_performance_stats()
        logger.info(f"WHash - Total processed: {whash_stats.get('total_images_processed', 0)}")
        logger.info(f"WHash - Average time per image: {whash_stats.get('average_time_per_image', 0):.3f}s")
        
        # SSIM statistics
        ssim_stats = ssim_calculator.get_stats()
        logger.info(f"SSIM - Total comparisons: {ssim_stats.get('total_comparisons', 0)}")
        logger.info(f"SSIM - GPU comparisons: {ssim_stats.get('gpu_comparisons', 0)}")
        logger.info(f"SSIM - CPU comparisons: {ssim_stats.get('cpu_comparisons', 0)}")
        logger.info(f"SSIM - Average time: {ssim_stats.get('average_time', 0):.3f}s")
        
        # Hybrid calculator statistics
        hybrid_stats = hybrid_calculator.get_stats()
        logger.info(f"Hybrid - Total comparisons: {hybrid_stats.get('total_comparisons', 0)}")
        logger.info(f"Hybrid - Cache hit rate: {hybrid_stats.get('hit_rate', 0):.1%}")
        logger.info(f"Hybrid - Average time: {hybrid_stats.get('average_time', 0):.3f}s")
        
        # Accuracy deduplicator statistics
        accuracy_stats = accuracy_deduplicator.get_stats()
        logger.info(f"Accuracy - Total processed: {accuracy_stats.get('total_images_processed', 0)}")
        logger.info(f"Accuracy - WHash groups: {accuracy_stats.get('whash_groups', 0)}")
        logger.info(f"Accuracy - Verified groups: {accuracy_stats.get('verified_groups', 0)}")
        logger.info(f"Accuracy - Total duplicates: {accuracy_stats.get('total_duplicates_found', 0)}")
        
        logger.info("=" * 60)
        
        # Step 9: Cleanup
        logger.info("🧹 Cleaning up resources...")
        accuracy_deduplicator.release()
        hybrid_calculator.release()
        ssim_calculator.release()
        whash_deduplicator.release()
        
        logger.info("✅ Accuracy-improved deduplication completed successfully!")
        
        return duplicate_groups
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please ensure all required modules are available.")
        return None
    except Exception as e:
        logger.error(f"Error during deduplication: {e}")
        return None

def compare_with_baseline():
    """
    Compare the accuracy-improved system with baseline methods.
    """
    logger.info("🔬 Comparing with baseline methods...")
    
    try:
        # Import baseline modules
        from modules.whash_deduplicator import create_whash_deduplicator
        from modules.color_optimized_deduplicator import create_color_optimized_deduplicator
        
        # Create test dataset
        test_images = create_test_dataset()
        
        if not test_images:
            logger.warning("No test images available for comparison.")
            return
        
        # Baseline 1: Original WHash
        logger.info("📊 Running baseline WHash deduplication...")
        baseline_whash = create_whash_deduplicator(
            hash_size=8,
            wavelet_level=2,
            threshold=0.85
        )
        
        start_time = time.time()
        baseline_whash_groups = baseline_whash.group_by_whash(test_images)
        baseline_whash_time = time.time() - start_time
        
        baseline_whash_duplicates = sum(len(group) - 1 for group in baseline_whash_groups if len(group) > 1)
        
        # Baseline 2: Color-optimized
        logger.info("📊 Running baseline color-optimized deduplication...")
        baseline_color = create_color_optimized_deduplicator(
            color_clusters=1000,
            parallel_processing=True,
            max_workers=4
        )
        
        start_time = time.time()
        baseline_color_groups, _ = baseline_color.deduplicate_with_color_prefiltering(
            test_images, "temp_output"
        )
        baseline_color_time = time.time() - start_time
        
        baseline_color_duplicates = sum(len(group) - 1 for group in baseline_color_groups if len(group) > 1)
        
        # Run accuracy-improved system
        logger.info("📊 Running accuracy-improved deduplication...")
        accuracy_groups = run_accuracy_improved_deduplication()
        
        if accuracy_groups:
            accuracy_duplicates = sum(len(group) - 1 for group in accuracy_groups if len(group) > 1)
            
            # Comparison results
            logger.info("=" * 60)
            logger.info("📊 COMPARISON RESULTS")
            logger.info("=" * 60)
            logger.info(f"Baseline WHash - Duplicates: {baseline_whash_duplicates}, Time: {baseline_whash_time:.2f}s")
            logger.info(f"Baseline Color - Duplicates: {baseline_color_duplicates}, Time: {baseline_color_time:.2f}s")
            logger.info(f"Accuracy-Improved - Duplicates: {accuracy_duplicates}")
            logger.info("=" * 60)
            
            # Calculate improvements
            if baseline_whash_duplicates > 0:
                whash_improvement = ((accuracy_duplicates - baseline_whash_duplicates) / baseline_whash_duplicates) * 100
                logger.info(f"Improvement over WHash: {whash_improvement:+.1f}%")
            
            if baseline_color_duplicates > 0:
                color_improvement = ((accuracy_duplicates - baseline_color_duplicates) / baseline_color_duplicates) * 100
                logger.info(f"Improvement over Color: {color_improvement:+.1f}%")
        
        # Cleanup
        baseline_whash.release()
        baseline_color.release()
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}")

def main():
    """
    Main function to run the accuracy improvement example.
    """
    logger.info("🎯 ACCURACY IMPROVEMENT DEMONSTRATION")
    logger.info("=" * 60)
    
    # Check if running in demo mode
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_with_baseline()
    else:
        # Run the main accuracy-improved deduplication
        run_accuracy_improved_deduplication()
    
    logger.info("🎉 Demo completed!")

if __name__ == "__main__":
    main()
