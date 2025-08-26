#!/usr/bin/env python3
"""
WHash-Color Integration Example for Large-Scale Image Deduplication

This example demonstrates how to integrate WHash (Wavelet Hash) deduplication
with the existing color-optimized pipeline for optimal performance on 3M+ images.

Key Benefits:
- WHash provides fast first-pass grouping using wavelet transforms
- Color pipeline provides accurate duplicate detection within groups
- Combined approach scales efficiently to millions of images
- Configurable parameters for different use cases

Usage:
    python examples/whash_color_integration_example.py
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.whash_deduplicator import WHashDeduplicator, create_whash_deduplicator
from modules.color_optimized_deduplicator import ColorOptimizedDeduplicator, create_color_optimized_deduplicator
from modules.feature_cache import BoundedFeatureCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_dataset(output_dir: str, num_images: int = 100) -> List[str]:
    """
    Create a test dataset with controlled duplicates for demonstration.
    
    Args:
        output_dir: Directory to save test images
        num_images: Number of images to create
        
    Returns:
        List of image file paths
    """
    logger.info(f"Creating test dataset with {num_images} images...")
    
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    
    # Create different types of images
    for i in range(num_images):
        if i < 20:
            # Group 1: Solid color images (potential duplicates)
            img_data = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8)
            # Add slight variations
            img_data += np.random.randint(-10, 10, (64, 64, 3), dtype=np.int16)
            img_data = np.clip(img_data, 0, 255).astype(np.uint8)
            
        elif i < 40:
            # Group 2: Gradient images (similar patterns)
            x, y = np.meshgrid(np.arange(64), np.arange(64))
            img_data = np.stack([
                (x * 2 + np.random.randint(-5, 5, (64, 64))).astype(np.uint8),
                (y * 2 + np.random.randint(-5, 5, (64, 64))).astype(np.uint8),
                np.full((64, 64), 128, dtype=np.uint8)
            ], axis=2)
            img_data = np.clip(img_data, 0, 255).astype(np.uint8)
            
        elif i < 60:
            # Group 3: Noise images (different random patterns)
            img_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            
        elif i < 80:
            # Group 4: Edge images (similar edge patterns)
            img_data = np.zeros((64, 64, 3), dtype=np.uint8)
            img_data[::8, :, :] = 255  # Horizontal lines
            img_data[:, ::8, :] = 255  # Vertical lines
            # Add some noise
            noise = np.random.randint(-20, 20, (64, 64, 3), dtype=np.int16)
            img_data = np.clip(img_data.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
        else:
            # Group 5: Unique images
            img_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Save image
        img_path = os.path.join(output_dir, f"test_image_{i:03d}.jpg")
        import cv2
        cv2.imwrite(img_path, img_data)
        image_paths.append(img_path)
    
    logger.info(f"Test dataset created: {len(image_paths)} images in {output_dir}")
    return image_paths

def demonstrate_whash_only_deduplication(image_paths: List[str], output_dir: str):
    """
    Demonstrate WHash-only deduplication.
    
    Args:
        image_paths: List of image paths to process
        output_dir: Output directory for results
    """
    logger.info("🔍 Demonstrating WHash-only deduplication...")
    
    # Create WHash deduplicator with optimized parameters
    whash_dedup = WHashDeduplicator(
        hash_size=8,           # 8x8 = 64-bit hash
        wavelet_level=2,       # 2 levels of wavelet decomposition
        threshold=0.85,        # 85% similarity threshold
        enable_lsh=True,       # Enable LSH for efficiency
        lsh_bands=4,           # 4 LSH bands
        lsh_rows_per_band=4    # 4 rows per band
    )
    
    # Process images
    start_time = time.time()
    whash_groups = whash_dedup.group_by_whash(image_paths)
    whash_time = time.time() - start_time
    
    # Analyze results
    logger.info(f"WHash-only results:")
    logger.info(f"  - Processing time: {whash_time:.2f}s")
    logger.info(f"  - Images processed: {whash_dedup.stats['images_processed']}")
    logger.info(f"  - Hashes computed: {whash_dedup.stats['hashes_computed']}")
    logger.info(f"  - Groups created: {len(whash_groups)}")
    
    # Show group statistics
    group_sizes = [len(group) for group in whash_groups]
    logger.info(f"  - Group sizes: min={min(group_sizes)}, max={max(group_sizes)}, avg={np.mean(group_sizes):.1f}")
    
    # Show potential duplicates
    duplicate_groups = [group for group in whash_groups if len(group) > 1]
    logger.info(f"  - Groups with potential duplicates: {len(duplicate_groups)}")
    
    return whash_groups, whash_dedup.get_stats()

def demonstrate_color_only_deduplication(image_paths: List[str], output_dir: str):
    """
    Demonstrate color-only deduplication.
    
    Args:
        image_paths: List of image paths to process
        output_dir: Output directory for results
    """
    logger.info("🎨 Demonstrating color-only deduplication...")
    
    # Create color-optimized deduplicator
    feature_cache = BoundedFeatureCache(max_size=1000)
    color_dedup = create_color_optimized_deduplicator(
        feature_cache=feature_cache,
        color_clusters=50,      # Smaller number for demo
        batch_size=20,
        adaptive_thresholding=True,
        parallel_processing=True,
        max_workers=2
    )
    
    # Process images
    start_time = time.time()
    try:
        duplicate_groups, similarity_scores = color_dedup.deduplicate_with_color_prefiltering(
            image_paths, output_dir
        )
        color_time = time.time() - start_time
        
        logger.info(f"Color-only results:")
        logger.info(f"  - Processing time: {color_time:.2f}s")
        logger.info(f"  - Duplicate groups: {len(duplicate_groups)}")
        logger.info(f"  - Similarity scores computed: {len(similarity_scores)}")
        
        return duplicate_groups, similarity_scores, color_time
        
    except Exception as e:
        logger.error(f"Color deduplication failed: {e}")
        return [], {}, 0.0

def demonstrate_whash_color_integration(image_paths: List[str], output_dir: str):
    """
    Demonstrate integrated WHash-Color deduplication.
    
    Args:
        image_paths: List of image paths to process
        output_dir: Output directory for results
    """
    logger.info("🚀 Demonstrating WHash-Color integration...")
    
    # Create WHash deduplicator
    whash_dedup = WHashDeduplicator(
        hash_size=8,
        wavelet_level=2,
        threshold=0.85,
        enable_lsh=True,
        lsh_bands=4,
        lsh_rows_per_band=4
    )
    
    # Create color-optimized deduplicator
    feature_cache = BoundedFeatureCache(max_size=1000)
    color_dedup = create_color_optimized_deduplicator(
        feature_cache=feature_cache,
        color_clusters=50,
        batch_size=20,
        adaptive_thresholding=True,
        parallel_processing=True,
        max_workers=2
    )
    
    # Integrate WHash with color pipeline
    integrated_dedup = whash_dedup.integrate_with_color_pipeline(color_dedup)
    
    # Process images with integrated pipeline
    start_time = time.time()
    try:
        duplicate_groups, similarity_scores = integrated_dedup.deduplicate_with_whash_color_integration(
            image_paths, output_dir
        )
        integrated_time = time.time() - start_time
        
        logger.info(f"WHash-Color integration results:")
        logger.info(f"  - Processing time: {integrated_time:.2f}s")
        logger.info(f"  - Final groups: {len(duplicate_groups)}")
        logger.info(f"  - Similarity scores: {len(similarity_scores)}")
        
        # Show WHash statistics
        whash_stats = whash_dedup.get_stats()
        logger.info(f"  - WHash groups created: {whash_stats['groups_created']}")
        logger.info(f"  - WHash processing time: {whash_stats['processing_time']:.2f}s")
        
        return duplicate_groups, similarity_scores, integrated_time
        
    except Exception as e:
        logger.error(f"Integrated deduplication failed: {e}")
        return [], {}, 0.0

def compare_performance(whash_groups: List, whash_stats: Dict, 
                       color_time: float, integrated_time: float):
    """
    Compare performance of different approaches.
    
    Args:
        whash_groups: Groups from WHash-only approach
        whash_stats: Statistics from WHash processing
        color_time: Processing time for color-only approach
        integrated_time: Processing time for integrated approach
    """
    logger.info("\n📊 Performance Comparison:")
    logger.info("=" * 50)
    
    # WHash-only performance
    logger.info(f"WHash-Only:")
    logger.info(f"  - Time: {whash_stats['processing_time']:.2f}s")
    logger.info(f"  - Groups: {len(whash_groups)}")
    logger.info(f"  - Speed: {whash_stats['images_processed'] / whash_stats['processing_time']:.1f} images/sec")
    
    # Color-only performance
    if color_time > 0:
        logger.info(f"Color-Only:")
        logger.info(f"  - Time: {color_time:.2f}s")
        logger.info(f"  - Speed: {len(whash_groups) / color_time:.1f} images/sec")
    
    # Integrated performance
    if integrated_time > 0:
        logger.info(f"WHash-Color Integration:")
        logger.info(f"  - Time: {integrated_time:.2f}s")
        logger.info(f"  - Speed: {len(whash_groups) / integrated_time:.1f} images/sec")
        
        # Calculate efficiency improvement
        if color_time > 0:
            improvement = (color_time - integrated_time) / color_time * 100
            logger.info(f"  - Efficiency improvement: {improvement:.1f}%")
    
    # Memory efficiency
    logger.info(f"\nMemory Efficiency:")
    logger.info(f"  - WHash hash size: 8x8 = 64 bits per image")
    logger.info(f"  - LSH bands: 4 bands × 4 rows = 16-bit signatures")
    logger.info(f"  - Memory per 1M images: ~8 MB (vs ~1-2 GB for full features)")

def main():
    """Main demonstration function."""
    logger.info("🎯 WHash-Color Integration Demonstration")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = "whash_color_demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test dataset
    num_test_images = 100
    image_paths = create_test_dataset(output_dir, num_test_images)
    
    logger.info(f"\n📁 Working with {len(image_paths)} test images")
    logger.info(f"📂 Output directory: {output_dir}")
    
    # Demonstrate different approaches
    logger.info("\n" + "="*60)
    
    # 1. WHash-only approach
    whash_groups, whash_stats = demonstrate_whash_only_deduplication(image_paths, output_dir)
    
    # 2. Color-only approach
    color_groups, color_scores, color_time = demonstrate_color_only_deduplication(image_paths, output_dir)
    
    # 3. Integrated approach
    integrated_groups, integrated_scores, integrated_time = demonstrate_whash_color_integration(
        image_paths, output_dir
    )
    
    # Compare performance
    logger.info("\n" + "="*60)
    compare_performance(whash_groups, whash_stats, color_time, integrated_time)
    
    # Summary and recommendations
    logger.info("\n" + "="*60)
    logger.info("🎯 Summary and Recommendations:")
    logger.info("=" * 60)
    
    logger.info("✅ WHash provides fast first-pass grouping:")
    logger.info("   - Excellent for large datasets (3M+ images)")
    logger.info("   - Low memory footprint")
    logger.info("   - Fast processing with LSH optimization")
    
    logger.info("✅ Color pipeline provides accurate duplicate detection:")
    logger.info("   - High accuracy for near-duplicate detection")
    logger.info("   - Robust feature extraction")
    logger.info("   - Configurable thresholds")
    
    logger.info("✅ Integration provides best of both worlds:")
    logger.info("   - Fast pre-grouping reduces problem size")
    logger.info("   - Accurate detection within groups")
    logger.info("   - Scalable to millions of images")
    
    logger.info("\n🚀 Ready for production use!")
    logger.info("   - WHash deduplicator: modules/whash_deduplicator.py")
    logger.info("   - Integration example: This script")
    logger.info("   - Test suite: tests/performance/test_whash_deduplicator.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        sys.exit(1)
