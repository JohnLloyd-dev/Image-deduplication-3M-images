#!/usr/bin/env python3
"""
Optimized accuracy test script for 5K images with balanced parameters.
This version focuses on better performance vs accuracy balance.
"""

import logging
import sys
import os
import time
import tempfile
import shutil
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.azure_utils import list_blobs_from_azure, SAS_URL
from modules.whash_deduplicator import WHashDeduplicator
from modules.color_optimized_deduplicator import ColorOptimizedDeduplicator
from modules.enhanced_whash_deduplicator import EnhancedWHashDeduplicator
from modules.structural_similarity import StructuralSimilarity
from modules.hybrid_similarity_calculator import HybridSimilarityCalculator
from modules.accuracy_optimized_deduplicator import AccuracyOptimizedDeduplicator
from modules.feature_cache import BoundedFeatureCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_enhanced_whash_deduplicator() -> EnhancedWHashDeduplicator:
    """Create optimized Enhanced WHash deduplicator."""
    return EnhancedWHashDeduplicator(
        hash_size=8,
        wavelet_level=2,  # Balanced level
        wavelet_name='haar',
        scale_factors=[0.9, 1.0, 1.1],  # Reduced scale factors for speed
        enable_lsh=True,
        lsh_bands=6,  # Increased for better grouping
        lsh_rows_per_band=3,  # Increased for better grouping
        similarity_threshold=0.65  # More conservative threshold
    )

def create_structural_similarity_calculator() -> StructuralSimilarity:
    """Create optimized Structural Similarity calculator."""
    return StructuralSimilarity(
        target_size=(128, 128),  # Balanced size
        use_gpu=False,  # Disable GPU for compatibility
        enable_preprocessing=True
    )

def create_hybrid_similarity_calculator(
    whash_dedup: EnhancedWHashDeduplicator,
    ssim_calc: StructuralSimilarity
) -> HybridSimilarityCalculator:
    """Create optimized Hybrid Similarity calculator."""
    return HybridSimilarityCalculator(
        whash_deduplicator=whash_dedup,
        ssim_calculator=ssim_calc,
        weights={'structural': 0.5, 'color': 0.3, 'whash': 0.2},  # Balanced weights
        thresholds={'global': 0.50, 'structural': 0.45, 'color': 0.40, 'whash': 0.45},  # More conservative
        enable_caching=True,
        cache_size=5000  # Smaller cache for 5K images
    )

def create_accuracy_optimized_deduplicator(
    hybrid_calc: HybridSimilarityCalculator,
    whash_dedup: EnhancedWHashDeduplicator,
    ssim_calc: StructuralSimilarity
) -> AccuracyOptimizedDeduplicator:
    """Create optimized Accuracy Optimized deduplicator."""
    return AccuracyOptimizedDeduplicator(
        hybrid_calculator=hybrid_calc,
        whash_deduplicator=whash_dedup,
        ssim_calculator=ssim_calc,
        enable_verification=True,
        verification_threshold=0.50,  # More conservative threshold
        max_group_size=200,  # Increased for better coverage
        enable_caching=True
    )

def create_baseline_deduplicators():
    """Create baseline deduplicators for comparison."""
    # Create feature cache
    feature_cache = BoundedFeatureCache(max_size=10000)
    
    # Baseline WHash
    whash_dedup = WHashDeduplicator(
        feature_cache=feature_cache,
        hash_size=8,
        similarity_threshold=0.8
    )
    
    # Baseline Color
    color_dedup = ColorOptimizedDeduplicator(
        feature_cache=feature_cache,
        global_threshold=0.85,
        local_threshold=0.75,
        color_threshold=0.85,
        wavelet_threshold=0.8,
        batch_size=32,
        num_workers=4
    )
    
    return whash_dedup, color_dedup

def run_baseline_whash_test(image_paths: List[str]) -> Dict[str, Any]:
    """Run baseline WHash test."""
    logger.info("🔸 Running baseline WHash test...")
    start_time = time.time()
    
    try:
        whash_dedup, _ = create_baseline_deduplicators()
        
        # Run deduplication
        duplicate_groups = whash_dedup.find_duplicates(image_paths)
        
        processing_time = time.time() - start_time
        total_duplicates = sum(len(group) - 1 for group in duplicate_groups if len(group) > 1)
        
        stats = whash_dedup.get_performance_stats()
        
        logger.info(f"✅ Baseline WHash: {total_duplicates} duplicates in {processing_time:.2f}s")
        
        return {
            'duplicates': total_duplicates,
            'time': processing_time,
            'groups': len(duplicate_groups),
            'stats': stats
        }
        
    except Exception as e:
        logger.error(f"❌ Baseline WHash test failed: {e}")
        return {'duplicates': 0, 'time': 0, 'groups': 0, 'stats': {}}

def run_baseline_color_test(image_paths: List[str]) -> Dict[str, Any]:
    """Run baseline Color test."""
    logger.info("🔸 Running baseline Color test...")
    start_time = time.time()
    
    try:
        _, color_dedup = create_baseline_deduplicators()
        
        # Run deduplication
        duplicate_groups = color_dedup.find_duplicates(image_paths)
        
        processing_time = time.time() - start_time
        total_duplicates = sum(len(group) - 1 for group in duplicate_groups if len(group) > 1)
        
        stats = color_dedup.get_performance_stats()
        
        logger.info(f"✅ Baseline Color: {total_duplicates} duplicates in {processing_time:.2f}s")
        
        return {
            'duplicates': total_duplicates,
            'time': processing_time,
            'groups': len(duplicate_groups),
            'stats': stats
        }
        
    except Exception as e:
        logger.error(f"❌ Baseline Color test failed: {e}")
        return {'duplicates': 0, 'time': 0, 'groups': 0, 'stats': {}}

def run_accuracy_optimized_test(image_paths: List[str]) -> Dict[str, Any]:
    """Run accuracy-optimized test with balanced parameters."""
    logger.info("🔸 Running accuracy-optimized test (balanced parameters)...")
    start_time = time.time()
    
    try:
        # Create optimized components
        whash_dedup = create_enhanced_whash_deduplicator()
        ssim_calc = create_structural_similarity_calculator()
        hybrid_calc = create_hybrid_similarity_calculator(whash_dedup, ssim_calc)
        accuracy_dedup = create_accuracy_optimized_deduplicator(hybrid_calc, whash_dedup, ssim_calc)
        
        # Progress callback
        def progress_callback(message: str):
            logger.info(f"📊 {message}")
        
        # Run deduplication
        duplicate_groups = accuracy_dedup.find_duplicates(image_paths, progress_callback)
        
        processing_time = time.time() - start_time
        total_duplicates = sum(len(group) - 1 for group in duplicate_groups if len(group) > 1)
        
        # Get detailed stats
        whash_stats = whash_dedup.get_stats()
        ssim_stats = ssim_calc.get_stats()
        hybrid_stats = hybrid_calc.get_stats()
        accuracy_stats = accuracy_dedup.get_stats()
        
        logger.info(f"✅ Accuracy-Optimized: {total_duplicates} duplicates in {processing_time:.2f}s")
        
        # Cleanup
        whash_dedup.release()
        ssim_calc.release()
        hybrid_calc.release()
        accuracy_dedup.release()
        
        return {
            'duplicates': total_duplicates,
            'time': processing_time,
            'groups': len(duplicate_groups),
            'whash_stats': whash_stats,
            'ssim_stats': ssim_stats,
            'hybrid_stats': hybrid_stats,
            'accuracy_stats': accuracy_stats
        }
        
    except Exception as e:
        logger.error(f"❌ Accuracy-optimized test failed: {e}")
        return {'duplicates': 0, 'time': 0, 'groups': 0, 'stats': {}}

def main():
    """Main test function."""
    logger.info("🚀 Starting optimized accuracy test for 5K images...")
    
    # Get image paths
    logger.info("📋 Fetching image paths from Azure...")
    try:
        all_blobs = list_blobs_from_azure(SAS_URL, use_cache=True)
        image_paths = [blob for blob in all_blobs if blob.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))]
        
        # Limit to 5000 images for testing
        if len(image_paths) > 5000:
            image_paths = image_paths[:5000]
        
        logger.info(f"📊 Testing with {len(image_paths)} images")
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch image paths: {e}")
        return
    
    # Run tests
    results = {}
    
    # Baseline tests
    results['whash'] = run_baseline_whash_test(image_paths)
    results['color'] = run_baseline_color_test(image_paths)
    
    # Accuracy-optimized test
    results['accuracy_optimized'] = run_accuracy_optimized_test(image_paths)
    
    # Print comparison
    logger.info("=" * 80)
    logger.info("📊 OPTIMIZED DEDUPLICATION COMPARISON RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"🔸 Baseline WHash:")
    logger.info(f"   - Duplicates found: {results['whash']['duplicates']}")
    logger.info(f"   - Processing time: {results['whash']['time']:.2f}s")
    if results['whash']['time'] > 0:
        logger.info(f"   - Speed: {len(image_paths)/results['whash']['time']:.1f} images/sec")
    
    logger.info(f"🔸 Baseline Color:")
    logger.info(f"   - Duplicates found: {results['color']['duplicates']}")
    logger.info(f"   - Processing time: {results['color']['time']:.2f}s")
    if results['color']['time'] > 0:
        logger.info(f"   - Speed: {len(image_paths)/results['color']['time']:.1f} images/sec")
    
    logger.info(f"🔸 Accuracy-Optimized:")
    logger.info(f"   - Duplicates found: {results['accuracy_optimized']['duplicates']}")
    logger.info(f"   - Processing time: {results['accuracy_optimized']['time']:.2f}s")
    if results['accuracy_optimized']['time'] > 0:
        logger.info(f"   - Speed: {len(image_paths)/results['accuracy_optimized']['time']:.1f} images/sec")
    
    # Improvement analysis
    if results['color']['duplicates'] > 0:
        improvement = ((results['accuracy_optimized']['duplicates'] - results['color']['duplicates']) / results['color']['duplicates']) * 100
        logger.info(f"📈 IMPROVEMENT ANALYSIS:")
        logger.info(f"   - vs Baseline Color: {improvement:+.1f}% duplicates found")
    
    # Performance analysis
    if 'whash_stats' in results['accuracy_optimized']:
        whash_stats = results['accuracy_optimized']['whash_stats']
        hybrid_stats = results['accuracy_optimized']['hybrid_stats']
        accuracy_stats = results['accuracy_optimized']['accuracy_stats']
        
        logger.info(f"📈 Detailed Performance Stats:")
        logger.info(f"   - WHash: {whash_stats.get('total_images_processed', 0)} images processed")
        logger.info(f"   - SSIM: {hybrid_stats.get('ssim_comparisons', 0)} comparisons")
        logger.info(f"   - Hybrid: {hybrid_stats.get('total_comparisons', 0)} comparisons, {hybrid_stats.get('hit_rate', 0)*100:.1f}% cache hit rate")
        logger.info(f"   - Accuracy: {accuracy_stats.get('whash_groups', 0)} WHash groups, {accuracy_stats.get('verified_groups', 0)} verified")
    
    logger.info("=" * 80)
    logger.info("🎉 Optimized accuracy test completed successfully!")

if __name__ == "__main__":
    main()

