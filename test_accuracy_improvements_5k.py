#!/usr/bin/env python3
"""
Test Accuracy Improvements on 5000 Image Dataset

This script tests the new accuracy-improved deduplication system on a subset
of 5000 images to validate performance and accuracy improvements.
"""

import os
import sys
import logging
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_test_dataset(max_images: int = 5000) -> List[str]:
    """
    Get a test dataset of up to 5000 images from Azure.
    
    Args:
        max_images: Maximum number of images to test
        
    Returns:
        List of image paths
    """
    try:
        from modules.azure_utils import AzureBlobManager
        
        logger.info(f"Fetching up to {max_images} images from Azure...")
        azure_manager = AzureBlobManager()
        
        # Get images from the target directory
        all_image_paths = azure_manager.list_blobs('Image_Dedup_Project/TestEquity/CompleteImageDataset/')
        
        if not all_image_paths:
            logger.error("No images found in Azure directory!")
            return []
        
        # Take a subset for testing
        test_images = all_image_paths[:max_images]
        logger.info(f"Selected {len(test_images)} images for testing")
        
        return test_images
        
    except Exception as e:
        logger.error(f"Failed to get test dataset from Azure: {e}")
        
        # Fallback to local test images if available
        logger.info("Falling back to local test images...")
        local_test_dir = "test_images"
        if os.path.exists(local_test_dir):
            local_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                local_images.extend(Path(local_test_dir).glob(f"**/{ext}"))
            
            test_images = [str(p) for p in local_images[:max_images]]
            logger.info(f"Found {len(test_images)} local test images")
            return test_images
        
        logger.warning("No test images available!")
        return []

def test_baseline_methods(image_paths: List[str], output_dir: str) -> Dict[str, Any]:
    """
    Test baseline deduplication methods for comparison.
    
    Args:
        image_paths: List of image paths to test
        output_dir: Output directory for results
        
    Returns:
        Dictionary with baseline results
    """
    results = {}
    
    try:
        # Test 1: Original WHash
        logger.info("🔄 Testing baseline WHash deduplication...")
        from modules.whash_deduplicator import create_whash_deduplicator
        
        whash_dedup = create_whash_deduplicator(
            hash_size=8,
            wavelet_level=2,
            threshold=0.85,
            enable_lsh=True,
            lsh_bands=4,
            lsh_rows_per_band=4
        )
        
        start_time = time.time()
        whash_groups = whash_dedup.group_by_whash(image_paths)
        whash_time = time.time() - start_time
        
        whash_duplicates = sum(len(group) - 1 for group in whash_groups if len(group) > 1)
        whash_stats = whash_dedup.get_performance_stats()
        
        results['baseline_whash'] = {
            'total_groups': len(whash_groups),
            'duplicate_groups': len([g for g in whash_groups if len(g) > 1]),
            'total_duplicates': whash_duplicates,
            'processing_time': whash_time,
            'images_per_second': len(image_paths) / whash_time if whash_time > 0 else 0,
            'stats': whash_stats
        }
        
        logger.info(f"✅ Baseline WHash: {whash_duplicates} duplicates in {whash_time:.2f}s")
        whash_dedup.release()
        
    except Exception as e:
        logger.error(f"Baseline WHash test failed: {e}")
        results['baseline_whash'] = {'error': str(e)}
    
    try:
        # Test 2: Color-optimized
        logger.info("🔄 Testing baseline color-optimized deduplication...")
        from modules.color_optimized_deduplicator import create_color_optimized_deduplicator
        from modules.feature_cache import BoundedFeatureCache
        
        color_dedup = create_color_optimized_deduplicator(
            feature_cache=BoundedFeatureCache(max_size=1000),
            global_threshold=0.85,
            local_threshold=0.75,
            color_threshold=0.85,
            wavelet_threshold=0.8,
            batch_size=32,
            num_workers=4
        )
        
        start_time = time.time()
        color_groups, color_scores = color_dedup.deduplicate_with_color_prefiltering(
            image_paths, output_dir
        )
        color_time = time.time() - start_time
        
        color_duplicates = sum(len(group) - 1 for group in color_groups if len(group) > 1)
        
        results['baseline_color'] = {
            'total_groups': len(color_groups),
            'duplicate_groups': len([g for g in color_groups if len(g) > 1]),
            'total_duplicates': color_duplicates,
            'processing_time': color_time,
            'images_per_second': len(image_paths) / color_time if color_time > 0 else 0,
            'similarity_scores_count': len(color_scores)
        }
        
        logger.info(f"✅ Baseline Color: {color_duplicates} duplicates in {color_time:.2f}s")
        color_dedup.release()
        
    except Exception as e:
        logger.error(f"Baseline color test failed: {e}")
        results['baseline_color'] = {'error': str(e)}
    
    return results

def test_accuracy_improved_method(image_paths: List[str], output_dir: str) -> Dict[str, Any]:
    """
    Test the new accuracy-improved deduplication method.
    
    Args:
        image_paths: List of image paths to test
        output_dir: Output directory for results
        
    Returns:
        Dictionary with accuracy-improved results
    """
    try:
        logger.info("🚀 Testing accuracy-improved deduplication...")
        
        # Import enhanced modules
        from modules.enhanced_whash_deduplicator import create_enhanced_whash_deduplicator
        from modules.structural_similarity import create_structural_similarity_calculator
        from modules.hybrid_similarity_calculator import create_hybrid_similarity_calculator
        from modules.accuracy_optimized_deduplicator import create_accuracy_optimized_deduplicator
        
        # Create enhanced WHash deduplicator
        whash_dedup = create_enhanced_whash_deduplicator(
            hash_size=8,
            wavelet_level=2,  # Reduced from 3
            wavelet_name='haar',
            scale_factors=[0.8, 1.0, 1.2],  # Reduced scale factors for speed
            enable_lsh=True,
            lsh_bands=4,  # Fixed: 4 bands for reasonable grouping
            lsh_rows_per_band=2,  # Fixed: 2 rows per band for reasonable grouping
            similarity_threshold=0.50  # Much more aggressive - reduced from 0.70
        )
        
        # Test WHash deduplicator directly first
        logger.info("🔍 Testing WHash deduplicator directly...")
        try:
            # Test with a small subset first
            test_subset = image_paths[:100]  # Test with first 100 images
            logger.info(f"Testing WHash with {len(test_subset)} images...")
            
            whash_groups = whash_dedup.group_by_whash(test_subset)
            logger.info(f"WHash created {len(whash_groups)} groups")
            
            if whash_groups:
                logger.info(f"Sample group sizes: {[len(g) for g in whash_groups[:5]]}")
            else:
                logger.warning("WHash created 0 groups - this indicates a problem!")
                
        except Exception as e:
            logger.error(f"WHash test failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Create SSIM calculator
        ssim_calc = create_structural_similarity_calculator(
            target_size=(128, 128),  # Reduced from 256x256 for speed
            use_gpu=False,  # Disable GPU for compatibility
            enable_preprocessing=True
        )
        
        # Create hybrid similarity calculator
        hybrid_calc = create_hybrid_similarity_calculator(
            whash_deduplicator=whash_dedup,
            ssim_calculator=ssim_calc,
            weights={'structural': 0.5, 'color': 0.3, 'whash': 0.2},  # Adjusted weights
            thresholds={'global': 0.40, 'structural': 0.35, 'color': 0.30, 'whash': 0.35},  # Much more aggressive thresholds
            enable_caching=True,
            cache_size=5000  # Smaller cache for 5K images
        )
        
        # Create accuracy-optimized deduplicator
        accuracy_dedup = create_accuracy_optimized_deduplicator(
            hybrid_calculator=hybrid_calc,
            whash_deduplicator=whash_dedup,
            ssim_calculator=ssim_calc,
            enable_verification=True,
            verification_threshold=0.40,  # Much more aggressive - reduced from 0.55
            max_group_size=100,  # Reduced from 500
            enable_caching=True
        )
        
        # Run accuracy-improved deduplication
        start_time = time.time()
        
        def progress_callback(msg):
            logger.info(f"📊 {msg}")
        
        accuracy_groups = accuracy_dedup.find_duplicates(
            image_paths, progress_callback=progress_callback
        )
        
        accuracy_time = time.time() - start_time
        accuracy_duplicates = sum(len(group) - 1 for group in accuracy_groups if len(group) > 1)
        
        # Get detailed statistics
        whash_stats = whash_dedup.get_performance_stats()
        ssim_stats = ssim_calc.get_stats()
        hybrid_stats = hybrid_calc.get_stats()
        accuracy_stats = accuracy_dedup.get_stats()
        
        results = {
            'total_groups': len(accuracy_groups),
            'duplicate_groups': len([g for g in accuracy_groups if len(g) > 1]),
            'total_duplicates': accuracy_duplicates,
            'processing_time': accuracy_time,
            'images_per_second': len(image_paths) / accuracy_time if accuracy_time > 0 else 0,
            'whash_stats': whash_stats,
            'ssim_stats': ssim_stats,
            'hybrid_stats': hybrid_stats,
            'accuracy_stats': accuracy_stats
        }
        
        logger.info(f"✅ Accuracy-Improved: {accuracy_duplicates} duplicates in {accuracy_time:.2f}s")
        
        # Cleanup
        whash_dedup.release()
        ssim_calc.release()
        hybrid_calc.release()
        accuracy_dedup.release()
        
        return results
        
    except Exception as e:
        logger.error(f"Accuracy-improved test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def analyze_results(baseline_results: Dict[str, Any], accuracy_results: Dict[str, Any]) -> None:
    """
    Analyze and display comparison results.
    
    Args:
        baseline_results: Results from baseline methods
        accuracy_results: Results from accuracy-improved method
    """
    logger.info("=" * 80)
    logger.info("📊 DEDUPLICATION COMPARISON RESULTS")
    logger.info("=" * 80)
    
    # Display baseline results
    if 'baseline_whash' in baseline_results and 'error' not in baseline_results['baseline_whash']:
        whash_data = baseline_results['baseline_whash']
        logger.info(f"🔸 Baseline WHash:")
        logger.info(f"   - Duplicates found: {whash_data['total_duplicates']}")
        logger.info(f"   - Processing time: {whash_data['processing_time']:.2f}s")
        logger.info(f"   - Speed: {whash_data['images_per_second']:.1f} images/sec")
    
    if 'baseline_color' in baseline_results and 'error' not in baseline_results['baseline_color']:
        color_data = baseline_results['baseline_color']
        logger.info(f"🔸 Baseline Color:")
        logger.info(f"   - Duplicates found: {color_data['total_duplicates']}")
        logger.info(f"   - Processing time: {color_data['processing_time']:.2f}s")
        logger.info(f"   - Speed: {color_data['images_per_second']:.1f} images/sec")
    
    # Display accuracy-improved results
    if 'error' not in accuracy_results:
        logger.info(f"🔸 Accuracy-Improved:")
        logger.info(f"   - Duplicates found: {accuracy_results['total_duplicates']}")
        logger.info(f"   - Processing time: {accuracy_results['processing_time']:.2f}s")
        logger.info(f"   - Speed: {accuracy_results['images_per_second']:.1f} images/sec")
        
        # Display detailed stats
        logger.info(f"📈 Detailed Performance Stats:")
        
        whash_stats = accuracy_results.get('whash_stats', {})
        logger.info(f"   - WHash: {whash_stats.get('total_images_processed', 0)} images processed")
        
        ssim_stats = accuracy_results.get('ssim_stats', {})
        logger.info(f"   - SSIM: {ssim_stats.get('total_comparisons', 0)} comparisons")
        
        hybrid_stats = accuracy_results.get('hybrid_stats', {})
        cache_hit_rate = hybrid_stats.get('hit_rate', 0)
        logger.info(f"   - Hybrid: {hybrid_stats.get('total_comparisons', 0)} comparisons, "
                   f"{cache_hit_rate:.1%} cache hit rate")
        
        accuracy_stats = accuracy_results.get('accuracy_stats', {})
        logger.info(f"   - Accuracy: {accuracy_stats.get('whash_groups', 0)} WHash groups, "
                   f"{accuracy_stats.get('verified_groups', 0)} verified")
    
    # Calculate improvements
    logger.info("📈 IMPROVEMENT ANALYSIS:")
    
    if ('baseline_whash' in baseline_results and 
        'error' not in baseline_results['baseline_whash'] and 
        'error' not in accuracy_results):
        
        whash_dups = baseline_results['baseline_whash']['total_duplicates']
        accuracy_dups = accuracy_results['total_duplicates']
        
        if whash_dups > 0:
            improvement = ((accuracy_dups - whash_dups) / whash_dups) * 100
            logger.info(f"   - vs Baseline WHash: {improvement:+.1f}% duplicates found")
    
    if ('baseline_color' in baseline_results and 
        'error' not in baseline_results['baseline_color'] and 
        'error' not in accuracy_results):
        
        color_dups = baseline_results['baseline_color']['total_duplicates']
        accuracy_dups = accuracy_results['total_duplicates']
        
        if color_dups > 0:
            improvement = ((accuracy_dups - color_dups) / color_dups) * 100
            logger.info(f"   - vs Baseline Color: {improvement:+.1f}% duplicates found")
    
    logger.info("=" * 80)

def main():
    """
    Main function to run the 5000 image accuracy test.
    """
    logger.info("🎯 ACCURACY IMPROVEMENTS TEST - 5000 IMAGES")
    logger.info("=" * 60)
    
    # Create temporary output directory
    temp_dir = tempfile.mkdtemp(prefix="accuracy_test_5k_")
    logger.info(f"📁 Using temporary directory: {temp_dir}")
    
    try:
        # Get test dataset
        logger.info("📋 Getting test dataset...")
        image_paths = get_test_dataset(max_images=5000)
        
        if not image_paths:
            logger.error("❌ No test images available!")
            return
        
        logger.info(f"✅ Test dataset ready: {len(image_paths)} images")
        
        # Test baseline methods
        logger.info("🔄 Testing baseline methods...")
        baseline_results = test_baseline_methods(image_paths, temp_dir)
        
        # Test accuracy-improved method
        logger.info("🔄 Testing accuracy-improved method...")
        accuracy_results = test_accuracy_improved_method(image_paths, temp_dir)
        
        # Analyze and display results
        analyze_results(baseline_results, accuracy_results)
        
        logger.info("🎉 Accuracy test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup temporary directory
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"🧹 Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary directory: {e}")

if __name__ == "__main__":
    main()
