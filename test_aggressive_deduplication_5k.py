#!/usr/bin/env python3
"""
Aggressive deduplication test script for 5K images.
This version uses more aggressive thresholds to find more duplicates.
"""

import logging
import sys
import os
import time
import tempfile
import shutil
import csv
from typing import List, Dict, Any
from datetime import datetime

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

def create_aggressive_whash_deduplicator() -> EnhancedWHashDeduplicator:
    """Create aggressive Enhanced WHash deduplicator to find more duplicates."""
    return EnhancedWHashDeduplicator(
        hash_size=8,
        wavelet_level=2,
        wavelet_name='haar',
        scale_factors=[0.8, 0.9, 1.0, 1.1, 1.2],  # More scale factors
        enable_lsh=True,
        lsh_bands=8,  # Moderate bands for better coverage
        lsh_rows_per_band=3,  # More rows for better recall
        similarity_threshold=0.60  # More aggressive threshold
    )

def create_aggressive_structural_similarity_calculator() -> StructuralSimilarity:
    """Create aggressive Structural Similarity calculator."""
    return StructuralSimilarity(
        target_size=(128, 128),
        use_gpu=False,
        enable_preprocessing=True
    )

def create_aggressive_hybrid_similarity_calculator(
    whash_dedup: EnhancedWHashDeduplicator,
    ssim_calc: StructuralSimilarity
) -> HybridSimilarityCalculator:
    """Create aggressive Hybrid Similarity calculator."""
    return HybridSimilarityCalculator(
        whash_deduplicator=whash_dedup,
        ssim_calculator=ssim_calc,
        weights={'structural': 0.4, 'color': 0.4, 'whash': 0.2},  # More balanced weights
        thresholds={'global': 0.45, 'structural': 0.40, 'color': 0.35, 'whash': 0.40},  # More aggressive
        enable_caching=True,
        cache_size=5000
    )

def create_aggressive_accuracy_optimized_deduplicator(
    hybrid_calc: HybridSimilarityCalculator,
    whash_dedup: EnhancedWHashDeduplicator,
    ssim_calc: StructuralSimilarity
) -> AccuracyOptimizedDeduplicator:
    """Create aggressive Accuracy Optimized deduplicator."""
    return AccuracyOptimizedDeduplicator(
        hybrid_calculator=hybrid_calc,
        whash_deduplicator=whash_dedup,
        ssim_calculator=ssim_calc,
        enable_verification=True,
        verification_threshold=0.45,  # More aggressive threshold
        max_group_size=300,  # Larger groups for more coverage
        enable_caching=True
    )

def create_baseline_deduplicators():
    """Create baseline deduplicators for comparison."""
    feature_cache = BoundedFeatureCache(max_size=10000)
    
    # More aggressive baseline WHash
    whash_dedup = WHashDeduplicator(
        feature_cache=feature_cache,
        hash_size=8,
        similarity_threshold=0.7  # More aggressive
    )
    
    # More aggressive baseline Color
    color_dedup = ColorOptimizedDeduplicator(
        feature_cache=feature_cache,
        global_threshold=0.75,  # More aggressive
        local_threshold=0.65,   # More aggressive
        color_threshold=0.75,   # More aggressive
        wavelet_threshold=0.7,  # More aggressive
        batch_size=32,
        num_workers=4
    )
    
    return whash_dedup, color_dedup

def run_baseline_whash_test(image_paths: List[str]) -> Dict[str, Any]:
    """Run baseline WHash test."""
    logger.info("🔸 Running aggressive baseline WHash test...")
    start_time = time.time()
    
    try:
        whash_dedup, _ = create_baseline_deduplicators()
        duplicate_groups = whash_dedup.find_duplicates(image_paths)
        processing_time = time.time() - start_time
        total_duplicates = sum(len(group) - 1 for group in duplicate_groups if len(group) > 1)
        stats = whash_dedup.get_performance_stats()
        
        logger.info(f"✅ Aggressive Baseline WHash: {total_duplicates} duplicates in {processing_time:.2f}s")
        
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
    logger.info("🔸 Running aggressive baseline Color test...")
    start_time = time.time()
    
    try:
        _, color_dedup = create_baseline_deduplicators()
        duplicate_groups = color_dedup.find_duplicates(image_paths)
        processing_time = time.time() - start_time
        total_duplicates = sum(len(group) - 1 for group in duplicate_groups if len(group) > 1)
        stats = color_dedup.get_performance_stats()
        
        logger.info(f"✅ Aggressive Baseline Color: {total_duplicates} duplicates in {processing_time:.2f}s")
        
        return {
            'duplicates': total_duplicates,
            'time': processing_time,
            'groups': len(duplicate_groups),
            'stats': stats
        }
        
    except Exception as e:
        logger.error(f"❌ Baseline Color test failed: {e}")
        return {'duplicates': 0, 'time': 0, 'groups': 0, 'stats': {}}

def generate_duplicate_csv_report(duplicate_groups: List[List[str]], output_file: str = "duplicate_report.csv") -> str:
    """Generate comprehensive CSV report of duplicate findings."""
    logger.info(f"📝 Generating CSV report: {output_file}")
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'group_id',
                'image_path',
                'image_index',
                'group_size',
                'is_representative',
                'whash_similarity',
                'ssim_similarity',
                'color_similarity',
                'hybrid_similarity',
                'verification_status',
                'file_size_bytes',
                'file_extension',
                'processing_timestamp'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            total_duplicates = 0
            group_id = 1
            
            for group in duplicate_groups:
                if len(group) > 1:  # Only process groups with duplicates
                    group_size = len(group)
                    total_duplicates += group_size - 1  # Subtract 1 for the representative
                    
                    # Write each image in the group
                    for i, image_path in enumerate(group):
                        is_representative = (i == 0)  # First image is representative
                        
                        # Get file info
                        try:
                            file_size = os.path.getsize(image_path) if os.path.exists(image_path) else 0
                            file_ext = os.path.splitext(image_path)[1].lower()
                        except:
                            file_size = 0
                            file_ext = 'unknown'
                        
                        # For the representative image, use perfect similarity scores
                        if is_representative:
                            whash_sim = 1.0
                            ssim_sim = 1.0
                            color_sim = 1.0
                            hybrid_sim = 1.0
                        else:
                            # For duplicates, we don't have the actual similarity scores from the verification
                            # This is a limitation - we should capture them during verification
                            # For now, use placeholder values to indicate they were verified as duplicates
                            whash_sim = "verified"
                            ssim_sim = "verified"
                            color_sim = "verified"
                            hybrid_sim = "verified"
                        
                        writer.writerow({
                            'group_id': group_id,
                            'image_path': image_path,
                            'image_index': i,
                            'group_size': group_size,
                            'is_representative': is_representative,
                            'whash_similarity': whash_sim,
                            'ssim_similarity': ssim_sim,
                            'color_similarity': color_sim,
                            'hybrid_similarity': hybrid_sim,
                            'verification_status': 'verified',
                            'file_size_bytes': file_size,
                            'file_extension': file_ext,
                            'processing_timestamp': datetime.now().isoformat()
                        })
                    
                    group_id += 1
            
            logger.info(f"✅ CSV report generated successfully!")
            logger.info(f"📊 Total duplicate groups: {group_id - 1}")
            logger.info(f"📊 Total duplicate images: {total_duplicates}")
            logger.info(f"📊 Report saved to: {output_file}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Failed to generate CSV report: {e}")
        return None

def generate_summary_csv_report(total_duplicates: int, total_groups: int, processing_time: float, 
                               total_images: int, output_file: str = "duplicate_summary.csv") -> str:
    """Generate a summary CSV report of the deduplication results."""
    logger.info(f"📋 Generating summary report: {output_file}")
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'metric',
                'value',
                'description',
                'timestamp'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Summary statistics
            summary_data = [
                {
                    'metric': 'total_images_processed',
                    'value': total_images,
                    'description': 'Total number of images in dataset',
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'metric': 'duplicate_groups_found',
                    'value': total_groups,
                    'description': 'Number of duplicate groups identified',
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'metric': 'total_duplicates',
                    'value': total_duplicates,
                    'description': 'Total number of duplicate images found',
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'metric': 'deduplication_rate_percent',
                    'value': round((total_duplicates / total_images) * 100, 2),
                    'description': 'Percentage of dataset that are duplicates',
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'metric': 'processing_time_minutes',
                    'value': round(processing_time / 60, 2),
                    'description': 'Total processing time in minutes',
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'metric': 'processing_speed_images_per_second',
                    'value': round(total_images / processing_time, 2),
                    'description': 'Processing speed in images per second',
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'metric': 'configuration_type',
                    'value': 'aggressive',
                    'description': 'Deduplication configuration used',
                    'timestamp': datetime.now().isoformat()
                }
            ]
            
            for row in summary_data:
                writer.writerow(row)
        
        logger.info(f"✅ Summary report generated: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Failed to generate summary report: {e}")
        return None

def run_aggressive_accuracy_test(image_paths: List[str]) -> Dict[str, Any]:
    """Run aggressive accuracy test to find more duplicates."""
    logger.info("🔸 Running aggressive accuracy test (more duplicates)...")
    start_time = time.time()
    
    try:
        # Create aggressive components
        whash_dedup = create_aggressive_whash_deduplicator()
        ssim_calc = create_aggressive_structural_similarity_calculator()
        hybrid_calc = create_aggressive_hybrid_similarity_calculator(whash_dedup, ssim_calc)
        accuracy_dedup = create_aggressive_accuracy_optimized_deduplicator(hybrid_calc, whash_dedup, ssim_calc)
        
        # Progress callback
        def progress_callback(message: str):
            logger.info(f"📊 {message}")
        
        # Run deduplication
        duplicate_groups = accuracy_dedup.find_duplicates(image_paths, progress_callback)
        
        processing_time = time.time() - start_time
        total_duplicates = sum(len(group) - 1 for group in duplicate_groups if len(group) > 1)
        total_groups = len([group for group in duplicate_groups if len(group) > 1])
        
        # Generate CSV report using the existing create_report method
        import tempfile
        temp_dir = tempfile.mkdtemp()
        try:
            # Create empty similarity_scores dict since we don't have them from the accuracy deduplicator
            similarity_scores = {}
            csv_file = accuracy_dedup.create_report(duplicate_groups, similarity_scores, temp_dir)
            summary_file = generate_summary_csv_report(total_duplicates, total_groups, processing_time, len(image_paths))
        except Exception as e:
            logger.error(f"Failed to create report: {e}")
            csv_file = None
            summary_file = None
        
        # Get detailed stats
        whash_stats = whash_dedup.get_stats()
        ssim_stats = ssim_calc.get_stats()
        hybrid_stats = hybrid_calc.get_stats()
        accuracy_stats = accuracy_dedup.get_stats()
        
        logger.info(f"✅ Aggressive Accuracy: {total_duplicates} duplicates in {processing_time:.2f}s")
        
        # Cleanup
        whash_dedup.release()
        ssim_calc.release()
        hybrid_calc.release()
        accuracy_dedup.release()
        
        return {
            'duplicates': total_duplicates,
            'time': processing_time,
            'groups': len(duplicate_groups),
            'duplicate_groups': total_groups,
            'csv_file': csv_file,
            'summary_file': summary_file,
            'temp_dir': temp_dir,
            'whash_stats': whash_stats,
            'ssim_stats': ssim_stats,
            'hybrid_stats': hybrid_stats,
            'accuracy_stats': accuracy_stats
        }
        
    except Exception as e:
        logger.error(f"❌ Aggressive accuracy test failed: {e}")
        return {'duplicates': 0, 'time': 0, 'groups': 0, 'stats': {}}

def main():
    """Main test function."""
    logger.info("🚀 Starting aggressive deduplication test for 5K images...")
    
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
    
    # Aggressive accuracy test
    results['aggressive'] = run_aggressive_accuracy_test(image_paths)
    
    # Print comparison
    logger.info("=" * 80)
    logger.info("📊 AGGRESSIVE DEDUPLICATION COMPARISON RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"🔸 Aggressive Baseline WHash:")
    logger.info(f"   - Duplicates found: {results['whash']['duplicates']}")
    logger.info(f"   - Processing time: {results['whash']['time']:.2f}s")
    if results['whash']['time'] > 0:
        logger.info(f"   - Speed: {len(image_paths)/results['whash']['time']:.1f} images/sec")
        logger.info(f"   - Deduplication rate: {results['whash']['duplicates']/len(image_paths)*100:.2f}%")
    
    logger.info(f"🔸 Aggressive Baseline Color:")
    logger.info(f"   - Duplicates found: {results['color']['duplicates']}")
    logger.info(f"   - Processing time: {results['color']['time']:.2f}s")
    if results['color']['time'] > 0:
        logger.info(f"   - Speed: {len(image_paths)/results['color']['time']:.1f} images/sec")
        logger.info(f"   - Deduplication rate: {results['color']['duplicates']/len(image_paths)*100:.2f}%")
    
    logger.info(f"🔸 Aggressive Accuracy-Optimized:")
    logger.info(f"   - Duplicates found: {results['aggressive']['duplicates']}")
    logger.info(f"   - Processing time: {results['aggressive']['time']:.2f}s")
    if results['aggressive']['time'] > 0:
        logger.info(f"   - Speed: {len(image_paths)/results['aggressive']['time']:.1f} images/sec")
        logger.info(f"   - Deduplication rate: {results['aggressive']['duplicates']/len(image_paths)*100:.2f}%")
    
    # Display CSV report information
    if 'csv_file' in results['aggressive'] and results['aggressive']['csv_file']:
        logger.info(f"📁 CSV Reports Generated:")
        logger.info(f"   - Detailed Report: {results['aggressive']['csv_file']}")
        logger.info(f"   - Summary Report: {results['aggressive']['summary_file']}")
        logger.info(f"   - Duplicate Groups: {results['aggressive'].get('duplicate_groups', 0)}")
        logger.info(f"   - Temp Directory: {results['aggressive'].get('temp_dir', 'N/A')}")
    
    # Improvement analysis
    if results['color']['duplicates'] > 0:
        improvement = ((results['aggressive']['duplicates'] - results['color']['duplicates']) / results['color']['duplicates']) * 100
        logger.info(f"📈 IMPROVEMENT ANALYSIS:")
        logger.info(f"   - vs Baseline Color: {improvement:+.1f}% duplicates found")
    
    # Performance analysis
    if 'whash_stats' in results['aggressive']:
        whash_stats = results['aggressive']['whash_stats']
        hybrid_stats = results['aggressive']['hybrid_stats']
        accuracy_stats = results['aggressive']['accuracy_stats']
        
        logger.info(f"📈 Detailed Performance Stats:")
        logger.info(f"   - WHash: {whash_stats.get('total_images_processed', 0)} images processed")
        logger.info(f"   - SSIM: {hybrid_stats.get('ssim_comparisons', 0)} comparisons")
        logger.info(f"   - Hybrid: {hybrid_stats.get('total_comparisons', 0)} comparisons, {hybrid_stats.get('hit_rate', 0)*100:.1f}% cache hit rate")
        logger.info(f"   - Accuracy: {accuracy_stats.get('whash_groups', 0)} WHash groups, {accuracy_stats.get('verified_groups', 0)} verified")
    
    logger.info("=" * 80)
    logger.info("🎉 Aggressive deduplication test completed successfully!")

if __name__ == "__main__":
    main()

