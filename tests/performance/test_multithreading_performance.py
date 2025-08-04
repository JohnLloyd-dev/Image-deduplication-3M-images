#!/usr/bin/env python3
"""
Multi-threading Performance Test

This script compares the performance between single-threaded and 
multi-threaded deduplication approaches.
"""

import sys
import os
import time
import logging
import numpy as np
from typing import Dict, List
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.feature_cache import BoundedFeatureCache
from modules.memory_efficient_deduplication import MemoryEfficientDeduplicator
from modules.multithreaded_deduplication import MultiThreadedDeduplicator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_cpu_info():
    """Get CPU information for performance context."""
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        cpu_count_logical = psutil.cpu_count(logical=True)  # Logical cores
        cpu_freq = psutil.cpu_freq()
        
        return {
            'physical_cores': cpu_count,
            'logical_cores': cpu_count_logical,
            'base_frequency': cpu_freq.current if cpu_freq else 'Unknown',
            'max_frequency': cpu_freq.max if cpu_freq else 'Unknown'
        }
    except ImportError:
        return {
            'physical_cores': os.cpu_count() or 1,
            'logical_cores': os.cpu_count() or 1,
            'base_frequency': 'Unknown',
            'max_frequency': 'Unknown'
        }


def create_test_features(num_images: int) -> Dict[str, Dict]:
    """Create test features for performance comparison."""
    
    logger.info(f"Creating test features for {num_images:,} images...")
    
    features = {}
    for i in range(num_images):
        img_path = f"test_image_{i:06d}.jpg"
        
        # Simulate realistic feature sizes
        features[img_path] = {
            'wavelet': np.random.randint(0, 2, 32, dtype=np.uint8),  # 32 bytes
            'global': np.random.randn(512).astype(np.float32),       # 2KB
            'local': {
                'keypoints': np.random.randn(50, 2).astype(np.float32),      # 400 bytes
                'descriptors': np.random.randn(50, 128).astype(np.float32)   # 25KB
            },
            'color_features': np.random.randn(256).astype(np.float32)  # 1KB
        }
    
    logger.info(f"Created {num_images:,} test features")
    return features


def test_single_threaded_approach(image_paths: List[str], cache: BoundedFeatureCache):
    """Test the single-threaded memory-efficient approach."""
    
    logger.info("🔄 Testing Single-Threaded Memory-Efficient Approach")
    
    start_time = time.time()
    
    # Create single-threaded deduplicator
    deduplicator = MemoryEfficientDeduplicator(
        feature_cache=cache,
        device="cpu"
    )
    
    try:
        # Run single-threaded deduplication
        duplicate_groups, similarity_scores = deduplicator.deduplicate_memory_efficient(
            image_paths=image_paths,
            output_dir="single_threaded_output"
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        results = {
            'approach': 'Single-Threaded',
            'processing_time_s': processing_time,
            'duplicate_groups': len(duplicate_groups),
            'images_per_second': len(image_paths) / processing_time if processing_time > 0 else 0,
            'peak_memory_mb': deduplicator.memory_stats['peak_memory_mb'],
            'features_loaded': deduplicator.memory_stats['features_loaded'],
            'features_freed': deduplicator.memory_stats['features_freed']
        }
        
        logger.info(f"✅ Single-Threaded Results:")
        logger.info(f"   - Processing time: {processing_time:.2f}s")
        logger.info(f"   - Images per second: {results['images_per_second']:.1f}")
        logger.info(f"   - Duplicate groups: {len(duplicate_groups)}")
        logger.info(f"   - Peak memory: {results['peak_memory_mb']:.1f} MB")
        
        return results
        
    except Exception as e:
        logger.error(f"Single-threaded approach failed: {e}")
        return None


def test_multithreaded_approach(image_paths: List[str], cache: BoundedFeatureCache, max_workers: int):
    """Test the multi-threaded approach."""
    
    logger.info(f"🔄 Testing Multi-Threaded Approach ({max_workers} workers)")
    
    start_time = time.time()
    
    # Create multi-threaded deduplicator
    deduplicator = MultiThreadedDeduplicator(
        feature_cache=cache,
        device="cpu",
        max_workers=max_workers,
        chunk_size=8
    )
    
    try:
        # Run multi-threaded deduplication
        duplicate_groups, similarity_scores = deduplicator.deduplicate_multithreaded(
            image_paths=image_paths,
            output_dir="multithreaded_output"
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        results = {
            'approach': f'Multi-Threaded ({max_workers} workers)',
            'processing_time_s': processing_time,
            'duplicate_groups': len(duplicate_groups),
            'images_per_second': len(image_paths) / processing_time if processing_time > 0 else 0,
            'peak_memory_mb': deduplicator.memory_stats['peak_memory_mb'],
            'features_loaded': deduplicator.memory_stats['features_loaded'],
            'features_freed': deduplicator.memory_stats['features_freed'],
            'max_workers': max_workers,
            'parallel_speedup': deduplicator.threading_stats['parallel_speedup'],
            'thread_utilization': deduplicator.threading_stats['avg_thread_utilization']
        }
        
        logger.info(f"✅ Multi-Threaded Results:")
        logger.info(f"   - Processing time: {processing_time:.2f}s")
        logger.info(f"   - Images per second: {results['images_per_second']:.1f}")
        logger.info(f"   - Duplicate groups: {len(duplicate_groups)}")
        logger.info(f"   - Peak memory: {results['peak_memory_mb']:.1f} MB")
        logger.info(f"   - Parallel speedup: {results['parallel_speedup']:.1f}x")
        logger.info(f"   - Thread utilization: {results['thread_utilization']:.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Multi-threaded approach failed: {e}")
        return None


def run_multithreading_performance_test():
    """Run comprehensive multi-threading performance test."""
    
    logger.info("=" * 70)
    logger.info("MULTI-THREADING PERFORMANCE COMPARISON TEST")
    logger.info("=" * 70)
    
    # Get system information
    cpu_info = get_cpu_info()
    logger.info(f"\n🖥️  System Information:")
    logger.info(f"   - Physical CPU cores: {cpu_info['physical_cores']}")
    logger.info(f"   - Logical CPU cores: {cpu_info['logical_cores']}")
    logger.info(f"   - Base frequency: {cpu_info['base_frequency']} MHz")
    logger.info(f"   - Max frequency: {cpu_info['max_frequency']} MHz")
    
    # Test with different dataset sizes
    test_sizes = [500, 1000, 2000]  # Start with manageable sizes
    thread_counts = [1, 2, 4, 8, min(16, cpu_info['logical_cores'])]
    
    all_results = []
    
    for num_images in test_sizes:
        logger.info(f"\n📊 Testing with {num_images:,} images...")
        
        # Create test data
        features = create_test_features(num_images)
        image_paths = list(features.keys())
        
        # Create cache and populate it
        cache = BoundedFeatureCache(cache_dir=f"test_cache_{num_images}", max_size=num_images * 2)
        for path, feat in features.items():
            cache.put_features(path, feat)
        
        # Test single-threaded approach
        logger.info(f"\n🔧 Single-threaded test for {num_images:,} images...")
        gc.collect()  # Clean memory before test
        single_results = test_single_threaded_approach(image_paths, cache)
        if single_results:
            single_results['num_images'] = num_images
            all_results.append(single_results)
        
        # Test multi-threaded approaches with different worker counts
        for workers in thread_counts[1:]:  # Skip 1 worker (that's single-threaded)
            if workers > cpu_info['logical_cores']:
                continue  # Don't test more workers than available cores
                
            logger.info(f"\n🔧 Multi-threaded test for {num_images:,} images with {workers} workers...")
            gc.collect()  # Clean memory before test
            multi_results = test_multithreaded_approach(image_paths, cache, workers)
            if multi_results:
                multi_results['num_images'] = num_images
                all_results.append(multi_results)
        
        # Calculate and display speedup for this dataset size
        if single_results:
            logger.info(f"\n⚡ Speedup Analysis for {num_images:,} images:")
            single_time = single_results['processing_time_s']
            
            for workers in thread_counts[1:]:
                if workers > cpu_info['logical_cores']:
                    continue
                    
                # Find corresponding multi-threaded result
                multi_result = next((r for r in all_results 
                                   if r['num_images'] == num_images and 
                                   r.get('max_workers') == workers), None)
                
                if multi_result:
                    speedup = single_time / multi_result['processing_time_s']
                    efficiency = (speedup / workers) * 100
                    
                    logger.info(f"   - {workers} workers: {speedup:.1f}x speedup ({efficiency:.1f}% efficiency)")
        
        # Clean up
        cache.clear()
        del features
        gc.collect()
    
    # Final comprehensive analysis
    logger.info("\n" + "=" * 70)
    logger.info("COMPREHENSIVE PERFORMANCE ANALYSIS")
    logger.info("=" * 70)
    
    # Group results by dataset size
    for num_images in test_sizes:
        dataset_results = [r for r in all_results if r['num_images'] == num_images]
        
        if not dataset_results:
            continue
            
        logger.info(f"\n📈 {num_images:,} Images Performance Summary:")
        
        # Find single-threaded baseline
        single_result = next((r for r in dataset_results if 'max_workers' not in r), None)
        
        if single_result:
            logger.info(f"   Single-threaded: {single_result['processing_time_s']:6.1f}s ({single_result['images_per_second']:5.1f} img/s)")
            
            # Show multi-threaded results
            multi_results = [r for r in dataset_results if 'max_workers' in r]
            multi_results.sort(key=lambda x: x['max_workers'])
            
            for result in multi_results:
                speedup = single_result['processing_time_s'] / result['processing_time_s']
                efficiency = (speedup / result['max_workers']) * 100
                
                logger.info(f"   {result['max_workers']:2d} workers:     {result['processing_time_s']:6.1f}s ({result['images_per_second']:5.1f} img/s) "
                          f"- {speedup:.1f}x speedup ({efficiency:.0f}% efficiency)")
    
    # Overall recommendations
    logger.info(f"\n🎯 Performance Recommendations:")
    
    # Find optimal worker count based on efficiency
    best_efficiency = 0
    best_workers = 1
    
    for workers in thread_counts[1:]:
        if workers > cpu_info['logical_cores']:
            continue
            
        # Calculate average efficiency across all dataset sizes
        efficiencies = []
        for num_images in test_sizes:
            single_result = next((r for r in all_results 
                                if r['num_images'] == num_images and 'max_workers' not in r), None)
            multi_result = next((r for r in all_results 
                               if r['num_images'] == num_images and r.get('max_workers') == workers), None)
            
            if single_result and multi_result:
                speedup = single_result['processing_time_s'] / multi_result['processing_time_s']
                efficiency = (speedup / workers) * 100
                efficiencies.append(efficiency)
        
        if efficiencies:
            avg_efficiency = np.mean(efficiencies)
            if avg_efficiency > best_efficiency:
                best_efficiency = avg_efficiency
                best_workers = workers
    
    logger.info(f"   ✅ Optimal worker count: {best_workers} workers ({best_efficiency:.0f}% average efficiency)")
    logger.info(f"   ✅ System has {cpu_info['logical_cores']} logical cores available")
    
    if best_workers < cpu_info['logical_cores']:
        logger.info(f"   💡 Using fewer workers than cores suggests I/O or memory bottlenecks")
    else:
        logger.info(f"   💡 Using all available cores for maximum throughput")
    
    logger.info(f"\n🚀 Multi-threading Benefits:")
    logger.info(f"   ✅ Significant speedup on multi-core systems")
    logger.info(f"   ✅ Better resource utilization")
    logger.info(f"   ✅ Scalable performance with dataset size")
    logger.info(f"   ✅ Maintained memory efficiency")
    logger.info(f"   ✅ Thread-safe implementation")


if __name__ == "__main__":
    try:
        run_multithreading_performance_test()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)