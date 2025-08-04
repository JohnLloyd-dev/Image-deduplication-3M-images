#!/usr/bin/env python3
"""
Comprehensive Multi-threading Test

This script runs multiple test scenarios to thoroughly validate
the multi-threading implementation under different conditions.
"""

import sys
import os
import time
import logging
import numpy as np
from typing import Dict, List
import tempfile
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.feature_cache import BoundedFeatureCache
from modules.memory_efficient_deduplication import MemoryEfficientDeduplicator
from modules.multithreaded_deduplication import MultiThreadedDeduplicator
from modules.threading_optimizer import ThreadingOptimizer, create_optimized_deduplicator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_dataset_with_duplicates(num_images: int = 100) -> Dict[str, Dict]:
    """Create a test dataset with realistic duplicate groups."""
    
    logger.info(f"Creating test dataset with {num_images} images and realistic duplicates...")
    
    features = {}
    
    # Create more realistic duplicate groups
    duplicate_groups = [
        # Group 1: 8 very similar images (vacation photos)
        list(range(0, 8)),
        # Group 2: 6 similar images (portraits)  
        list(range(15, 21)),
        # Group 3: 5 similar images (buildings)
        list(range(30, 35)),
        # Group 4: 4 similar images (landscapes)
        list(range(50, 54)),
        # Group 5: 3 similar images (food photos)
        list(range(70, 73)),
        # Rest are unique images
    ]
    
    for i in range(num_images):
        img_path = f"test_image_{i:04d}.jpg"
        
        # Determine if this image is part of a duplicate group
        group_id = None
        for gid, group in enumerate(duplicate_groups):
            if i in group:
                group_id = gid
                break
        
        if group_id is not None:
            # Create similar features for duplicate groups
            base_seed = group_id * 1000
            np.random.seed(base_seed)
            
            # Add small variations to make them similar but not identical
            variation_factor = (i % len(duplicate_groups[group_id])) * 0.05
            
            # Base features for the group
            wavelet_base = np.random.randint(0, 2, 32, dtype=np.uint8)
            global_base = np.random.randn(512).astype(np.float32)
            color_base = np.random.randn(256).astype(np.float32)
            
            # Add controlled variations
            wavelet = wavelet_base.copy()
            # Flip 1-2 bits randomly for slight variation
            if np.random.random() < 0.2:  # 20% chance
                flip_indices = np.random.choice(32, size=np.random.randint(1, 3), replace=False)
                for idx in flip_indices:
                    wavelet[idx] = 1 - wavelet[idx]
            
            # Add small noise to global features
            global_feat = global_base + np.random.randn(512).astype(np.float32) * 0.03
            
            # Add small noise to color features
            color_feat = color_base + np.random.randn(256).astype(np.float32) * 0.05
            
            # Create similar local features
            keypoints = np.random.randn(25, 2).astype(np.float32) + variation_factor
            descriptors = np.random.randn(25, 128).astype(np.float32) + variation_factor * 0.1
            
        else:
            # Create unique features for non-duplicate images
            np.random.seed(i * 42 + 12345)  # Different seed space
            wavelet = np.random.randint(0, 2, 32, dtype=np.uint8)
            global_feat = np.random.randn(512).astype(np.float32)
            color_feat = np.random.randn(256).astype(np.float32)
            keypoints = np.random.randn(25, 2).astype(np.float32)
            descriptors = np.random.randn(25, 128).astype(np.float32)
        
        features[img_path] = {
            'wavelet': wavelet,
            'global': global_feat,
            'local': {
                'keypoints': keypoints,
                'descriptors': descriptors
            },
            'color_features': color_feat
        }
    
    logger.info(f"Created {num_images} test images with {len(duplicate_groups)} duplicate groups")
    logger.info(f"Expected duplicates: {sum(len(group) for group in duplicate_groups)} images")
    return features


def run_performance_test(dataset_size: int, test_name: str):
    """Run a performance test with specified dataset size."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PERFORMANCE TEST: {test_name}")
    logger.info(f"Dataset Size: {dataset_size} images")
    logger.info(f"{'='*60}")
    
    # Create test dataset
    features = create_test_dataset_with_duplicates(dataset_size)
    image_paths = list(features.keys())
    
    results = {}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "cache")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup cache
        cache = BoundedFeatureCache(cache_dir=cache_dir, max_size=dataset_size * 2)
        for path, feat in features.items():
            cache.put_features(path, feat)
        
        # Test 1: Single-threaded
        logger.info(f"\n🔄 Testing Single-threaded (baseline)...")
        start_time = time.time()
        
        try:
            single_deduplicator = MemoryEfficientDeduplicator(
                feature_cache=cache,
                device="cpu"
            )
            
            single_groups, single_scores = single_deduplicator.deduplicate_memory_efficient(
                image_paths=image_paths,
                output_dir=os.path.join(output_dir, "single")
            )
            
            single_time = time.time() - start_time
            
            results['single'] = {
                'time': single_time,
                'groups': len(single_groups),
                'duplicates': sum(len(group) for group in single_groups),
                'memory': single_deduplicator.memory_stats.get('peak_memory_mb', 0),
                'rate': dataset_size / single_time
            }
            
            logger.info(f"✅ Single-threaded: {single_time:.2f}s, {len(single_groups)} groups")
            
        except Exception as e:
            logger.error(f"Single-threaded test failed: {e}")
            results['single'] = None
        
        # Test 2: Multi-threaded (auto-optimized)
        logger.info(f"\n🚀 Testing Multi-threaded (auto-optimized)...")
        start_time = time.time()
        
        try:
            multi_deduplicator = create_optimized_deduplicator(
                feature_cache=cache,
                device="cpu"
            )
            
            logger.info(f"   Configuration: {multi_deduplicator.max_workers} workers, {multi_deduplicator.chunk_size} chunk size")
            
            multi_groups, multi_scores = multi_deduplicator.deduplicate_multithreaded(
                image_paths=image_paths,
                output_dir=os.path.join(output_dir, "multi")
            )
            
            multi_time = time.time() - start_time
            
            results['multi'] = {
                'time': multi_time,
                'groups': len(multi_groups),
                'duplicates': sum(len(group) for group in multi_groups),
                'memory': multi_deduplicator.memory_stats.get('peak_memory_mb', 0),
                'rate': dataset_size / multi_time,
                'workers': multi_deduplicator.max_workers,
                'speedup': multi_deduplicator.threading_stats.get('parallel_speedup', 1.0),
                'efficiency': multi_deduplicator.threading_stats.get('avg_thread_utilization', 0)
            }
            
            logger.info(f"✅ Multi-threaded: {multi_time:.2f}s, {len(multi_groups)} groups")
            
        except Exception as e:
            logger.error(f"Multi-threaded test failed: {e}")
            results['multi'] = None
        
        # Test 3: Multi-threaded (manual high-performance config)
        logger.info(f"\n⚡ Testing Multi-threaded (high-performance config)...")
        start_time = time.time()
        
        try:
            hp_deduplicator = MultiThreadedDeduplicator(
                feature_cache=cache,
                device="cpu",
                max_workers=6,  # Higher worker count
                chunk_size=8   # Larger chunks
            )
            
            logger.info(f"   Configuration: {hp_deduplicator.max_workers} workers, {hp_deduplicator.chunk_size} chunk size")
            
            hp_groups, hp_scores = hp_deduplicator.deduplicate_multithreaded(
                image_paths=image_paths,
                output_dir=os.path.join(output_dir, "hp")
            )
            
            hp_time = time.time() - start_time
            
            results['high_perf'] = {
                'time': hp_time,
                'groups': len(hp_groups),
                'duplicates': sum(len(group) for group in hp_groups),
                'memory': hp_deduplicator.memory_stats.get('peak_memory_mb', 0),
                'rate': dataset_size / hp_time,
                'workers': hp_deduplicator.max_workers,
                'speedup': hp_deduplicator.threading_stats.get('parallel_speedup', 1.0),
                'efficiency': hp_deduplicator.threading_stats.get('avg_thread_utilization', 0)
            }
            
            logger.info(f"✅ High-performance: {hp_time:.2f}s, {len(hp_groups)} groups")
            
        except Exception as e:
            logger.error(f"High-performance test failed: {e}")
            results['high_perf'] = None
        
        cache.clear()
    
    return results


def analyze_results(results: Dict, test_name: str, dataset_size: int):
    """Analyze and display test results."""
    
    logger.info(f"\n📊 RESULTS ANALYSIS: {test_name}")
    logger.info(f"{'='*50}")
    
    if results['single'] and results['multi']:
        single = results['single']
        multi = results['multi']
        
        speedup = single['time'] / multi['time']
        memory_change = ((multi['memory'] - single['memory']) / single['memory']) * 100 if single['memory'] > 0 else 0
        
        logger.info(f"\n⏱️  Performance Comparison:")
        logger.info(f"   Single-threaded: {single['time']:.2f}s ({single['rate']:.1f} images/sec)")
        logger.info(f"   Multi-threaded:  {multi['time']:.2f}s ({multi['rate']:.1f} images/sec)")
        logger.info(f"   Speedup:         {speedup:.1f}x")
        logger.info(f"   Efficiency:      {(speedup / multi['workers']) * 100:.1f}%")
        
        logger.info(f"\n💾 Memory Comparison:")
        logger.info(f"   Single-threaded: {single['memory']:.1f} MB")
        logger.info(f"   Multi-threaded:  {multi['memory']:.1f} MB")
        logger.info(f"   Memory change:   {memory_change:+.1f}%")
        
        logger.info(f"\n🎯 Quality Comparison:")
        logger.info(f"   Single-threaded: {single['groups']} groups, {single['duplicates']} duplicates")
        logger.info(f"   Multi-threaded:  {multi['groups']} groups, {multi['duplicates']} duplicates")
        
        quality_match = (single['groups'] == multi['groups'] and 
                        single['duplicates'] == multi['duplicates'])
        logger.info(f"   Quality match:   {'✅ Identical' if quality_match else '⚠️ Different'}")
        
        logger.info(f"\n🚀 Threading Analysis:")
        logger.info(f"   Workers used:    {multi['workers']}")
        logger.info(f"   Thread speedup:  {multi['speedup']:.1f}x")
        logger.info(f"   Thread efficiency: {multi['efficiency']:.1f}%")
    
    if results['high_perf']:
        hp = results['high_perf']
        logger.info(f"\n⚡ High-Performance Config:")
        logger.info(f"   Time:            {hp['time']:.2f}s ({hp['rate']:.1f} images/sec)")
        logger.info(f"   Workers:         {hp['workers']}")
        logger.info(f"   Thread speedup:  {hp['speedup']:.1f}x")
        logger.info(f"   Thread efficiency: {hp['efficiency']:.1f}%")
        
        if results['multi']:
            hp_vs_auto = results['multi']['time'] / hp['time']
            logger.info(f"   vs Auto-config:  {hp_vs_auto:.1f}x {'faster' if hp_vs_auto > 1 else 'slower'}")


def run_comprehensive_tests():
    """Run comprehensive multi-threading tests."""
    
    logger.info("🧪 COMPREHENSIVE MULTI-THREADING TESTS")
    logger.info("="*70)
    
    # Show system info
    optimizer = ThreadingOptimizer()
    system_info = optimizer.get_system_info()
    config = optimizer.get_optimal_config()
    
    logger.info(f"\n🖥️  System Information:")
    logger.info(f"   Platform: {system_info['platform']}")
    logger.info(f"   CPU Cores: {system_info['cpu_count_logical']}")
    logger.info(f"   Memory: {system_info['total_memory_gb']:.1f} GB")
    logger.info(f"   Auto-config: {config['max_workers']} workers, {config['chunk_size']} chunk size")
    
    # Test scenarios
    test_scenarios = [
        (50, "Small Dataset"),
        (100, "Medium Dataset"),
        (200, "Large Dataset")
    ]
    
    all_results = {}
    
    for dataset_size, test_name in test_scenarios:
        try:
            results = run_performance_test(dataset_size, test_name)
            all_results[dataset_size] = results
            analyze_results(results, test_name, dataset_size)
            
        except Exception as e:
            logger.error(f"Test {test_name} failed: {e}")
            continue
    
    # Overall summary
    logger.info(f"\n🎉 COMPREHENSIVE TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for dataset_size, results in all_results.items():
        if results['single'] and results['multi']:
            speedup = results['single']['time'] / results['multi']['time']
            efficiency = (speedup / results['multi']['workers']) * 100
            
            logger.info(f"\n📊 {dataset_size} images:")
            logger.info(f"   Speedup: {speedup:.1f}x")
            logger.info(f"   Efficiency: {efficiency:.1f}%")
            logger.info(f"   Rate: {results['multi']['rate']:.1f} images/sec")
    
    logger.info(f"\n✅ All comprehensive tests completed!")


if __name__ == "__main__":
    try:
        run_comprehensive_tests()
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
    except Exception as e:
        logger.error(f"Tests failed: {e}", exc_info=True)
        sys.exit(1)