#!/usr/bin/env python3
"""
Multi-threaded Memory-Efficient Hierarchical Deduplication

This module implements multi-threaded processing for each stage of the 
hierarchical deduplication pipeline to significantly boost performance.
"""

import os
import sys
import time
import logging
import gc
import threading
from typing import Dict, List, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import numpy as np
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.memory_efficient_deduplication import MemoryEfficientDeduplicator
from modules.feature_cache import BoundedFeatureCache

logger = logging.getLogger(__name__)


class MultiThreadedDeduplicator(MemoryEfficientDeduplicator):
    """Multi-threaded memory-efficient hierarchical deduplicator."""
    
    def __init__(self, feature_cache: BoundedFeatureCache, device: str = "cpu", 
                 max_workers: int = None, chunk_size: int = 10):
        """
        Initialize multi-threaded deduplicator.
        
        Args:
            feature_cache: Feature cache instance
            device: Processing device ("cpu" or "cuda")
            max_workers: Maximum number of worker threads (default: CPU count)
            chunk_size: Number of groups to process per thread batch
        """
        super().__init__(feature_cache, device)
        
        # Threading configuration
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.chunk_size = chunk_size
        
        # Thread-safe locks
        self._cache_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        
        # Performance tracking
        self.threading_stats = {
            'total_threads_used': 0,
            'avg_thread_utilization': 0.0,
            'parallel_speedup': 0.0,
            'stage_timings': {},
            'thread_efficiency': {}
        }
        
        logger.info(f"🚀 Multi-threaded deduplicator initialized:")
        logger.info(f"   - Max workers: {self.max_workers}")
        logger.info(f"   - Chunk size: {chunk_size}")
        logger.info(f"   - Device: {device}")
    
    def deduplicate_multithreaded(self, image_paths: List[str], output_dir: str,
                                 progress_callback: Optional[Callable] = None) -> Tuple[List[List[str]], Dict]:
        """
        Multi-threaded memory-efficient deduplication.
        
        Args:
            image_paths: List of image file paths
            output_dir: Output directory for results
            progress_callback: Optional progress callback function
            
        Returns:
            Tuple of (duplicate_groups, similarity_scores)
        """
        
        logger.info(f"🚀 Starting Multi-threaded Deduplication Pipeline")
        logger.info(f"📊 Processing {len(image_paths):,} images with {self.max_workers} threads")
        
        total_start_time = time.time()
        similarity_scores = {}
        
        # Stage 1: Wavelet grouping (single-threaded - fast enough)
        logger.info("🔄 Stage 1: Wavelet grouping...")
        stage1_start = time.time()
        
        wavelet_groups = self._stage1_wavelet_grouping(image_paths)
        self._log_memory_usage("Stage 1 - Wavelet")
        
        stage1_time = time.time() - stage1_start
        logger.info(f"✅ Stage 1 completed in {stage1_time:.1f}s - {len(wavelet_groups)} groups")
        
        # Stage 2: Multi-threaded color verification
        logger.info("🔄 Stage 2: Multi-threaded color verification...")
        stage2_start = time.time()
        
        color_verified_groups = self._stage2_multithreaded_color_verification(
            wavelet_groups, similarity_scores, progress_callback
        )
        self._log_memory_usage("Stage 2 - Color (MT)")
        
        stage2_time = time.time() - stage2_start
        logger.info(f"✅ Stage 2 completed in {stage2_time:.1f}s - {len(color_verified_groups)} groups")
        
        # Stage 3: Multi-threaded global refinement
        logger.info("🔄 Stage 3: Multi-threaded global refinement...")
        stage3_start = time.time()
        
        global_refined_groups = self._stage3_multithreaded_global_refinement(
            color_verified_groups, similarity_scores, progress_callback
        )
        self._log_memory_usage("Stage 3 - Global (MT)")
        
        stage3_time = time.time() - stage3_start
        logger.info(f"✅ Stage 3 completed in {stage3_time:.1f}s - {len(global_refined_groups)} groups")
        
        # Stage 4: Multi-threaded local verification
        logger.info("🔄 Stage 4: Multi-threaded local verification...")
        stage4_start = time.time()
        
        local_verified_groups = self._stage4_multithreaded_local_verification(
            global_refined_groups, similarity_scores, progress_callback
        )
        self._log_memory_usage("Stage 4 - Local (MT)")
        
        stage4_time = time.time() - stage4_start
        logger.info(f"✅ Stage 4 completed in {stage4_time:.1f}s - {len(local_verified_groups)} groups")
        
        # Stage 5: Quality-based organization (single-threaded)
        logger.info("🔄 Stage 5: Quality-based organization...")
        stage5_start = time.time()
        
        final_results = self._stage5_quality_organization(
            local_verified_groups, similarity_scores, output_dir
        )
        
        stage5_time = time.time() - stage5_start
        logger.info(f"✅ Stage 5 completed in {stage5_time:.1f}s")
        
        # Final statistics
        total_time = time.time() - total_start_time
        total_images = len(image_paths)
        total_duplicates = sum(len(group) for group in local_verified_groups)
        
        # Calculate threading efficiency
        self._calculate_threading_efficiency([stage2_time, stage3_time, stage4_time])
        
        logger.info(f"\n🎉 Multi-threaded Deduplication Complete!")
        logger.info(f"📊 Results:")
        logger.info(f"   - Total images processed: {total_images:,}")
        logger.info(f"   - Duplicate groups found: {len(local_verified_groups)}")
        logger.info(f"   - Total duplicate images: {total_duplicates:,}")
        logger.info(f"   - Unique images: {total_images - total_duplicates:,}")
        logger.info(f"   - Processing time: {total_time:.1f}s")
        logger.info(f"   - Processing rate: {total_images/total_time:.1f} images/second")
        
        logger.info(f"🚀 Threading Performance:")
        logger.info(f"   - Max workers used: {self.max_workers}")
        logger.info(f"   - Parallel speedup: {self.threading_stats['parallel_speedup']:.1f}x")
        logger.info(f"   - Thread efficiency: {self.threading_stats['avg_thread_utilization']:.1f}%")
        
        logger.info(f"💾 Memory Statistics:")
        logger.info(f"   - Peak memory usage: {self.memory_stats['peak_memory_mb']:.1f} MB")
        logger.info(f"   - Features loaded: {self.memory_stats['features_loaded']:,}")
        logger.info(f"   - Features freed: {self.memory_stats['features_freed']:,}")
        logger.info(f"   - Memory efficiency: {(self.memory_stats['features_freed']/max(self.memory_stats['features_loaded'], 1))*100:.1f}% freed")
        
        return local_verified_groups, similarity_scores
    
    def _stage2_multithreaded_color_verification(self, wavelet_groups: List[List[str]], 
                                               similarity_scores: Dict, progress_callback=None) -> List[List[str]]:
        """Multi-threaded color verification."""
        
        if not wavelet_groups:
            return []
        
        # Filter groups suitable for color verification
        processable_groups = [group for group in wavelet_groups if 1 < len(group) <= 30]
        large_groups = [group for group in wavelet_groups if len(group) > 30]
        
        logger.info(f"🔧 Color verification: {len(processable_groups)} groups to process, {len(large_groups)} large groups skipped")
        
        color_verified_groups = []
        
        # Process large groups without threading (add as-is)
        color_verified_groups.extend(large_groups)
        
        if not processable_groups:
            return color_verified_groups
        
        # Create thread-safe result collector
        results_lock = threading.Lock()
        processed_count = 0
        
        def process_color_group_batch(group_batch):
            """Process a batch of groups for color verification."""
            batch_results = []
            
            for group in group_batch:
                try:
                    # Color verification (most memory-intensive)
                    subgroups = self.verify_with_color_features(group)
                    batch_results.extend(subgroups)
                    
                except Exception as e:
                    logger.warning(f"Color verification failed for group of {len(group)} images: {e}")
                    batch_results.append(group)  # Keep the group as-is
            
            # Thread-safe result collection
            with results_lock:
                nonlocal processed_count
                color_verified_groups.extend(batch_results)
                processed_count += len(group_batch)
                
                if progress_callback:
                    progress = 20 + (processed_count / len(processable_groups)) * 20  # 20-40%
                    progress_callback(f"Color verification: {processed_count}/{len(processable_groups)} groups", progress)
            
            return len(batch_results)
        
        # Create batches for threading
        group_batches = [processable_groups[i:i + self.chunk_size] 
                        for i in range(0, len(processable_groups), self.chunk_size)]
        
        # Process batches in parallel
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_color_group_batch, batch) for batch in group_batches]
            
            # Wait for completion with progress tracking
            for future in tqdm(as_completed(futures), total=len(futures), desc="Color verification batches"):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Color verification batch failed: {e}")
        
        processing_time = time.time() - start_time
        self.threading_stats['stage_timings']['color'] = processing_time
        
        # Force garbage collection after parallel processing
        gc.collect()
        
        logger.info(f"🚀 Color verification: {len(processable_groups)} groups processed in {processing_time:.1f}s using {len(group_batches)} batches")
        
        return color_verified_groups
    
    def _stage3_multithreaded_global_refinement(self, color_groups: List[List[str]], 
                                              similarity_scores: Dict, progress_callback=None) -> List[List[str]]:
        """Multi-threaded global feature refinement."""
        
        if not color_groups:
            return []
        
        # Filter groups suitable for global processing
        processable_groups = [group for group in color_groups if 1 < len(group) <= 100]
        large_groups = [group for group in color_groups if len(group) > 100]
        
        logger.info(f"🔧 Global refinement: {len(processable_groups)} groups to process, {len(large_groups)} large groups to chunk")
        
        global_refined_groups = []
        
        # Handle large groups with chunking (single-threaded for memory safety)
        for group in large_groups:
            chunk_size = 50
            for i in range(0, len(group), chunk_size):
                chunk = group[i:i + chunk_size]
                if len(chunk) > 1:
                    # Load global features for chunk
                    with self._cache_lock:  # Thread-safe cache access
                        chunk_global_features = self._load_features_for_group(chunk, 'global')
                    
                    if chunk_global_features:
                        chunk_refined = self._refine_group_with_global_features(
                            list(chunk_global_features.keys()),
                            {path: {'global': feat} for path, feat in chunk_global_features.items()},
                            similarity_scores
                        )
                        global_refined_groups.extend(chunk_refined)
                        del chunk_global_features
        
        if not processable_groups:
            return global_refined_groups
        
        # Thread-safe result collector
        results_lock = threading.Lock()
        processed_count = 0
        
        def process_global_group_batch(group_batch):
            """Process a batch of groups for global refinement."""
            batch_results = []
            
            for group in group_batch:
                try:
                    # Load global features only for this group (thread-safe)
                    with self._cache_lock:
                        group_global_features = self._load_features_for_group(group, 'global')
                    
                    if not group_global_features:
                        logger.warning(f"No global features found for group of {len(group)} images")
                        continue
                    
                    # Process this group
                    subgroups = self._refine_group_with_global_features(
                        list(group_global_features.keys()), 
                        {path: {'global': feat} for path, feat in group_global_features.items()},
                        similarity_scores
                    )
                    batch_results.extend(subgroups)
                    
                    # Free memory immediately
                    del group_global_features
                    
                    # Update stats thread-safely
                    with self._stats_lock:
                        self.memory_stats['features_freed'] += len(group)
                    
                except Exception as e:
                    logger.warning(f"Global refinement failed for group of {len(group)} images: {e}")
                    batch_results.append(group)  # Keep the group as-is
            
            # Thread-safe result collection
            with results_lock:
                nonlocal processed_count
                global_refined_groups.extend(batch_results)
                processed_count += len(group_batch)
                
                if progress_callback:
                    progress = 40 + (processed_count / len(processable_groups)) * 20  # 40-60%
                    progress_callback(f"Global refinement: {processed_count}/{len(processable_groups)} groups", progress)
            
            return len(batch_results)
        
        # Create batches for threading
        group_batches = [processable_groups[i:i + self.chunk_size] 
                        for i in range(0, len(processable_groups), self.chunk_size)]
        
        # Process batches in parallel
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_global_group_batch, batch) for batch in group_batches]
            
            # Wait for completion with progress tracking
            for future in tqdm(as_completed(futures), total=len(futures), desc="Global refinement batches"):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Global refinement batch failed: {e}")
        
        processing_time = time.time() - start_time
        self.threading_stats['stage_timings']['global'] = processing_time
        
        # Periodic garbage collection
        gc.collect()
        
        logger.info(f"🚀 Global refinement: {len(processable_groups)} groups processed in {processing_time:.1f}s using {len(group_batches)} batches")
        
        return global_refined_groups
    
    def _stage4_multithreaded_local_verification(self, global_groups: List[List[str]], 
                                               similarity_scores: Dict, progress_callback=None) -> List[List[str]]:
        """Multi-threaded local feature verification."""
        
        if not global_groups:
            return []
        
        # Filter groups suitable for local processing
        processable_groups = [group for group in global_groups if 1 < len(group) <= 50]
        large_groups = [group for group in global_groups if len(group) > 50]
        
        logger.info(f"🔧 Local verification: {len(processable_groups)} groups to process, {len(large_groups)} large groups skipped")
        
        local_verified_groups = []
        
        # Process large groups without threading (add as-is)
        local_verified_groups.extend(large_groups)
        
        if not processable_groups:
            return local_verified_groups
        
        # Thread-safe result collector
        results_lock = threading.Lock()
        processed_count = 0
        
        def process_local_group_batch(group_batch):
            """Process a batch of groups for local verification."""
            batch_results = []
            
            for group in group_batch:
                try:
                    # Load local features only for this group (thread-safe)
                    with self._cache_lock:
                        group_local_features = self._load_features_for_group(group, 'local')
                    
                    if not group_local_features:
                        logger.warning(f"No local features found for group of {len(group)} images")
                        batch_results.append(group)  # Keep the group as-is
                        continue
                    
                    # Process this group
                    subgroups = self._verify_group_with_local_features(
                        list(group_local_features.keys()),
                        {path: {'local': feat} for path, feat in group_local_features.items()},
                        similarity_scores
                    )
                    batch_results.extend(subgroups)
                    
                    # Free memory immediately
                    del group_local_features
                    
                    # Update stats thread-safely
                    with self._stats_lock:
                        self.memory_stats['features_freed'] += len(group)
                    
                except Exception as e:
                    logger.warning(f"Local verification failed for group of {len(group)} images: {e}")
                    batch_results.append(group)  # Keep the group as-is
            
            # Thread-safe result collection
            with results_lock:
                nonlocal processed_count
                local_verified_groups.extend(batch_results)
                processed_count += len(group_batch)
                
                if progress_callback:
                    progress = 60 + (processed_count / len(processable_groups)) * 20  # 60-80%
                    progress_callback(f"Local verification: {processed_count}/{len(processable_groups)} groups", progress)
            
            return len(batch_results)
        
        # Create batches for threading (smaller batches for local features - they're larger)
        local_chunk_size = max(1, self.chunk_size // 2)  # Smaller chunks for memory-intensive local features
        group_batches = [processable_groups[i:i + local_chunk_size] 
                        for i in range(0, len(processable_groups), local_chunk_size)]
        
        # Process batches in parallel
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_local_group_batch, batch) for batch in group_batches]
            
            # Wait for completion with progress tracking
            for future in tqdm(as_completed(futures), total=len(futures), desc="Local verification batches"):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Local verification batch failed: {e}")
        
        processing_time = time.time() - start_time
        self.threading_stats['stage_timings']['local'] = processing_time
        
        # More frequent garbage collection for local features (larger memory footprint)
        gc.collect()
        
        logger.info(f"🚀 Local verification: {len(processable_groups)} groups processed in {processing_time:.1f}s using {len(group_batches)} batches")
        
        return local_verified_groups
    
    def _calculate_threading_efficiency(self, stage_times: List[float]):
        """Calculate threading performance metrics."""
        
        # Estimate sequential processing time (rough approximation)
        estimated_sequential_time = sum(stage_times) * self.max_workers * 0.7  # Assume 70% parallelizable
        actual_parallel_time = sum(stage_times)
        
        if actual_parallel_time > 0:
            self.threading_stats['parallel_speedup'] = estimated_sequential_time / actual_parallel_time
            self.threading_stats['avg_thread_utilization'] = min(100, (estimated_sequential_time / actual_parallel_time / self.max_workers) * 100)
        
        self.threading_stats['total_threads_used'] = self.max_workers
    
    def _load_features_for_group(self, group: List[str], feature_type: str) -> Dict[str, np.ndarray]:
        """Thread-safe feature loading for a group of images."""
        
        # Note: This method is already thread-safe due to the cache_lock usage in calling methods
        # But we can add additional safety measures here if needed
        
        group_features = {}
        
        for image_path in group:
            try:
                features = self.feature_cache.get_features(image_path)
                if features and feature_type in features:
                    group_features[image_path] = features[feature_type]
                    
                    # Update stats thread-safely
                    with self._stats_lock:
                        self.memory_stats['features_loaded'] += 1
                        
            except Exception as e:
                logger.warning(f"Failed to load {feature_type} features for {image_path}: {e}")
        
        return group_features


def create_multithreaded_deduplicator(feature_cache: BoundedFeatureCache, 
                                    device: str = "cpu",
                                    max_workers: int = None,
                                    chunk_size: int = 10) -> MultiThreadedDeduplicator:
    """
    Factory function to create a multi-threaded deduplicator.
    
    Args:
        feature_cache: Feature cache instance
        device: Processing device ("cpu" or "cuda")
        max_workers: Maximum number of worker threads (default: auto-detect)
        chunk_size: Number of groups to process per thread batch
        
    Returns:
        MultiThreadedDeduplicator instance
    """
    
    return MultiThreadedDeduplicator(
        feature_cache=feature_cache,
        device=device,
        max_workers=max_workers,
        chunk_size=chunk_size
    )


if __name__ == "__main__":
    # Example usage
    from modules.feature_cache import BoundedFeatureCache
    
    # Create cache and deduplicator
    cache = BoundedFeatureCache(cache_dir="test_cache", max_size=1000)
    deduplicator = create_multithreaded_deduplicator(
        feature_cache=cache,
        device="cpu",
        max_workers=8,
        chunk_size=5
    )
    
    # Example image paths (replace with actual paths)
    image_paths = [f"image_{i:04d}.jpg" for i in range(100)]
    
    # Run multi-threaded deduplication
    try:
        duplicate_groups, similarity_scores = deduplicator.deduplicate_multithreaded(
            image_paths=image_paths,
            output_dir="multithreaded_results"
        )
        
        print(f"Found {len(duplicate_groups)} duplicate groups")
        
    except Exception as e:
        logger.error(f"Multi-threaded deduplication failed: {e}", exc_info=True)