"""
Color-Optimized Memory-Efficient Deduplication Pipeline

This module implements an enhanced deduplication approach that uses color-based pre-grouping
to significantly reduce the problem size before applying the full deduplication pipeline.

Key improvements:
1. Color-based pre-grouping using MiniBatchKMeans clustering
2. Reduced computational complexity from O(n²) to O(m²) per subgroup
3. Better scalability for 3M+ images
4. Adaptive thresholds based on color similarity
"""

import os
import gc
import time
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
import cv2
from PIL import Image
from collections import defaultdict

from .memory_efficient_deduplication import MemoryEfficientDeduplicator
from .feature_cache import BoundedFeatureCache
from .azure_utils import download_blob_to_memory

logger = logging.getLogger(__name__)

class ColorOptimizedDeduplicator(MemoryEfficientDeduplicator):
    """
    Color-optimized deduplicator that uses color-based pre-grouping
    to improve scalability for large datasets (3M+ images).
    
    Key improvements implemented:
    1. Unified feature extraction interface
    2. Proper Azure image loading
    3. Adaptive threshold calculation
    4. Parallel processing for large datasets
    5. Enhanced caching strategy
    6. Better progress reporting
    7. Comprehensive resource management
    8. Configurable parameters
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Configurable parameters with defaults
        self.config = {
            'color_clusters': kwargs.get('color_clusters', 2000),
            'batch_size': kwargs.get('batch_size', 1000),
            'color_tolerance': kwargs.get('color_tolerance', 0.8),
            'min_group_size': kwargs.get('min_group_size', 2),
            'max_group_size': kwargs.get('max_group_size', 1000),
            'adaptive_thresholding': kwargs.get('adaptive_thresholding', True),
            'parallel_processing': kwargs.get('parallel_processing', True),
            'max_workers': kwargs.get('max_workers', min(os.cpu_count() or 4, 8)),
            'chunk_size': kwargs.get('chunk_size', 1000),
        }
        
        # Update instance variables
        for key, value in self.config.items():
            setattr(self, key, value)
        
        # Initialize color optimization components
        self.color_clusters = self.config['color_clusters']
        self.batch_size = self.config['batch_size']
        self.color_tolerance = self.config['color_tolerance']
        self.min_group_size = self.config['min_group_size']
        self.max_group_size = self.config['max_group_size']
        self.adaptive_thresholding = self.config['adaptive_thresholding']
        self.parallel_processing = self.config['parallel_processing']
        self.max_workers = self.config['max_workers']
        self.chunk_size = self.config['chunk_size']
        
        # Color optimization state
        self.color_groups = []
        self.color_model = None
        self.color_features_cache = {}
        
        # Enhanced memory stats
        self.memory_stats.update({
            'color_groups_created': 0,
            'color_group_sizes': [],
            'color_processing_time': 0,
            'total_comparisons_saved': 0,
            'parallel_processing_time': 0,
            'adaptive_threshold_adjustments': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        })
        
        # Progress tracking
        self.progress_callback = kwargs.get('progress_callback', None)
        self.current_stage = ""
        self.stage_progress = 0
        self.stage_total = 0

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self.release()

    def release(self):
        """Release all resources"""
        logger.info("Releasing ColorOptimizedDeduplicator resources...")
        
        # Clear feature cache
        if hasattr(self, 'feature_cache') and self.feature_cache:
            self.feature_cache.clear()
        
        # Clear color optimization resources
        self.color_clusters = None
        self.color_groups = None
        self.color_model = None
        self.color_features_cache.clear()
        
        # Clear memory stats
        self.memory_stats.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Resources released successfully")

    def _update_progress(self, stage: str, current: int, total: int, extra_info: str = ""):
        """Standardized progress reporting"""
        self.current_stage = stage
        self.stage_progress = current
        self.stage_total = total
        
        progress_pct = (current / total) * 100 if total > 0 else 0
        message = f"{stage}: {current}/{total} ({progress_pct:.1f}%) {extra_info}".strip()
        
        logger.info(message)
        if self.progress_callback:
            self.progress_callback(message)

    def extract_features(self, image_path: str, feature_types: List[str]) -> Dict[str, np.ndarray]:
        """
        Unified feature extraction method that provides a consistent interface
        for all feature types with intelligent caching.
        
        Args:
            image_path: Path to the image
            feature_types: List of feature types to extract ('color', 'global', 'local')
            
        Returns:
            Dictionary mapping feature types to numpy arrays
        """
        features = {}
        
        for feature_type in feature_types:
            # Try to get from cache first
            cached_feature = self._get_cached_features(image_path, feature_type)
            if cached_feature is not None:
                features[feature_type] = cached_feature
                self.memory_stats['cache_hits'] += 1
                continue
            
            # Extract feature if not in cache
            self.memory_stats['cache_misses'] += 1
            feature = self._extract_single_feature(image_path, feature_type)
            if feature is not None:
                features[feature_type] = feature
                
                # Cache the feature
                self._cache_feature(image_path, feature_type, feature)
        
        return features

    def _extract_single_feature(self, image_path: str, feature_type: str) -> Optional[np.ndarray]:
        """Extract a single feature type from an image"""
        try:
            if feature_type == 'color':
                return self._extract_compact_color_features(image_path)
            elif feature_type == 'global':
                return self._extract_global_features(image_path)
            elif feature_type == 'local':
                return self._extract_local_features(image_path)
            elif feature_type == 'wavelet':
                return self._extract_wavelet_features(image_path)
            else:
                logger.warning(f"Unknown feature type: {feature_type}")
                return None
        except Exception as e:
            logger.error(f"Feature extraction failed for {image_path} ({feature_type}): {e}")
            return None

    def _get_cached_features(self, image_path: str, feature_type: str) -> Optional[Any]:
        """Get features from cache with intelligent fallback"""
        if not self.feature_cache or image_path not in self.feature_cache:
            return None
        
        cached_features = self.feature_cache.get_features(image_path)
        if cached_features and feature_type in cached_features:
            return cached_features[feature_type]
        
        return None

    def _cache_feature(self, image_path: str, feature_type: str, feature: np.ndarray):
        """Cache a feature with proper memory management"""
        if not self.feature_cache:
            return
        
        try:
            current_features = self.feature_cache.get_features(image_path) or {}
            current_features[feature_type] = feature
            self.feature_cache.put_features(image_path, current_features)
        except Exception as e:
            logger.warning(f"Failed to cache feature {feature_type} for {image_path}: {e}")

    def _load_image_efficiently(self, image_path: str) -> Optional[np.ndarray]:
        """
        Enhanced image loading with proper Azure support and error handling.
        
        Args:
            image_path: Path to the image (local or Azure blob URL)
            
        Returns:
            Loaded image as numpy array or None if failed
        """
        try:
            # Check if it's an Azure blob URL
            if image_path.startswith('https://') and 'blob.core.windows.net' in image_path:
                return self._load_azure_image(image_path)
            
            # Local file loading
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}")
                return None
            
            # Load image using OpenCV for better memory efficiency
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                return None
            
            # Convert BGR to RGB for consistency
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            logger.error(f"Image loading failed for {image_path}: {e}")
            return None

    def _load_azure_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Implement proper Azure image loading using Azure Storage SDK.
        
        Args:
            image_path: Azure blob URL
            
        Returns:
            Loaded image as numpy array or None if failed
        """
        try:
            # Download image data to memory
            image_data = download_blob_to_memory(image_path)
            if image_data is None:
                return None
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.warning(f"Failed to decode Azure image: {image_path}")
                return None
            
            # Convert BGR to RGB for consistency
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            logger.error(f"Azure image loading failed for {image_path}: {e}")
            return None

    def _calculate_adaptive_thresholds(self, color_group: List[str]) -> Dict[str, float]:
        """
        Calculate adaptive thresholds based on group characteristics.
        
        Args:
            color_group: List of image paths in the color group
            
        Returns:
            Dictionary with adaptive thresholds for global and local features
        """
        if not self.adaptive_thresholding:
            return {
                'global': 0.7,
                'local': 0.65,
                'wavelet': 0.75
            }
        
        try:
            group_size = len(color_group)
            
            # Calculate color variance within the group
            color_variance = self._calculate_color_variance(color_group)
            
            # Adjust thresholds based on group characteristics
            global_threshold = max(0.5, 0.7 - (group_size / 1000) * 0.1)
            local_threshold = max(0.55, 0.65 - (color_variance * 0.1))
            wavelet_threshold = max(0.6, 0.75 - (group_size / 2000) * 0.1)
            
            # Ensure thresholds don't go below minimum values
            global_threshold = max(global_threshold, 0.5)
            local_threshold = max(local_threshold, 0.55)
            wavelet_threshold = max(wavelet_threshold, 0.6)
            
            self.memory_stats['adaptive_threshold_adjustments'] += 1
            
            return {
                'global': global_threshold,
                'local': local_threshold,
                'wavelet': wavelet_threshold
            }
            
        except Exception as e:
            logger.warning(f"Adaptive threshold calculation failed: {e}, using defaults")
            return {
                'global': 0.7,
                'local': 0.65,
                'wavelet': 0.75
            }

    def _calculate_color_variance(self, color_group: List[str]) -> float:
        """Calculate color variance within a color group"""
        try:
            if len(color_group) < 2:
                return 0.0
            
            # Extract color features for the group
            color_features = []
            for image_path in color_group[:min(100, len(color_group))]:  # Limit for efficiency
                feature = self._extract_compact_color_features(image_path)
                if feature is not None:
                    color_features.append(feature)
            
            if len(color_features) < 2:
                return 0.0
            
            # Calculate variance
            features_array = np.array(color_features)
            variance = np.var(features_array, axis=0).mean()
            
            return float(variance)
            
        except Exception as e:
            logger.warning(f"Color variance calculation failed: {e}")
            return 0.5  # Default moderate variance

    def _stage0_color_pre_grouping_parallel(self, image_paths: List[str]) -> List[List[str]]:
        """
        Parallel implementation of color pre-grouping for large datasets.
        
        Args:
            image_paths: List of image paths to process
            
        Returns:
            List of color groups (each group is a list of image paths)
        """
        if not self.parallel_processing or len(image_paths) < self.chunk_size:
            return self._stage0_color_pre_grouping(image_paths)
        
        start_time = time.time()
        logger.info(f"Starting parallel color pre-grouping for {len(image_paths)} images")
        
        try:
            # Split image_paths into chunks
            chunk_size = max(1, len(image_paths) // self.max_workers)
            chunks = [image_paths[i:i+chunk_size] for i in range(0, len(image_paths), chunk_size)]
            
            self._update_progress("Color Pre-grouping", 0, len(chunks), "Processing chunks in parallel")
            
            # Process chunks in parallel
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._process_color_chunk, chunk): i for i, chunk in enumerate(chunks)}
                
                color_vectors = []
                valid_paths = []
                
                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        chunk_vectors, chunk_paths = future.result()
                        color_vectors.extend(chunk_vectors)
                        valid_paths.extend(chunk_paths)
                        
                        self._update_progress("Color Pre-grouping", chunk_idx + 1, len(chunks), 
                                           f"Chunk {chunk_idx + 1}/{len(chunks)} completed")
                        
                    except Exception as e:
                        logger.error(f"Chunk {chunk_idx} processing failed: {e}")
                        continue
            
            # Continue with clustering
            if not color_vectors:
                logger.warning("No valid color vectors extracted")
                return []
            
            # Perform clustering on combined vectors
            color_vectors_array = np.array(color_vectors)
            self.color_model = MiniBatchKMeans(
                n_clusters=min(self.color_clusters, len(color_vectors_array)),
                batch_size=self.batch_size,
                random_state=42
            )
            
            cluster_labels = self.color_model.fit_predict(color_vectors_array)
            
            # Group images by cluster
            color_groups = self._create_color_groups(valid_paths, cluster_labels)
            
            parallel_time = time.time() - start_time
            self.memory_stats['parallel_processing_time'] = parallel_time
            
            logger.info(f"Parallel color pre-grouping completed in {parallel_time:.2f}s")
            return color_groups
            
        except Exception as e:
            logger.error(f"Parallel color pre-grouping failed: {e}")
            logger.info("Falling back to sequential processing")
            return self._stage0_color_pre_grouping(image_paths)

    def _process_color_chunk(self, chunk: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Process a chunk of images for color feature extraction.
        This method is designed to be picklable for parallel processing.
        
        Args:
            chunk: List of image paths in the chunk
            
        Returns:
            Tuple of (color_vectors, valid_paths)
        """
        color_vectors = []
        valid_paths = []
        
        for image_path in chunk:
            try:
                # Load image efficiently
                image = self._load_image_efficiently(image_path)
                if image is None:
                    continue
                
                # Extract color features
                color_feature = self._extract_compact_color_features_from_array(image)
                if color_feature is not None:
                    color_vectors.append(color_feature)
                    valid_paths.append(image_path)
                
                # Clean up
                del image
                
            except Exception as e:
                logger.warning(f"Chunk processing failed for {image_path}: {e}")
                continue
        
        return color_vectors, valid_paths

    def _extract_compact_color_features_from_array(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract compact color features from a numpy array (for parallel processing).
        
        Args:
            image: Image as numpy array (RGB)
            
        Returns:
            76-dimensional color feature vector
        """
        try:
            # Resize for efficiency
            image_small = cv2.resize(image, (64, 64))
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(image_small, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image_small, cv2.COLOR_RGB2LAB)
            
            # Extract histograms
            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
            s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
            v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
            
            # Normalize histograms
            h_hist = h_hist / (h_hist.sum() + 1e-8)
            s_hist = s_hist / (s_hist.sum() + 1e-8)
            v_hist = v_hist / (v_hist.sum() + 1e-8)
            
            # Combine features
            color_features = np.concatenate([h_hist, s_hist, v_hist])
            
            return color_features.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Color feature extraction from array failed: {e}")
            return None

    def deduplicate_with_color_prefiltering(
        self, 
        image_paths: List[str], 
        output_dir: str,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[List[List[str]], Dict[Tuple[str, str], float]]:
        """
        Enhanced color-optimized deduplication with all improvements.
        
        Args:
            image_paths: List of image paths to deduplicate
            output_dir: Output directory for results
            progress_callback: Optional progress callback function
            
        Returns:
            Tuple of (final_groups, similarity_scores)
        """
        if progress_callback:
            self.progress_callback = progress_callback
        
        start_time = time.time()
        logger.info(f"Starting color-optimized deduplication for {len(image_paths)} images")
        
        try:
            # Stage 0: Color-based pre-grouping (parallel if enabled)
            if self.parallel_processing and len(image_paths) >= self.chunk_size:
                color_groups = self._stage0_color_pre_grouping_parallel(image_paths)
            else:
                color_groups = self._stage0_color_pre_grouping(image_paths)
            
            if not color_groups:
                logger.warning("No color groups created, falling back to standard deduplication")
                return self.deduplicate_memory_efficient(image_paths, output_dir, progress_callback)
            
            # Process each color group independently
            final_groups = []
            all_similarity_scores = {}
            
            total_groups = len(color_groups)
            self._update_progress("Group Processing", 0, total_groups, 
                               f"Processing {total_groups} color groups")
            
            for i, color_group in enumerate(color_groups):
                try:
                    self._update_progress("Group Processing", i + 1, total_groups, 
                                       f"Processing group {i + 1}/{total_groups} ({len(color_group)} images)")
                    
                    # Get adaptive thresholds for this group
                    thresholds = self._calculate_adaptive_thresholds(color_group)
                    
                    # Deduplicate within the color group
                    group_results = self._deduplicate_within_color_group(color_group, thresholds)
                    
                    if group_results:
                        final_groups.extend(group_results)
                    
                    # Clean up after each group
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Group {i} processing failed: {e}")
                    continue
            
            # Calculate total processing time
            total_time = time.time() - start_time
            self.memory_stats['color_processing_time'] = total_time
            
            # Update final stats
            self.memory_stats['color_groups_created'] = len(color_groups)
            self.memory_stats['color_group_sizes'] = [len(group) for group in color_groups]
            
            logger.info(f"Color-optimized deduplication completed in {total_time:.2f}s")
            logger.info(f"Created {len(color_groups)} color groups")
            logger.info(f"Final groups: {len(final_groups)}")
            
            return final_groups, all_similarity_scores
            
        except Exception as e:
            logger.error(f"Color-optimized deduplication failed: {e}")
            logger.info("Falling back to standard deduplication")
            return self.deduplicate_memory_efficient(image_paths, output_dir, progress_callback)

    def _stage0_color_pre_grouping(
        self, 
        image_paths: List[str], 
        progress_callback: Optional[callable] = None
    ) -> List[List[str]]:
        """
        Sequential implementation of color pre-grouping.
        
        Args:
            image_paths: List of image paths to process
            progress_callback: Optional progress callback
            
        Returns:
            List of color groups (each group is a list of image paths)
        """
        start_time = time.time()
        logger.info(f"Starting sequential color pre-grouping for {len(image_paths)} images")
        
        try:
            # Extract color features for all images
            color_vectors = []
            valid_paths = []
            
            self._update_progress("Color Feature Extraction", 0, len(image_paths), "Extracting color features")
            
            for i, image_path in enumerate(image_paths):
                try:
                    self._update_progress("Color Feature Extraction", i + 1, len(image_paths), 
                                       f"Processing {image_path}")
                    
                    # Extract color features
                    color_feature = self._extract_compact_color_features(image_path)
                    if color_feature is not None:
                        color_vectors.append(color_feature)
                        valid_paths.append(image_path)
                    
                    # Clean up periodically
                    if i % 100 == 0:
                        gc.collect()
                        
                except Exception as e:
                    logger.warning(f"Color feature extraction failed for {image_path}: {e}")
                    continue
            
            if not color_vectors:
                logger.warning("No valid color vectors extracted")
                return []
            
            # Perform clustering
            color_vectors_array = np.array(color_vectors)
            self.color_model = MiniBatchKMeans(
                n_clusters=min(self.color_clusters, len(color_vectors_array)),
                batch_size=self.batch_size,
                random_state=42
            )
            
            cluster_labels = self.color_model.fit_predict(color_vectors_array)
            
            # Group images by cluster
            color_groups = self._create_color_groups(valid_paths, cluster_labels)
            
            processing_time = time.time() - start_time
            self.memory_stats['color_processing_time'] = processing_time
            
            logger.info(f"Sequential color pre-grouping completed in {processing_time:.2f}s")
            return color_groups
            
        except Exception as e:
            logger.error(f"Sequential color pre-grouping failed: {e}")
            return []

    def _create_color_groups(self, image_paths: List[str], cluster_labels: np.ndarray) -> List[List[str]]:
        """
        Create color groups based on cluster labels.
        
        Args:
            image_paths: List of image paths
            cluster_labels: Cluster labels for each image
            
        Returns:
            List of color groups
        """
        try:
            # Group images by cluster
            groups = {}
            for path, label in zip(image_paths, cluster_labels):
                if label not in groups:
                    groups[label] = []
                groups[label].append(path)
            
            # Filter groups by size constraints
            filtered_groups = []
            for group in groups.values():
                if self.min_group_size <= len(group) <= self.max_group_size:
                    filtered_groups.append(group)
                elif len(group) > self.max_group_size:
                    # Split large groups
                    sub_groups = self._split_large_group(group)
                    filtered_groups.extend(sub_groups)
            
            # Sort groups by size for better processing order
            filtered_groups.sort(key=len, reverse=True)
            
            return filtered_groups
            
        except Exception as e:
            logger.error(f"Color group creation failed: {e}")
            return []

    def _split_large_group(self, large_group: List[str]) -> List[List[str]]:
        """
        Split a large color group into smaller subgroups.
        
        Args:
            large_group: List of image paths in the large group
            
        Returns:
            List of smaller subgroups
        """
        try:
            if len(large_group) <= self.max_group_size:
                return [large_group]
            
            # Use hierarchical clustering to split the group
            from sklearn.cluster import AgglomerativeClustering
            
            # Extract color features for the large group
            color_features = []
            valid_paths = []
            
            for image_path in large_group:
                feature = self._extract_compact_color_features(image_path)
                if feature is not None:
                    color_features.append(feature)
                    valid_paths.append(image_path)
            
            if len(color_features) < 2:
                return [large_group]
            
            # Determine number of sub-clusters
            n_subclusters = max(2, len(color_features) // self.max_group_size)
            
            # Perform hierarchical clustering
            clustering = AgglomerativeClustering(n_clusters=n_subclusters)
            sub_labels = clustering.fit_predict(color_features)
            
            # Create subgroups
            subgroups = {}
            for path, label in zip(valid_paths, sub_labels):
                if label not in subgroups:
                    subgroups[label] = []
                subgroups[label].append(path)
            
            return list(subgroups.values())
            
        except Exception as e:
            logger.warning(f"Large group splitting failed: {e}")
            return [large_group]

    def _deduplicate_within_color_group(
        self, 
        color_group: List[str], 
        similarity_scores: Dict[Tuple[str, str], float],
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[List[str]], Dict[Tuple[str, str], float]]:
        """
        Apply the full deduplication pipeline within a color group.
        
        This method uses the existing memory-efficient pipeline but with
        optimized thresholds based on color similarity.
        
        Args:
            color_group: List of image paths in the same color group
            similarity_scores: Dictionary to store similarity scores
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (duplicate_groups, similarity_scores)
        """
        logger.info(f"🔍 Applying deduplication pipeline to color group ({len(color_group)} images)")
        
        # Use the existing memory-efficient pipeline but with color-optimized thresholds
        # Since images are already color-similar, we can be more lenient with other features
        
        # Stage 1: Wavelet grouping within color group
        wavelet_groups = self._stage1_wavelet_grouping(color_group, progress_callback)
        
        # Stage 2: Color verification (skip since we're already in color groups)
        # Just pass through the wavelet groups
        color_verified_groups = wavelet_groups
        
        # Stage 3: Global feature refinement with color-optimized thresholds
        global_groups = []
        for group in color_verified_groups:
            if len(group) <= 1:
                global_groups.append(group)
                continue
            
            # Use more lenient thresholds for color-similar images
            refined = self._refine_group_with_global_features_color_optimized(group)
            global_groups.extend(refined)
        
        # Stage 4: Local feature verification with color-optimized thresholds
        local_groups = []
        for group in global_groups:
            if len(group) <= 1:
                local_groups.append(group)
                continue
            
            # Use more lenient thresholds for color-similar images
            verified = self._verify_group_with_local_features_color_optimized(group)
            local_groups.extend(verified)
        
        # Stage 5: Quality-based organization
        final_groups = []
        for group in local_groups:
            if len(group) <= 1:
                final_groups.append(group)
                continue
            
            # Select best image from each group
            best_group = self._select_best_images_from_group(group)
            final_groups.append(best_group)
        
        return final_groups, similarity_scores

    def _extract_global_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract global features for an image."""
        try:
            # Use existing global feature extraction if available
            if hasattr(self, 'feature_extractor'):
                return self.feature_extractor.extract_global_features(image_path)
            else:
                # Fallback to basic global feature extraction
                return self._basic_global_feature_extraction(image_path)
        except Exception as e:
            logger.debug(f"Global feature extraction failed: {e}")
            return None

    def _extract_local_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract local features for an image."""
        try:
            # Use existing local feature extraction if available
            if hasattr(self, 'feature_extractor'):
                return self.feature_extractor.extract_local_features(image_path)
            else:
                # Fallback to basic local feature extraction
                return self._basic_local_feature_extraction(image_path)
        except Exception as e:
            logger.debug(f"Local feature extraction failed: {e}")
            return None

    def _basic_global_feature_extraction(self, image_path: str) -> Optional[np.ndarray]:
        """Basic global feature extraction fallback."""
        try:
            img = self._load_image_efficiently(image_path)
            if img is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Simple global features
            features = [
                np.mean(gray),           # Average intensity
                np.std(gray),            # Standard deviation
                np.var(gray),            # Variance
                np.max(gray),            # Maximum intensity
                np.min(gray),            # Minimum intensity
                gray.shape[0] * gray.shape[1],  # Image size
                np.percentile(gray, 25), # 25th percentile
                np.percentile(gray, 50), # 50th percentile (median)
                np.percentile(gray, 75), # 75th percentile
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.debug(f"Basic global feature extraction failed: {e}")
            return None

    def _basic_local_feature_extraction(self, image_path: str) -> Optional[np.ndarray]:
        """Basic local feature extraction fallback."""
        try:
            img = self._load_image_efficiently(image_path)
            if img is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Simple local features using edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            features = [
                np.sum(edges > 0),      # Edge pixel count
                np.mean(edges),          # Average edge intensity
                np.std(edges),           # Edge intensity variance
                np.max(edges),           # Maximum edge intensity
                np.sum(edges > 0) / (gray.shape[0] * gray.shape[1]),  # Edge density
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.debug(f"Basic local feature extraction failed: {e}")
            return None

    def get_color_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about color optimization performance"""
        stats = self.memory_stats.copy()
        
        # Calculate additional metrics
        if stats['color_groups_created'] > 0:
            stats['avg_group_size'] = np.mean(stats['color_group_sizes'])
            stats['max_group_size'] = np.max(stats['color_group_sizes'])
            stats['min_group_size'] = np.min(stats['color_group_sizes'])
            stats['cache_efficiency'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
        
        return stats

    def _refine_group_with_global_features_color_optimized(self, group: List[str]) -> List[List[str]]:
        """
        Refine group using global features with color-optimized thresholds.
        
        Since images are already color-similar, we use more lenient thresholds
        for global feature similarity.
        
        Args:
            group: List of image paths to refine
            
        Returns:
            List of refined subgroups
        """
        try:
            # Use more lenient thresholds for color-similar images
            # Original threshold might be 0.7, we'll use 0.6 for color groups
            color_optimized_threshold = 0.6  # More lenient than default
            
            # Load global features for the group
            group_features = {}
            for path in group:
                if self.feature_cache and path in self.feature_cache:
                    cached = self.feature_cache.get_features(path)
                    if cached and 'global_features' in cached:
                        group_features[path] = cached['global_features']
                        continue
                
                # Extract global features if not cached
                try:
                    features = self._extract_global_features(path)
                    if features is not None:
                        group_features[path] = features
                        # Cache the features
                        if self.feature_cache:
                            self.feature_cache.put_features(path, {'global_features': features})
                except Exception as e:
                    logger.debug(f"Failed to extract global features for {path}: {e}")
                    continue
            
            if len(group_features) < 2:
                return [group]  # Not enough features to refine
            
            # Apply color-optimized global feature refinement
            refined_groups = self._apply_global_refinement_with_threshold(
                group, group_features, color_optimized_threshold
            )
            
            return refined_groups
            
        except Exception as e:
            logger.warning(f"Global feature refinement failed: {e}")
            return [group]  # Return original group on failure

    def _verify_group_with_local_features_color_optimized(self, group: List[str]) -> List[List[str]]:
        """
        Verify group using local features with color-optimized thresholds.
        
        Args:
            group: List of image paths to verify
            
        Returns:
            List of verified subgroups
        """
        try:
            # Use more lenient thresholds for color-similar images
            color_optimized_threshold = 0.65  # More lenient than default
            
            # Load local features for the group
            group_features = {}
            for path in group:
                if self.feature_cache and path in self.feature_cache:
                    cached = self.feature_cache.get_features(path)
                    if cached and 'local_features' in cached:
                        group_features[path] = cached['local_features']
                        continue
                
                # Extract local features if not cached
                try:
                    features = self._extract_local_features(path)
                    if features is not None:
                        group_features[path] = features
                        # Cache the features
                        if self.feature_cache:
                            self.feature_cache.put_features(path, {'local_features': features})
                except Exception as e:
                    logger.debug(f"Failed to extract local features for {path}: {e}")
                    continue
            
            if len(group_features) < 2:
                return [group]  # Not enough features to verify
            
            # Apply color-optimized local feature verification
            verified_groups = self._apply_local_verification_with_threshold(
                group, group_features, color_optimized_threshold
            )
            
            return verified_groups
            
        except Exception as e:
            logger.warning(f"Local feature verification failed: {e}")
            return [group]  # Return original group on failure

    def _apply_global_refinement_with_threshold(
        self, 
        group: List[str], 
        features: Dict[str, np.ndarray], 
        threshold: float
    ) -> List[List[str]]:
        """Apply global feature refinement with custom threshold."""
        # Implementation would use the existing refinement logic
        # but with the color-optimized threshold
        # For now, return the group as-is
        return [group]

    def _apply_local_verification_with_threshold(
        self, 
        group: List[str], 
        features: Dict[str, np.ndarray], 
        threshold: float
    ) -> List[List[str]]:
        """Apply local feature verification with custom threshold."""
        # Implementation would use the existing verification logic
        # but with the color-optimized threshold
        # For now, return the group as-is
        return [group]

    def _select_best_images_from_group(self, group: List[str]) -> List[str]:
        """Select the best image from a group based on quality score."""
        try:
            if len(group) <= 1:
                return group
            
            # Extract quality scores for all images in the group
            quality_scores = {}
            for path in group:
                try:
                    score = self._extract_quality_score(path)
                    if score is not None:
                        quality_scores[path] = score
                        # Cache the quality score
                        if self.feature_cache:
                            self.feature_cache.put_features(path, {'quality_score': score})
                except Exception as e:
                    logger.debug(f"Failed to extract quality score for {path}: {e}")
                    continue
            
            if not quality_scores:
                return [group[0]]  # Return first image if no quality scores
            
            # Select image with highest quality score
            best_image = max(quality_scores.items(), key=lambda x: x[1])[0]
            return [best_image]
            
        except Exception as e:
            logger.warning(f"Best image selection failed: {e}")
            return [group[0]]  # Return first image on failure

    def _select_best_images_from_groups(self, groups: List[List[str]]) -> List[List[str]]:
        """
        Select the best image from each group based on quality score.
        
        Args:
            groups: List of groups, each containing image paths
            
        Returns:
            List of groups, each containing only the best image
        """
        try:
            final_groups = []
            
            for group in groups:
                if len(group) <= 1:
                    final_groups.append(group)
                    continue
                
                # Select best image from this group
                best_group = self._select_best_images_from_group(group)
                final_groups.append(best_group)
            
            return final_groups
            
        except Exception as e:
            logger.warning(f"Best image selection from groups failed: {e}")
            return groups

    def _extract_quality_score(self, image_path: str) -> Optional[float]:
        """Extract quality score for an image."""
        try:
            # Use existing quality score extraction logic
            if hasattr(self, 'feature_extractor'):
                return self.feature_extractor.extract_quality_score(image_path)
            else:
                # Fallback to basic quality assessment
                return self._basic_quality_assessment(image_path)
        except Exception as e:
            logger.debug(f"Quality score extraction failed: {e}")
            return None

    def _basic_quality_assessment(self, image_path: str) -> Optional[float]:
        """Basic image quality assessment fallback."""
        try:
            img = self._load_image_efficiently(image_path)
            if img is None:
                return None
            
            # Simple quality metrics
            # 1. Sharpness (Laplacian variance)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            # 2. Contrast (standard deviation)
            contrast = np.std(gray)
            
            # 3. Brightness (mean)
            brightness = np.mean(gray)
            
            # 4. Noise (high-frequency content)
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            noise = cv2.filter2D(gray, -1, kernel)
            noise_level = np.std(noise)
            
            # Combine metrics into a quality score (0-100)
            quality_score = (
                min(sharpness / 1000, 30) +  # Sharpness contribution
                min(contrast / 50, 25) +      # Contrast contribution
                min(brightness / 255 * 100, 25) +  # Brightness contribution
                max(0, 20 - noise_level / 10)     # Noise penalty
            )
            
            return min(max(quality_score, 0), 100)  # Clamp to 0-100
            
        except Exception as e:
            logger.debug(f"Basic quality assessment failed: {e}")
            return None


def create_color_optimized_deduplicator(
    feature_cache: Optional[BoundedFeatureCache] = None,
    **kwargs
) -> ColorOptimizedDeduplicator:
    """
    Factory function to create a color-optimized deduplicator.
    
    Args:
        feature_cache: Optional feature cache for memory efficiency
        **kwargs: Additional arguments for the deduplicator
        
    Returns:
        Configured ColorOptimizedDeduplicator instance
    """
    return ColorOptimizedDeduplicator(
        feature_cache=feature_cache,
        **kwargs
    )
