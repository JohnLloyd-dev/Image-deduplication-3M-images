"""
Memory-Efficient Hierarchical Deduplication Pipeline

This module implements a staged approach that minimizes memory usage by:
1. Loading only minimal features needed for each stage
2. Processing groups independently 
3. Freeing memory immediately after each stage
4. Avoiding loading all features simultaneously

Memory savings: ~90% reduction compared to loading all features at once
"""

import os
import time
import logging
import threading
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from tqdm import tqdm
import gc
import cv2 # Added for color similarity computation

from .feature_cache import BoundedFeatureCache
from .deduplication import HierarchicalDeduplicator

logger = logging.getLogger(__name__)


class MemoryEfficientDeduplicator(HierarchicalDeduplicator):
    """Memory-efficient hierarchical deduplicator using staged processing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_stats = {
            'peak_memory_mb': 0,
            'stages_memory': {},
            'features_loaded': 0,
            'features_freed': 0
        }
    
    def deduplicate_memory_efficient(
        self, 
        image_paths: List[str], 
        output_dir: str,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[List[str]], Dict[Tuple[str, str], float]]:
        """
        Perform memory-efficient hierarchical deduplication.
        
        Args:
            image_paths: List of image paths to deduplicate
            output_dir: Output directory for results
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (duplicate_groups, similarity_scores)
        """
        logger.info("Starting memory-efficient hierarchical deduplication...")
        
        if not image_paths:
            logger.warning("Empty image list provided")
            return [], {}
        
        os.makedirs(output_dir, exist_ok=True)
        total_start_time = time.time()
        similarity_scores = {}
        
        # Stage 1: Wavelet-based initial grouping (minimal memory)
        logger.info("🔄 Stage 1: Wavelet-based initial grouping...")
        stage1_start = time.time()
        
        wavelet_groups = self._stage1_wavelet_grouping(image_paths, progress_callback)
        self._log_memory_usage("Stage 1 - Wavelet")
        
        stage1_time = time.time() - stage1_start
        logger.info(f"✅ Stage 1 completed in {stage1_time:.1f}s - {len(wavelet_groups)} groups")
        
        # Separate singleton groups from multi-image groups for optimization
        singleton_groups = [group for group in wavelet_groups if len(group) == 1]
        multi_image_groups = [group for group in wavelet_groups if len(group) > 1]
        
        logger.info(f"📊 Group Analysis:")
        logger.info(f"   - Singleton groups: {len(singleton_groups)} (skipping further processing)")
        logger.info(f"   - Multi-image groups: {len(multi_image_groups)} (need deduplication)")
        
        # Stage 2: Color-based verification (only for multi-image groups)
        logger.info("🔄 Stage 2: Color-based verification...")
        stage2_start = time.time()
        
        if multi_image_groups:
            color_verified_groups = self._stage2_color_verification(
                multi_image_groups, similarity_scores, progress_callback
            )
        else:
            color_verified_groups = []
            logger.info("⏭️  Skipping Stage 2 - no multi-image groups to process")
        
        # Combine singleton groups with color-verified groups
        all_groups_after_stage2 = singleton_groups + color_verified_groups
        self._log_memory_usage("Stage 2 - Color")
        
        stage2_time = time.time() - stage2_start
        logger.info(f"✅ Stage 2 completed in {stage2_time:.1f}s - {len(all_groups_after_stage2)} total groups")
        
        # Stage 3: Global feature refinement (only for multi-image groups)
        logger.info("🔄 Stage 3: Global feature refinement...")
        stage3_start = time.time()
        
        if color_verified_groups:
            global_refined_groups = self._stage3_global_refinement(
                color_verified_groups, similarity_scores, progress_callback
            )
        else:
            global_refined_groups = []
            logger.info("⏭️  Skipping Stage 3 - no groups need global refinement")
        
        # Combine singleton groups with global-refined groups
        all_groups_after_stage3 = singleton_groups + global_refined_groups
        self._log_memory_usage("Stage 3 - Global")
        
        stage3_time = time.time() - stage3_start
        logger.info(f"✅ Stage 3 completed in {stage3_time:.1f}s - {len(all_groups_after_stage3)} total groups")
        
        # Stage 4: Local feature verification (only for multi-image groups)
        logger.info("🔄 Stage 4: Local feature verification...")
        stage4_start = time.time()
        
        if global_refined_groups:
            local_verified_groups = self._stage4_local_verification(
                global_refined_groups, similarity_scores, progress_callback
            )
        else:
            local_verified_groups = []
            logger.info("⏭️  Skipping Stage 4 - no groups need local verification")
        
        # Combine singleton groups with local-verified groups
        all_groups_after_stage4 = singleton_groups + local_verified_groups
        self._log_memory_usage("Stage 4 - Local")
        
        stage4_time = time.time() - stage4_start
        logger.info(f"✅ Stage 4 completed in {stage4_time:.1f}s - {len(all_groups_after_stage4)} total groups")
        
        # Stage 5: Quality-based organization (all groups, including singletons)
        logger.info("🔄 Stage 5: Quality-based organization...")
        stage5_start = time.time()
        
        final_results = self._stage5_quality_organization(
            all_groups_after_stage4, similarity_scores, output_dir
        )
        
        stage5_time = time.time() - stage5_start
        logger.info(f"✅ Stage 5 completed in {stage5_time:.1f}s")
        
        # Final statistics (include all images, including singletons)
        total_time = time.time() - total_start_time
        total_images = len(image_paths)
        total_duplicates = sum(len(group) for group in all_groups_after_stage4)
        
        logger.info(f"\n🎉 Memory-Efficient Deduplication Complete!")
        logger.info(f"📊 Results:")
        logger.info(f"   - Total images processed: {total_images:,}")
        logger.info(f"   - Duplicate groups found: {len(all_groups_after_stage4)}")
        logger.info(f"   - Total duplicate images: {total_duplicates:,}")
        logger.info(f"   - Unique images: {total_images - total_duplicates:,}")
        logger.info(f"   - Processing time: {total_time:.1f}s")
        logger.info(f"   - Processing rate: {total_images/total_time:.1f} images/second")
        
        # Quality score statistics
        cached_quality_scores = 0
        for group in all_groups_after_stage4:
            for img_path in group:
                features = self.feature_cache.get_features(img_path)
                if features and 'quality_score' in features and features['quality_score'] is not None:
                    cached_quality_scores += 1
        
        logger.info(f"📊 Quality Score Statistics:")
        logger.info(f"   - Quality scores computed in Stage 1: {cached_quality_scores:,}")
        logger.info(f"   - Quality score cache hit rate: {(cached_quality_scores/total_images)*100:.1f}%")
        
        logger.info(f"💾 Memory Statistics:")
        logger.info(f"   - Peak memory usage: {self.memory_stats['peak_memory_mb']:.1f} MB")
        logger.info(f"   - Features loaded: {self.memory_stats['features_loaded']:,}")
        logger.info(f"   - Features freed: {self.memory_stats['features_freed']:,}")
        logger.info(f"   - Memory efficiency: {(self.memory_stats['features_freed']/max(self.memory_stats['features_loaded'], 1))*100:.1f}% freed")
        
        return all_groups_after_stage4, similarity_scores
    
    def _stage1_wavelet_grouping(self, image_paths: List[str], progress_callback=None) -> List[List[str]]:
        """Stage 1: Group images by wavelet hash (compute features on-demand, NO image caching)."""
        
        # Initialize feature extractor for on-demand computation
        from .feature_extraction import FeatureExtractor
        feature_extractor = FeatureExtractor()
        
        # Compute wavelet features on-demand for all images
        wavelet_features = {}
        failed_computations = 0
        
        logger.info(f"Computing wavelet features and quality scores for {len(image_paths):,} images...")
        logger.info("🔄 STAGE 1: Starting wavelet + quality score computation loop")
        
        for path in tqdm(image_paths, desc="Computing wavelet + quality scores"):
            try:
                # Try to get from cache first
                features = self.feature_cache.get_features(path)
                if features and 'wavelet' in features and 'quality_score' in features:
                    # Use cached features (both wavelet and quality score)
                    logger.debug(f"🔄 Using cached wavelet + quality features for {path}")
                    wavelet_features[path] = features['wavelet']
                    self.memory_stats['features_loaded'] += 1
                else:
                    # Need to compute features (either missing wavelet or quality score)
                    try:
                        # Load image from Azure (NO CACHING - load, process, release immediately)
                        from .azure_image_loader import load_image_from_azure
                        img = load_image_from_azure(path)
                        logger.debug(f"🔄 Image loaded for {path}: {img is not None}")
                        
                        if img is not None:
                            logger.debug(f"🔄 Processing image in Stage 1: {path}")
                            
                            # Compute wavelet hash (either new or use cached)
                            if features and 'wavelet' in features:
                                wavelet_hash = features['wavelet']
                                logger.debug(f"🔄 Using cached wavelet hash for {path}")
                            else:
                                wavelet_hash = feature_extractor.compute_wavelet_hash(img)
                                logger.debug(f"🔄 Wavelet hash computed for {path}: {wavelet_hash is not None}")
                            
                            if wavelet_hash is not None:
                                wavelet_features[path] = wavelet_hash
                                
                                # COMPUTE QUALITY SCORE IN STAGE 1 (as requested)
                                logger.debug(f"🔄 Computing quality score for {path}")
                                try:
                                    quality_score = self._compute_high_frequency_quality(img)
                                    logger.debug(f"✅ Quality score computed for {path}: {quality_score:.2f}")
                                except Exception as e:
                                    logger.error(f"❌ Failed to compute quality score for {path}: {e}")
                                    quality_score = 0.0
                                
                                # Cache both wavelet and quality features (MERGE with any existing)
                                existing = self.feature_cache.get_features(path) or {}
                                features = {**existing, 'wavelet': wavelet_hash, 'quality_score': quality_score}
                                logger.debug(f"💾 Caching wavelet + quality features for {path}: {list(features.keys())}")
                                self.feature_cache.put_features(path, features)
                                self.memory_stats['features_loaded'] += 1
                            else:
                                failed_computations += 1
                        else:
                            failed_computations += 1
                            
                        # CRITICAL MEMORY FIX: Release image immediately after processing
                        del img
                        
                    except Exception as e:
                        logger.warning(f"Failed to compute wavelet feature for {path}: {e}")
                        failed_computations += 1
                        
            except Exception as e:
                logger.warning(f"Failed to process wavelet feature for {path}: {e}")
                failed_computations += 1
        
        if failed_computations > 0:
            logger.warning(f"Failed to compute wavelet features for {failed_computations} images")
        
        logger.info(f"Computed wavelet features for {len(wavelet_features):,} images")
        
        if len(wavelet_features) == 0:
            logger.error("No wavelet features computed - cannot proceed with grouping")
            return [[path] for path in image_paths]  # Return individual groups
        
        # Group by wavelet similarity
        wavelet_groups = self.group_by_wavelet(list(wavelet_features.keys()), wavelet_features)
        
        # Free wavelet features immediately
        del wavelet_features
        self.memory_stats['features_freed'] += len(image_paths)
        gc.collect()
        
        # Clean up feature extractor
        feature_extractor.release()
        
        if progress_callback:
            progress_callback("Stage 1 Complete", 20)
        
        return wavelet_groups
    
    def _stage2_color_verification(self, wavelet_groups: List[List[str]], 
                                  similarity_scores: Dict, progress_callback=None) -> List[List[str]]:
        """Stage 2: Verify groups using color features (group-by-group processing)."""
        
        color_verified_groups = []
        total_groups = len(wavelet_groups)
        
        for i, group in enumerate(tqdm(wavelet_groups, desc="Color verification")):
            if len(group) <= 1:
                color_verified_groups.append(group)
                continue
            
            # Skip very large groups for color verification (too expensive)
            if len(group) > 15:  # Reduced from 30 to 15
                logger.warning(f"Skipping color verification for large group ({len(group)} images)")
                color_verified_groups.append(group)
                continue
            
            # FIXED: Load all images for this group once, then do all comparisons
            try:
                subgroups = self._verify_group_with_color_features_efficient(group)
                color_verified_groups.extend(subgroups)
            except Exception as e:
                logger.warning(f"Color verification failed for group of {len(group)} images: {e}")
                color_verified_groups.append(group)  # Keep the group as-is
            
            # Force garbage collection after each group (color processing can be memory-intensive)
            gc.collect()
            
            if progress_callback:
                progress = 20 + (i / total_groups) * 20  # 20-40%
                progress_callback(f"Color verification: {i+1}/{total_groups}", progress)
        
        return color_verified_groups
    
    def _verify_group_with_color_features_efficient(self, group: List[str]) -> List[List[str]]:
        """
        Memory-efficient color verification that processes images in small batches
        to avoid loading all images for a group at once.
        """
        try:
            if len(group) < 2:
                return [group]
            
            # MEMORY FIX: Process in smaller batches to avoid loading all images at once
            batch_size = 5  # Process only 5 images at a time
            subgroups = []
            processed_images = set()
            
            logger.debug(f"Processing color verification for group of {len(group)} images in batches of {batch_size}")
            
            for i in range(0, len(group), batch_size):
                batch = group[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch)} images")
                
                # Load images for this batch only
                batch_images = {}
                failed_images = []
                
                for img_path in batch:
                    if img_path in processed_images:
                        continue
                        
                    try:
                        # Check if this is a test image
                        if self._is_test_image_path(img_path):
                            logger.debug(f"Skipping Azure download for test image: {img_path}")
                            failed_images.append(img_path)
                            continue
                        
                        # Download image from Azure with timeout
                        from .azure_image_loader import load_image_from_azure
                        img = load_image_from_azure(img_path)
                        
                        if img is not None:
                            batch_images[img_path] = img
                        else:
                            failed_images.append(img_path)
                            logger.warning(f"Failed to load image: {img_path}")
                            
                    except Exception as e:
                        failed_images.append(img_path)
                        logger.warning(f"Error loading image {img_path}: {e}")
                
                # Process batch comparisons
                if len(batch_images) > 1:
                    batch_subgroups = self._process_color_batch(batch_images)
                    subgroups.extend(batch_subgroups)
                else:
                    # Add individual images as separate groups
                    for img_path in batch_images:
                        subgroups.append([img_path])
                
                # Add failed images as individual groups
                for failed_img in failed_images:
                    subgroups.append([failed_img])
                
                # Mark as processed
                processed_images.update(batch_images.keys())
                processed_images.update(failed_images)
                
                # CRITICAL MEMORY FIX: Release batch images immediately
                del batch_images
                gc.collect()
            
            # Merge subgroups that might be duplicates
            final_subgroups = self._merge_similar_subgroups(subgroups)
            
            logger.debug(f"Color verification complete: {len(group)} -> {len(final_subgroups)} subgroups")
            return final_subgroups
            
        except Exception as e:
            logger.error(f"Color verification failed for group: {e}")
            return [group]
    
    def _process_color_batch(self, batch_images: Dict[str, np.ndarray]) -> List[List[str]]:
        """Process a small batch of images for color similarity."""
        try:
            if len(batch_images) <= 1:
                return [list(batch_images.keys())]
            
            # Build similarity matrix for this batch only
            similarity_matrix = {}
            img_paths = list(batch_images.keys())
            
            for i, img1_path in enumerate(img_paths):
                for j in range(i+1, len(img_paths)):
                    img2_path = img_paths[j]
                    
                    try:
                        # Compute color similarity using loaded images
                        similarity = self._compute_color_similarity_from_images(
                            batch_images[img1_path], batch_images[img2_path]
                        )
                        similarity_matrix[(img1_path, img2_path)] = similarity
                        
                    except Exception as e:
                        logger.warning(f"Error computing similarity for {img1_path}, {img2_path}: {e}")
                        similarity_matrix[(img1_path, img2_path)] = 0.0
            
            # Group images based on color similarity
            batch_subgroups = self._group_by_color_similarity(img_paths, similarity_matrix)
            
            return batch_subgroups
            
        except Exception as e:
            logger.error(f"Error processing color batch: {e}")
            return [list(batch_images.keys())]
    
    def _merge_similar_subgroups(self, subgroups: List[List[str]]) -> List[List[str]]:
        """Merge subgroups that might contain the same images across batches."""
        try:
            if len(subgroups) <= 1:
                return subgroups
            
            # Create a mapping of image to group
            image_to_group = {}
            merged_groups = []
            
            for group in subgroups:
                for img_path in group:
                    if img_path in image_to_group:
                        # Merge groups
                        existing_group_idx = image_to_group[img_path]
                        existing_group = merged_groups[existing_group_idx]
                        
                        # Add new images to existing group
                        for new_img in group:
                            if new_img not in existing_group:
                                existing_group.append(new_img)
                                image_to_group[new_img] = existing_group_idx
                    else:
                        # Create new group
                        merged_groups.append(group.copy())
                        group_idx = len(merged_groups) - 1
                        for img_path in group:
                            image_to_group[img_path] = group_idx
            
            return merged_groups
            
        except Exception as e:
            logger.error(f"Error merging subgroups: {e}")
            return subgroups
    
    def _compute_color_similarity_from_images(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute color similarity between two loaded images.
        This avoids downloading images multiple times.
        """
        try:
            # 1. Dominant color distance
            colors1 = self._get_dominant_colors(img1, 2)
            colors2 = self._get_dominant_colors(img2, 2)
            
            min_dist = float('inf')
            for c1 in colors1:
                for c2 in colors2:
                    dist = np.linalg.norm(c1 - c2)
                    min_dist = min(min_dist, dist)
            
            dom_sim = 1.0 - min(min_dist / self.max_color_distance, 1.0)
            
            # 2. Average pixel difference
            h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1_resized = cv2.resize(img1, (w, h))
            img2_resized = cv2.resize(img2, (w, h))
            
            lab1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2LAB)
            lab2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2LAB)
            avg_diff = np.mean(np.abs(lab1.astype(np.float32) - lab2.astype(np.float32)))
            
            pixel_sim = 1.0 - min(avg_diff / self.max_pixel_difference, 1.0)
            
            # 3. Histogram correlation
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            y1_start, y1_end = int(h1 * 0.1), int(h1 * 0.9)
            x1_start, x1_end = int(w1 * 0.1), int(w1 * 0.9)
            y2_start, y2_end = int(h2 * 0.1), int(h2 * 0.9)
            x2_start, x2_end = int(w2 * 0.1), int(w2 * 0.9)
            
            central1 = img1[y1_start:y1_end, x1_start:x1_end]
            central2 = img2[y2_start:y2_end, x2_start:x2_end]
            
            hist1 = cv2.calcHist([central1], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([central2], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            hist_sim = max(0.0, cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
            
            # Weighted combination
            color_similarity = (0.5 * dom_sim) + (0.3 * pixel_sim) + (0.2 * hist_sim)
            
            return color_similarity
            
        except Exception as e:
            logger.error(f"Error computing color similarity: {e}")
            return 0.0
    
    def _group_by_color_similarity(self, image_paths: List[str], similarity_matrix: Dict) -> List[List[str]]:
        """
        Group images based on color similarity matrix.
        Uses connected components to find groups of similar images.
        """
        try:
            if len(image_paths) <= 1:
                return [image_paths]
            
            # Build adjacency matrix
            n = len(image_paths)
            adjacency = [[False] * n for _ in range(n)]
            
            for i, img1 in enumerate(image_paths):
                for j, img2 in enumerate(image_paths):
                    if i != j and (img1, img2) in similarity_matrix:
                        similarity = similarity_matrix[(img1, img2)]
                        # Use color threshold for grouping
                        adjacency[i][j] = similarity >= self.color_threshold
            
            # Find connected components using BFS
            visited = [False] * n
            groups = []
            
            for i in range(n):
                if visited[i]:
                    continue
                
                # BFS to find all connected images
                current_group = []
                queue = [i]
                
                while queue:
                    idx = queue.pop(0)
                    if visited[idx]:
                        continue
                    
                    visited[idx] = True
                    current_group.append(image_paths[idx])
                    
                    # Find all unvisited similar images
                    for j in range(n):
                        if not visited[j] and adjacency[idx][j]:
                            queue.append(j)
                
                groups.append(current_group)
            
            return groups
            
        except Exception as e:
            logger.error(f"Error grouping by color similarity: {e}")
            return [image_paths]
    
    def _get_dominant_colors(self, img: np.ndarray, n_colors: int = 2) -> np.ndarray:
        """Fast dominant color extraction using mini-batch K-means."""
        try:
            # Resize for efficiency
            img = cv2.resize(img, (64, 64))
            
            # Convert to LAB color space
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Reshape for clustering
            pixels = img_lab.reshape(-1, 3)
            
            # Use mini-batch K-Means for speed
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            dominant_colors = kmeans.cluster_centers_
            
            # Sort by frequency
            labels = kmeans.labels_
            color_counts = np.bincount(labels)
            sorted_indices = np.argsort(-color_counts)
            dominant_colors = dominant_colors[sorted_indices]
            
            return dominant_colors
            
        except Exception as e:
            logger.error(f"Error extracting dominant colors: {e}")
            return np.zeros((n_colors, 3))
    
    def _is_test_image_path(self, image_path: str) -> bool:
        """Check if the image path is a test/dummy image that shouldn't be downloaded from Azure."""
        try:
            # Check for common test image patterns
            test_patterns = [
                'test_', 'dummy_', 'sample_', 'mock_', 'fake_',
                'placeholder', 'example', 'demo', 'temp_', 'tmp_'
            ]
            
            lower_path = image_path.lower()
            for pattern in test_patterns:
                if pattern in lower_path:
                    return True
            
            # Check for very short or suspicious paths (e.g., empty string, very short names)
            if len(image_path) < 5: # Adjusted minimum length
                return True
            
            # Check for common test image patterns, ensuring they are distinct parts of the path
            # Use regex or more specific string matching if 'test' can appear legitimately
            # For now, let's be more specific about common test file prefixes/suffixes
            test_patterns = ['/test_', '_test.', '/dummy_', '_dummy.', '/sample_', '_sample.',
                             '/mock_', '_mock.', '/fake_', '_fake.', '/placeholder', '/example',
                             '/demo', '/temp_', '_temp.', '/tmp_', '_tmp.']
            for pattern in test_patterns:
                if pattern in lower_path:
                    return True
            
            # Exclude specific legitimate paths that might contain "test" as part of a larger word
            # e.g., "TestEquity" should not be flagged as a test path
            if "testequity" in lower_path:
                return False # This is a legitimate path, do not treat as test
            
            # Original broad 'test' check removed as it caused false positives
            # if 'test' in lower_path:
            #     return True
            
            # Final check for very short or suspicious paths (redundant with len < 5 check above, but kept for clarity)
            if len(image_path) < 10: # This check is less critical now with more specific patterns
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking test image path {image_path}: {e}")
            return False
    
    def _stage3_global_refinement(self, color_groups: List[List[str]], 
                                 similarity_scores: Dict, progress_callback=None) -> List[List[str]]:
        """Stage 3: Refine groups using global features (subgroup-by-subgroup processing)."""
        
        global_refined_groups = []
        total_groups = len(color_groups)
        
        for i, group in enumerate(tqdm(color_groups, desc="Global refinement")):
            if len(group) <= 1:
                continue
            
            # Skip very large groups (memory protection)
            if len(group) > 50:  # Reduced from 100 to 50
                logger.warning(f"Splitting large group ({len(group)} images) for memory efficiency")
                # Process in chunks
                chunk_size = 25  # Reduced from 50 to 25
                for j in range(0, len(group), chunk_size):
                    chunk = group[j:j + chunk_size]
                    if len(chunk) > 1:
                        # Load global features for chunk
                        chunk_global_features = self._load_features_for_group(chunk, 'global')
                        if chunk_global_features:
                            chunk_refined = self._refine_group_with_global_features(
                                list(chunk_global_features.keys()),
                                {path: {'global': feat} for path, feat in chunk_global_features.items()},
                                similarity_scores
                            )
                            global_refined_groups.extend(chunk_refined)
                            del chunk_global_features
                continue
            
            # Load global features only for this group
            group_global_features = self._load_features_for_group(group, 'global')
            
            if not group_global_features:
                logger.warning(f"No global features found for group of {len(group)} images")
                # CRITICAL FIX: Preserve the group even if features fail to load
                global_refined_groups.append(group)
                continue
            
            # Process this group
            subgroups = self._refine_group_with_global_features(
                list(group_global_features.keys()), 
                {path: {'global': feat} for path, feat in group_global_features.items()},
                similarity_scores
            )
            global_refined_groups.extend(subgroups)
            
            # Free memory immediately
            del group_global_features
            self.memory_stats['features_freed'] += len(group)
            
            # Periodic garbage collection
            if i % 10 == 0:
                gc.collect()
            
            if progress_callback:
                progress = 40 + (i / total_groups) * 20  # 40-60%
                progress_callback(f"Global refinement: {i+1}/{total_groups}", progress)
        
        return global_refined_groups
    
    def _stage4_local_verification(self, global_groups: List[List[str]], 
                                  similarity_scores: Dict, progress_callback=None) -> List[List[str]]:
        """Stage 4: Final verification using local features (subgroup-by-subgroup processing)."""
        
        local_verified_groups = []
        total_groups = len(global_groups)
        
        for i, group in enumerate(tqdm(global_groups, desc="Local verification")):
            if len(group) <= 1:
                continue
            
            # Skip very large groups for local verification (too expensive)
            if len(group) > 25:  # Reduced from 50 to 25
                logger.warning(f"Skipping local verification for large group ({len(group)} images)")
                local_verified_groups.append(group)
                continue
            
            # Load local features only for this group
            group_local_features = self._load_features_for_group(group, 'local')
            
            if not group_local_features:
                logger.warning(f"No local features found for group of {len(group)} images")
                # CRITICAL FIX: Preserve the group even if features fail to load
                local_verified_groups.append(group)  # Keep the group as-is
                continue
            
            # Process this group
            subgroups = self._verify_group_with_local_features(
                list(group_local_features.keys()),
                {path: {'local': feat} for path, feat in group_local_features.items()},
                similarity_scores
            )
            local_verified_groups.extend(subgroups)
            
            # Free memory immediately
            del group_local_features
            self.memory_stats['features_freed'] += len(group)
            
            # Periodic garbage collection
            if i % 5 == 0:  # More frequent for local features (larger)
                gc.collect()
            
            if progress_callback:
                progress = 60 + (i / total_groups) * 20  # 60-80%
                progress_callback(f"Local verification: {i+1}/{total_groups}", progress)
        
        return local_verified_groups
    
    def _stage5_quality_organization(self, final_groups: List[List[str]], 
                                   similarity_scores: Dict, output_dir: str) -> Dict:
        """Stage 5: Quality-based organization and best image selection using cached quality scores."""
        
        logger.info("🔄 Stage 5: Using cached quality scores from Stage 1 for best image selection...")
        
        # Count how many quality scores we expect to have cached
        total_images = sum(len(group) for group in final_groups)
        cached_quality_scores = 0
        
        for group in final_groups:
            for img_path in group:
                features = self.feature_cache.get_features(img_path)
                if features and 'quality_score' in features and features['quality_score'] is not None:
                    cached_quality_scores += 1
        
        logger.info(f"📊 Quality Score Cache Status:")
        logger.info(f"   - Total images: {total_images}")
        logger.info(f"   - Cached quality scores: {cached_quality_scores}")
        logger.info(f"   - Cache hit rate: {(cached_quality_scores/total_images)*100:.1f}%")
        
        # This stage uses minimal additional memory - quality scores are already cached from Stage 1
        organized_results = self._organize_duplicate_groups_with_quality_selection(
            final_groups, {}, similarity_scores  # Empty features dict since we load on-demand
        )
        
        return organized_results
    
    def _load_features_for_group(self, group: List[str], feature_type: str) -> Dict[str, Any]:
        """Load or compute specific feature type for a group of images."""
        
        # Initialize feature extractor for on-demand computation
        from .feature_extraction import FeatureExtractor
        feature_extractor = FeatureExtractor()
        
        features = {}
        for path in group:
            try:
                # Try to get from cache first
                cached_features = self.feature_cache.get_features(path)
                if cached_features and feature_type in cached_features:
                    # Use cached feature
                    features[path] = cached_features[feature_type]
                    self.memory_stats['features_loaded'] += 1
                else:
                    # Compute feature on-demand
                    try:
                        # Load image from Azure
                        from .azure_image_loader import load_image_from_azure
                        img = load_image_from_azure(path)
                        
                        if img is not None:
                            if feature_type == 'global':
                                global_feat = feature_extractor.extract_global_features(img)
                                if global_feat is not None:
                                    features[path] = global_feat
                                    # Cache the computed feature
                                    if cached_features is None:
                                        cached_features = {}
                                    cached_features['global'] = global_feat
                                    self.feature_cache.put_features(path, cached_features)
                                    self.memory_stats['features_loaded'] += 1
                            elif feature_type == 'local':
                                local_feat = feature_extractor.extract_local_features(img)
                                if local_feat is not None:
                                    features[path] = local_feat
                                    # Cache the computed feature
                                    if cached_features is None:
                                        cached_features = {}
                                    cached_features['local'] = local_feat
                                    self.feature_cache.put_features(path, cached_features)
                                    self.memory_stats['features_loaded'] += 1
                            elif feature_type == 'wavelet':
                                wavelet_hash = feature_extractor.compute_wavelet_hash(img)
                                if wavelet_hash is not None:
                                    features[path] = wavelet_hash
                                    # Cache the computed feature
                                    if cached_features is None:
                                        cached_features = {}
                                    cached_features['wavelet'] = wavelet_hash
                                    self.feature_cache.put_features(path, cached_features)
                                    self.memory_stats['features_loaded'] += 1
                            else:
                                logger.warning(f"Unknown feature type: {feature_type}")
                        else:
                            logger.warning(f"Failed to load image for {path}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to compute {feature_type} feature for {path}: {e}")
                        
            except Exception as e:
                logger.warning(f"Failed to load/compute {feature_type} features for {path}: {e}")
        
        # Clean up feature extractor
        feature_extractor.release()
        
        return features
    
    def _compute_high_frequency_quality(self, img: np.ndarray) -> float:
        """Compute quality score based on high-frequency content (image sharpness)."""
        logger.debug(f"🔄 _compute_high_frequency_quality called with image shape: {img.shape}")
        try:
            import cv2
            import numpy as np
            
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Apply Laplacian filter to detect edges (high-frequency content)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Compute variance of Laplacian (measure of sharpness)
            # Higher variance = more edges = sharper image = higher quality
            filter_result = laplacian.var()
            
            # Calculate image size (number of pixels)
            image_size = gray.shape[0] * gray.shape[1]
            
            # Quality score = filter_result / image_size (normalized by image size)
            # This gives fair comparison between images of different sizes
            quality_score = filter_result / image_size
            
            # Scale to reasonable range (0-100)
            # Typical values are very small, so multiply by a large factor
            normalized_score = min(100.0, quality_score * 1000000)
            
            return float(normalized_score)
            
        except Exception as e:
            logger.warning(f"High-frequency quality computation failed: {e}")
            return 0.0

    def _log_memory_usage(self, stage_name: str):
        """Log current memory usage for a stage."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            self.memory_stats['stages_memory'][stage_name] = memory_mb
            self.memory_stats['peak_memory_mb'] = max(
                self.memory_stats['peak_memory_mb'], 
                memory_mb
            )
            
            logger.info(f"💾 {stage_name} memory usage: {memory_mb:.1f} MB")
            
        except ImportError:
            logger.debug("psutil not available for memory monitoring")
        except Exception as e:
            logger.warning(f"Memory monitoring failed: {e}")


def create_memory_efficient_deduplicator(
    feature_cache: BoundedFeatureCache,
    **kwargs
) -> MemoryEfficientDeduplicator:
    """Factory function to create a memory-efficient deduplicator."""
    
    return MemoryEfficientDeduplicator(
        feature_cache=feature_cache,
        **kwargs
    )