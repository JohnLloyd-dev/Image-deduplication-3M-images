#!/usr/bin/env python3
"""
Wavelet Hash (WHash) Deduplicator for Large-Scale Image Processing

This module implements WHash deduplication that integrates seamlessly with the
existing color-optimized pipeline. WHash provides fast first-pass grouping
using wavelet transforms, significantly reducing the problem size before
applying more expensive color-based deduplication.

Key Features:
- Fast wavelet hash computation using PyWavelets
- Locality-Sensitive Hashing (LSH) for efficient grouping
- Seamless integration with ColorOptimizedDeduplicator
- Configurable hash size, wavelet levels, and similarity thresholds
- Memory-efficient processing for 3M+ images
"""

import os
import logging
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import cv2
from tqdm import tqdm

# Try to import PyWavelets, fall back to basic implementation if not available
try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    logging.warning("PyWavelets not available. Using basic hash implementation.")

logger = logging.getLogger(__name__)

class WHashDeduplicator:
    """
    Wavelet Hash (WHash) deduplication implementation for large-scale image processing.
    Integrates with the existing color-optimized pipeline for optimal performance.
    """
    
    def __init__(self, 
                 hash_size: int = 8, 
                 wavelet_level: int = 1, 
                 threshold: float = 0.85,
                 wavelet_name: str = 'haar',
                 lsh_bands: int = 4,
                 lsh_rows_per_band: int = 4,
                 enable_lsh: bool = True):
        """
        Initialize WHash deduplicator.
        
        Args:
            hash_size: Size of the hash (default 8x8 = 64 bits)
            wavelet_level: Number of wavelet decomposition levels
            threshold: Similarity threshold for considering images as duplicates
            wavelet_name: Wavelet type to use ('haar', 'db1', 'db2', etc.)
            lsh_bands: Number of bands for Locality-Sensitive Hashing
            lsh_rows_per_band: Number of rows per band for LSH
            enable_lsh: Whether to use LSH for efficient grouping
        """
        self.hash_size = hash_size
        self.wavelet_level = wavelet_level
        self.threshold = threshold
        self.wavelet_name = wavelet_name
        self.lsh_bands = lsh_bands
        self.lsh_rows_per_band = lsh_rows_per_band
        self.enable_lsh = enable_lsh and PYWAVELETS_AVAILABLE
        
        # Validate parameters
        if self.hash_size < 4:
            raise ValueError("Hash size must be at least 4x4")
        if self.wavelet_level < 1:
            raise ValueError("Wavelet level must be at least 1")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        # LSH validation
        if self.enable_lsh:
            required_bits = self.lsh_bands * self.lsh_rows_per_band
            if required_bits > self.hash_size * self.hash_size:
                logger.warning(f"LSH requires {required_bits} bits but hash provides {self.hash_size * self.hash_size} bits")
                self.enable_lsh = False
        
        logger.info(f"WHash Deduplicator initialized: hash_size={self.hash_size}, "
                   f"wavelet_level={self.wavelet_level}, threshold={self.threshold}")
        
        # Performance tracking
        self.stats = {
            'images_processed': 0,
            'hashes_computed': 0,
            'groups_created': 0,
            'processing_time': 0.0,
            'memory_usage_mb': 0.0
        }
    
    def compute_whash(self, image_path: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Compute wavelet hash for an image.
        
        Args:
            image_path: Path to image file, Azure blob reference, or numpy array
            
        Returns:
            whash: Wavelet hash as a flattened numpy array, or None if failed
        """
        try:
            # Load image if path provided
            if isinstance(image_path, str):
                img = self._load_image_efficiently(image_path)
            else:
                img = image_path
                
            if img is None:
                return None
                
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                img = self._rgb_to_grayscale(img)
            
            # Ensure image dimensions are appropriate for wavelet transform
            img = self._prepare_image_for_wavelet(img)
            
            if img is None:
                return None
            
            # Compute wavelet hash
            if PYWAVELETS_AVAILABLE:
                whash = self._compute_wavelet_hash(img)
            else:
                whash = self._compute_basic_hash(img)
            
            if whash is not None:
                self.stats['hashes_computed'] += 1
            
            return whash
            
        except Exception as e:
            logger.error(f"WHash computation failed for {image_path}: {e}")
            return None
    
    def _load_image_efficiently(self, image_path: str) -> Optional[np.ndarray]:
        """Load image efficiently, handling both local and Azure paths."""
        try:
            # Check if it's an Azure URL
            if image_path.startswith(('http://', 'https://')):
                # For Azure URLs, we'll need to implement download logic
                # For now, return None to indicate we need Azure integration
                logger.warning(f"Azure URL detected: {image_path}. Azure integration needed.")
                return None
            
            # Local file loading
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def _rgb_to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale using luminance weights."""
        # Use standard luminance weights
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    
    def _prepare_image_for_wavelet(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Prepare image for wavelet transform by ensuring proper dimensions."""
        try:
            h, w = img.shape
            
            # Find the closest dimensions that are divisible by 2^wavelet_level
            min_dim = 2 ** self.wavelet_level
            new_h = (h // min_dim) * min_dim
            new_w = (w // min_dim) * min_dim
            
            # Ensure minimum size for hash
            if new_h < self.hash_size or new_w < self.hash_size:
                # Resize to minimum required size
                new_h = max(new_h, self.hash_size)
                new_w = max(new_w, self.hash_size)
            
            if new_h != h or new_w != w:
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            return img
            
        except Exception as e:
            logger.error(f"Image preparation failed: {e}")
            return None
    
    def _compute_wavelet_hash(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Compute wavelet hash using PyWavelets."""
        try:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec2(img, self.wavelet_name, level=self.wavelet_level)
            
            # Get the approximation coefficients (lowest frequency)
            approx = coeffs[0]
            
            # Resize to hash_size
            if approx.shape[0] != self.hash_size or approx.shape[1] != self.hash_size:
                approx = cv2.resize(approx, (self.hash_size, self.hash_size), 
                                  interpolation=cv2.INTER_AREA)
            
            # Compute the hash by comparing to median
            median_val = np.median(approx)
            whash = (approx > median_val).astype(np.uint8)
            
            return whash.flatten()
            
        except Exception as e:
            logger.error(f"Wavelet hash computation failed: {e}")
            return None
    
    def _compute_basic_hash(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Fallback hash computation when PyWavelets is not available."""
        try:
            # Simple downsampling and thresholding
            img_resized = cv2.resize(img, (self.hash_size, self.hash_size), 
                                   interpolation=cv2.INTER_AREA)
            
            # Apply Gaussian blur for noise reduction
            img_blurred = cv2.GaussianBlur(img_resized, (3, 3), 0)
            
            # Compute hash using mean threshold
            mean_val = np.mean(img_blurred)
            basic_hash = (img_blurred > mean_val).astype(np.uint8)
            
            return basic_hash.flatten()
            
        except Exception as e:
            logger.error(f"Basic hash computation failed: {e}")
            return None
    
    def whash_similarity(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        """
        Compute similarity between two WHashes using Hamming distance.
        
        Args:
            hash1: First wavelet hash
            hash2: Second wavelet hash
            
        Returns:
            similarity: Similarity score between 0 and 1
        """
        if hash1 is None or hash2 is None:
            return 0.0
            
        # Ensure hashes are the same length
        if len(hash1) != len(hash2):
            return 0.0
            
        # Compute Hamming distance
        hamming_dist = np.sum(hash1 != hash2)
        
        # Convert to similarity (0 to 1)
        similarity = 1.0 - (hamming_dist / len(hash1))
        
        return similarity
    
    def group_by_whash(self, 
                       image_paths: List[str], 
                       features_dict: Optional[Dict] = None,
                       progress_callback: Optional[callable] = None) -> List[List[str]]:
        """
        Group images by their wavelet hashes.
        
        Args:
            image_paths: List of image paths to process
            features_dict: Optional pre-computed features dictionary
            progress_callback: Optional callback for progress reporting
            
        Returns:
            groups: List of groups containing similar images
        """
        import time
        start_time = time.time()
        
        logger.info(f"Grouping {len(image_paths)} images by WHash...")
        
        # Compute or retrieve WHashes for all images
        whashes = {}
        valid_paths = []
        
        for i, path in enumerate(image_paths):
            try:
                # Try to get from features_dict if provided
                if (features_dict and path in features_dict and 
                    'whash' in features_dict[path]):
                    whash = features_dict[path]['whash']
                else:
                    # Compute WHash
                    whash = self.compute_whash(path)
                    
                    # Store in features_dict if provided
                    if features_dict is not None:
                        if path not in features_dict:
                            features_dict[path] = {}
                        features_dict[path]['whash'] = whash
                
                if whash is not None:
                    whashes[path] = whash
                    valid_paths.append(path)
                    
                # Progress callback
                if progress_callback:
                    progress = (i + 1) / len(image_paths)
                    progress_callback(f"Computing WHashes: {progress:.1%}")
                    
            except Exception as e:
                logger.warning(f"Failed to process {path} for WHash: {e}")
                continue
        
        if not whashes:
            logger.warning("No valid WHashes computed")
            return [[path] for path in image_paths]
        
        # Create groups using appropriate method
        if self.enable_lsh and len(whashes) > 100:
            groups = self._lsh_grouping(whashes)
        else:
            groups = self._simple_grouping(whashes)
        
        # Update statistics
        self.stats['images_processed'] = len(image_paths)
        self.stats['groups_created'] = len(groups)
        self.stats['processing_time'] = time.time() - start_time
        
        logger.info(f"WHash grouping created {len(groups)} groups in "
                   f"{self.stats['processing_time']:.2f}s")
        return groups
    
    def _lsh_grouping(self, whashes: Dict[str, np.ndarray]) -> List[List[str]]:
        """
        Group WHashes using Locality-Sensitive Hashing (LSH) for efficiency.
        
        Args:
            whashes: Dictionary of image paths to their WHashes
            
        Returns:
            groups: List of groups of similar images
        """
        logger.info("Using LSH-based grouping for efficiency...")
        
        # Create LSH bands
        lsh_buckets = defaultdict(list)
        
        for path, whash in whashes.items():
            # For each band, create a signature
            for band in range(self.lsh_bands):
                start = band * self.lsh_rows_per_band
                end = start + self.lsh_rows_per_band
                
                # Ensure we don't go out of bounds
                if end <= len(whash):
                    band_hash = tuple(whash[start:end])
                    bucket_id = (band, band_hash)
                    lsh_buckets[bucket_id].append(path)
        
        # Create union-find data structure for connected components
        parent = {}
        rank = {}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
        
        # Initialize union-find
        for path in whashes.keys():
            parent[path] = path
            rank[path] = 0
        
        # Union images in the same LSH bucket
        for bucket_paths in lsh_buckets.values():
            if len(bucket_paths) > 1:
                for i in range(len(bucket_paths)):
                    for j in range(i+1, len(bucket_paths)):
                        path1, path2 = bucket_paths[i], bucket_paths[j]
                        # Only union if actually similar (avoid false positives)
                        similarity = self.whash_similarity(whashes[path1], whashes[path2])
                        if similarity >= self.threshold:
                            union(path1, path2)
        
        # Extract connected components
        groups_map = defaultdict(list)
        for path in whashes.keys():
            root = find(path)
            groups_map[root].append(path)
        
        return list(groups_map.values())
    
    def _simple_grouping(self, whashes: Dict[str, np.ndarray]) -> List[List[str]]:
        """
        Simple grouping by comparing all pairs (for smaller datasets).
        
        Args:
            whashes: Dictionary of image paths to their WHashes
            
        Returns:
            groups: List of groups of similar images
        """
        logger.info("Using simple pairwise grouping...")
        
        # This is O(n^2) so only use for small datasets
        if len(whashes) > 1000:
            logger.warning(f"Simple grouping called for {len(whashes)} images, which may be slow")
        
        # Create union-find data structure for connected components
        parent = {}
        rank = {}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
        
        # Initialize union-find
        paths = list(whashes.keys())
        for path in paths:
            parent[path] = path
            rank[path] = 0
        
        # Compare all pairs
        for i in range(len(paths)):
            for j in range(i+1, len(paths)):
                path1, path2 = paths[i], paths[j]
                similarity = self.whash_similarity(whashes[path1], whashes[path2])
                if similarity >= self.threshold:
                    union(path1, path2)
        
        # Extract connected components
        groups_map = defaultdict(list)
        for path in paths:
            root = find(path)
            groups_map[root].append(path)
        
        return list(groups_map.values())
    
    def integrate_with_color_pipeline(self, color_optimized_deduplicator):
        """
        Integrate WHash deduplication with the existing color-optimized pipeline.
        
        Args:
            color_optimized_deduplicator: Instance of ColorOptimizedDeduplicator
            
        Returns:
            Integrated deduplication pipeline
        """
        logger.info("Integrating WHash with color-optimized pipeline...")
        
        # Store original method - try both possible method names
        original_deduplicate = getattr(color_optimized_deduplicator, 
                                     'deduplicate_with_color_prefiltering', None)
        
        if original_deduplicate is None:
            # Fallback to memory-efficient method if color method not available
            original_deduplicate = getattr(color_optimized_deduplicator, 
                                         'deduplicate_memory_efficient', None)
        
        if original_deduplicate is None:
            logger.warning("Color deduplicator doesn't have expected method. "
                          "Integration may not work as expected.")
            return color_optimized_deduplicator
        
        def whash_color_integrated_deduplicate(image_paths, output_dir, 
                                            progress_callback=None, **kwargs):
            """
            Integrated WHash-Color deduplication pipeline.
            
            This method provides a fast first pass using WHash to group similar images,
            then applies the full color-optimized pipeline to each group.
            """
            logger.info("🚀 Starting WHash-Color Integrated Deduplication Pipeline...")
            
            # First pass: WHash grouping
            whash_groups = self.group_by_whash(image_paths, progress_callback=progress_callback)
            
            # Filter out single-image groups
            multi_image_groups = [group for group in whash_groups if len(group) > 1]
            single_images = [group[0] for group in whash_groups if len(group) == 1]
            
            logger.info(f"WHash pre-grouping: {len(multi_image_groups)} groups with potential duplicates, "
                       f"{len(single_images)} unique images")
            
            # Process multi-image groups with color pipeline
            all_duplicate_groups = []
            similarity_scores = {}
            
            for i, group in enumerate(multi_image_groups):
                if progress_callback:
                    progress_callback(f"Processing WHash group {i+1}/{len(multi_image_groups)}")
                
                try:
                    group_duplicates, group_scores = original_deduplicate(
                        group, output_dir, progress_callback, **kwargs
                    )
                    all_duplicate_groups.extend(group_duplicates)
                    similarity_scores.update(group_scores)
                except Exception as e:
                    logger.error(f"Failed to process WHash group {i}: {e}")
                    # Fall back to treating the group as a single group
                    all_duplicate_groups.append(group)
            
            # Add single images as their own groups
            for img in single_images:
                all_duplicate_groups.append([img])
            
            logger.info(f"WHash-Color integration complete: {len(all_duplicate_groups)} final groups")
            return all_duplicate_groups, similarity_scores
        
        # Replace the deduplicate method
        color_optimized_deduplicator.deduplicate_with_whash_color_integration = whash_color_integrated_deduplicate
        
        # Add WHash-specific methods to the deduplicator
        color_optimized_deduplicator.compute_whash = self.compute_whash
        color_optimized_deduplicator.whash_similarity = self.whash_similarity
        color_optimized_deduplicator.group_by_whash = self.group_by_whash
        
        logger.info("WHash integration complete!")
        return color_optimized_deduplicator
    
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'images_processed': 0,
            'hashes_computed': 0,
            'groups_created': 0,
            'processing_time': 0.0,
            'memory_usage_mb': 0.0
        }
    
    def __str__(self) -> str:
        """String representation of the WHash deduplicator."""
        return (f"WHashDeduplicator(hash_size={self.hash_size}, "
                f"wavelet_level={self.wavelet_level}, threshold={self.threshold}, "
                f"lsh_enabled={self.enable_lsh})")


# Factory function for easy creation
def create_whash_deduplicator(**kwargs) -> WHashDeduplicator:
    """
    Factory function to create a WHash deduplicator with sensible defaults.
    
    Args:
        **kwargs: Configuration parameters for WHashDeduplicator
        
    Returns:
        Configured WHashDeduplicator instance
    """
    return WHashDeduplicator(**kwargs)


# Example usage and integration
if __name__ == "__main__":
    # Example of how to use the WHash deduplicator
    logging.basicConfig(level=logging.INFO)
    
    # Create WHash deduplicator
    whash_dedup = WHashDeduplicator(
        hash_size=8,
        wavelet_level=2,
        threshold=0.85,
        enable_lsh=True
    )
    
    print(f"WHash Deduplicator created: {whash_dedup}")
    print(f"PyWavelets available: {PYWAVELETS_AVAILABLE}")
    
    # Example of hash computation
    # test_image = np.random.rand(64, 64).astype(np.uint8)
    # hash_result = whash_dedup.compute_whash(test_image)
    # print(f"Test hash computed: {hash_result is not None}")
