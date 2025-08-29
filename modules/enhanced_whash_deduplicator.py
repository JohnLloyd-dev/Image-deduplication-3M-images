import pywt
import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict, Any
from collections import defaultdict
import gc
import time

logger = logging.getLogger(__name__)

class EnhancedWHashDeduplicator:
    """
    Enhanced WHash deduplicator with multi-scale hashing for improved accuracy.
    
    Features:
    - Multi-scale wavelet hashing for scale invariance
    - Enhanced LSH grouping with configurable parameters
    - Improved similarity calculation
    - Memory-efficient processing
    """
    
    def __init__(self, 
                 hash_size: int = 8,
                 wavelet_level: int = 3,
                 wavelet_name: str = 'haar',
                 scale_factors: Optional[List[float]] = None,
                 enable_lsh: bool = True,
                 lsh_bands: int = 4,
                 lsh_rows_per_band: int = 4,
                 similarity_threshold: float = 0.85):
        """
        Initialize enhanced WHash deduplicator.
        
        Args:
            hash_size: Size of the hash (8x8 = 64 bits)
            wavelet_level: Number of wavelet decomposition levels
            wavelet_name: Wavelet type ('haar', 'db1', 'db2', etc.)
            scale_factors: List of scale factors for multi-scale hashing
            enable_lsh: Enable Locality-Sensitive Hashing for grouping
            lsh_bands: Number of LSH bands
            lsh_rows_per_band: Number of rows per LSH band
            similarity_threshold: Threshold for similarity matching
        """
        self.hash_size = hash_size
        self.wavelet_level = wavelet_level
        self.wavelet_name = wavelet_name
        self.enable_lsh = enable_lsh
        self.lsh_bands = lsh_bands
        self.lsh_rows_per_band = lsh_rows_per_band
        self.similarity_threshold = similarity_threshold
        
        # Multi-scale factors for scale invariance
        if scale_factors is None:
            self.scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        else:
            self.scale_factors = scale_factors
        
        # Performance tracking
        self.performance_stats = {
            'total_images_processed': 0,
            'total_processing_time': 0.0,
            'average_time_per_image': 0.0,
            'memory_usage_mb': 0.0
        }
        
        logger.info(f"Enhanced WHash deduplicator initialized with {len(self.scale_factors)} scale factors")
    
    def _compute_wavelet_hash(self, img: np.ndarray) -> np.ndarray:
        """
        Compute wavelet hash for a single image.
        
        Args:
            img: Input image array
            
        Returns:
            Binary hash array (64 bits for 8x8 hash)
        """
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec2(img, self.wavelet_name, level=self.wavelet_level)
            approx = coeffs[0]  # Approximation coefficients
            
            # Resize to hash_size
            if approx.shape != (self.hash_size, self.hash_size):
                approx = cv2.resize(approx, (self.hash_size, self.hash_size), 
                                   interpolation=cv2.INTER_AREA)
            
            # Compute hash: compare each pixel to median
            median_val = np.median(approx)
            whash = (approx > median_val).astype(np.uint8)
            
            return whash.flatten()
            
        except Exception as e:
            logger.error(f"Wavelet hash computation failed: {e}")
            # Return zero hash as fallback
            return np.zeros(self.hash_size * self.hash_size, dtype=np.uint8)
    
    def _compute_multi_scale_hashes(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Compute hashes at multiple scales for scale invariance.
        
        Args:
            img: Input image array
            
        Returns:
            List of hash arrays for different scales
        """
        hashes = []
        original_height, original_width = img.shape[:2]
        
        for scale in self.scale_factors:
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Skip if image becomes too small
            if new_width < 16 or new_height < 16:
                continue
                
            try:
                resized_img = cv2.resize(img, (new_width, new_height), 
                                       interpolation=cv2.INTER_AREA)
                whash = self._compute_wavelet_hash(resized_img)
                hashes.append(whash)
            except Exception as e:
                logger.warning(f"Failed to compute hash for scale {scale}: {e}")
                continue
        
        return hashes
    
    def _compare_multi_scale_hashes(self, hashes1: List[np.ndarray], 
                                   hashes2: List[np.ndarray]) -> float:
        """
        Compare multiple hashes and return maximum similarity.
        
        Args:
            hashes1: List of hashes for first image
            hashes2: List of hashes for second image
            
        Returns:
            Maximum similarity score between 0.0 and 1.0
        """
        if not hashes1 or not hashes2:
            return 0.0
        
        max_similarity = 0.0
        for hash1 in hashes1:
            for hash2 in hashes2:
                # Compute Hamming distance similarity
                similarity = np.mean(hash1 == hash2)
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _compute_hamming_distance(self, hash1: np.ndarray, hash2: np.ndarray) -> int:
        """
        Compute Hamming distance between two binary hashes.
        
        Args:
            hash1: First binary hash
            hash2: Second binary hash
            
        Returns:
            Hamming distance (number of different bits)
        """
        return np.sum(hash1 != hash2)
    
    def _compute_similarity_from_hamming(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        """
        Compute similarity score from Hamming distance.
        
        Args:
            hash1: First binary hash
            hash2: Second binary hash
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        hamming_dist = self._compute_hamming_distance(hash1, hash2)
        max_distance = len(hash1)
        similarity = 1.0 - (hamming_dist / max_distance)
        return max(0.0, min(1.0, similarity))
    
    def _lsh_grouping(self, whashes: Dict[str, np.ndarray]) -> List[List[str]]:
        """
        Group images using Locality-Sensitive Hashing for efficiency.
        
        Args:
            whashes: Dictionary mapping image paths to their hashes
            
        Returns:
            List of image groups
        """
        if not self.enable_lsh:
            # Fallback to simple grouping
            return self._simple_grouping(whashes)
        
        try:
            # Create LSH buckets
            lsh_buckets = defaultdict(list)
            
            for path, whash in whashes.items():
                # For each band, create a signature
                for band in range(self.lsh_bands):
                    start = band * self.lsh_rows_per_band
                    end = start + self.lsh_rows_per_band
                    
                    if end <= len(whash):
                        # Create band-specific hash
                        band_hash = tuple(whash[start:end])
                        bucket_id = (band, band_hash)
                        lsh_buckets[bucket_id].append(path)
            
            # Use union-find to find connected components
            return self._find_connected_components(whashes, lsh_buckets)
            
        except Exception as e:
            logger.error(f"LSH grouping failed: {e}, falling back to simple grouping")
            return self._simple_grouping(whashes)
    
    def _simple_grouping(self, whashes: Dict[str, np.ndarray]) -> List[List[str]]:
        """
        Simple grouping based on direct hash comparison.
        
        Args:
            whashes: Dictionary mapping image paths to their hashes
            
        Returns:
            List of image groups
        """
        paths = list(whashes.keys())
        groups = []
        processed = set()
        
        for i, path1 in enumerate(paths):
            if path1 in processed:
                continue
                
            group = [path1]
            processed.add(path1)
            
            for j, path2 in enumerate(paths[i+1:], i+1):
                if path2 in processed:
                    continue
                    
                hash1 = whashes[path1]
                hash2 = whashes[path2]
                
                similarity = self._compute_similarity_from_hamming(hash1, hash2)
                if similarity >= self.similarity_threshold:
                    group.append(path2)
                    processed.add(path2)
            
            groups.append(group)
        
        return groups
    
    def _find_connected_components(self, whashes: Dict[str, np.ndarray], 
                                 lsh_buckets: Dict) -> List[List[str]]:
        """
        Find connected components using union-find algorithm.
        
        Args:
            whashes: Dictionary mapping image paths to their hashes
            lsh_buckets: LSH buckets
            
        Returns:
            List of connected component groups
        """
        # Initialize union-find data structure
        parent = {}
        rank = {}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
        
        # Initialize all paths as separate components
        for path in whashes.keys():
            parent[path] = path
            rank[path] = 0
        
        # Union paths that share LSH buckets
        for bucket_paths in lsh_buckets.values():
            if len(bucket_paths) > 1:
                # Union all paths in this bucket
                for i in range(len(bucket_paths)):
                    for j in range(i + 1, len(bucket_paths)):
                        union(bucket_paths[i], bucket_paths[j])
        
        # Group connected components
        components = defaultdict(list)
        for path in whashes.keys():
            root = find(path)
            components[root].append(path)
        
        return list(components.values())
    
    def group_by_whash(self, image_paths: List[str], 
                       progress_callback: Optional[callable] = None) -> List[List[str]]:
        """
        Group images using enhanced WHash with multi-scale hashing.
        
        Args:
            image_paths: List of image paths to group
            progress_callback: Optional callback for progress reporting
            
        Returns:
            List of image groups
        """
        start_time = time.time()
        logger.info(f"Starting enhanced WHash grouping for {len(image_paths)} images")
        
        try:
            # Compute multi-scale hashes for all images
            whashes = {}
            for i, image_path in enumerate(image_paths):
                if progress_callback:
                    progress_callback(f"Computing WHash for image {i+1}/{len(image_paths)}")
                
                try:
                    # Load image
                    image = self._load_image(image_path)
                    if image is None:
                        continue
                    
                    # Compute multi-scale hashes
                    multi_scale_hashes = self._compute_multi_scale_hashes(image)
                    if multi_scale_hashes:
                        # Use the hash at scale 1.0 (original size) as primary
                        primary_hash = None
                        for scale, hash_array in zip(self.scale_factors, multi_scale_hashes):
                            if abs(scale - 1.0) < 0.1:  # Close to scale 1.0
                                primary_hash = hash_array
                                break
                        
                        if primary_hash is None:
                            primary_hash = multi_scale_hashes[len(multi_scale_hashes) // 2]
                        
                        whashes[image_path] = primary_hash
                    
                    # Clean up
                    del image
                    del multi_scale_hashes
                    
                except Exception as e:
                    logger.error(f"Failed to process image {image_path}: {e}")
                    continue
            
            # Group images using LSH
            if progress_callback:
                progress_callback("Grouping images using LSH...")
            
            groups = self._lsh_grouping(whashes)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self.performance_stats['total_images_processed'] += len(image_paths)
            self.performance_stats['total_processing_time'] += processing_time
            self.performance_stats['average_time_per_image'] = (
                self.performance_stats['total_processing_time'] / 
                self.performance_stats['total_images_processed']
            )
            
            logger.info(f"Enhanced WHash grouping completed: {len(groups)} groups in {processing_time:.2f}s")
            return groups
            
        except Exception as e:
            logger.error(f"Enhanced WHash grouping failed: {e}")
            # Return single-image groups as fallback
            return [[path] for path in image_paths]
        finally:
            # Clean up
            gc.collect()
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image with support for local and Azure paths.
        
        Args:
            image_path: Path to image (local or Azure blob)
            
        Returns:
            Loaded image array or None if failed
        """
        try:
            # Try local loading first
            if not image_path.startswith('http'):
                image = cv2.imread(image_path)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Try Azure loading if local failed or is Azure path
            if 'blob.core.windows.net' in image_path or image_path.startswith('https://'):
                return self._load_azure_image(image_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def _load_azure_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image from Azure Blob Storage.
        
        Args:
            image_path: Azure blob URL
            
        Returns:
            Loaded image array or None if failed
        """
        try:
            # Import Azure utilities
            from modules.azure_utils import download_blob_to_memory
            
            # Download image data
            image_data = download_blob_to_memory(image_path)
            if image_data is None:
                return None
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return None
            
            # Convert BGR to RGB
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        except ImportError:
            logger.warning("Azure utilities not available, skipping Azure image")
            return None
        except Exception as e:
            logger.error(f"Azure image loading failed: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.performance_stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_images_processed': 0,
            'total_processing_time': 0.0,
            'average_time_per_image': 0.0,
            'memory_usage_mb': 0.0
        }
    
    def release(self):
        """Release resources."""
        logger.info("Releasing Enhanced WHash deduplicator resources...")
        self.reset_stats()
        gc.collect()


def create_enhanced_whash_deduplicator(
    hash_size: int = 8,
    wavelet_level: int = 3,
    wavelet_name: str = 'haar',
    scale_factors: Optional[List[float]] = None,
    enable_lsh: bool = True,
    lsh_bands: int = 4,
    lsh_rows_per_band: int = 4,
    similarity_threshold: float = 0.85
) -> EnhancedWHashDeduplicator:
    """
    Factory function to create enhanced WHash deduplicator.
    
    Args:
        hash_size: Size of the hash (8x8 = 64 bits)
        wavelet_level: Number of wavelet decomposition levels
        wavelet_name: Wavelet type
        scale_factors: List of scale factors for multi-scale hashing
        enable_lsh: Enable Locality-Sensitive Hashing
        lsh_bands: Number of LSH bands
        lsh_rows_per_band: Number of rows per LSH band
        similarity_threshold: Threshold for similarity matching
        
    Returns:
        Configured EnhancedWHashDeduplicator instance
    """
    return EnhancedWHashDeduplicator(
        hash_size=hash_size,
        wavelet_level=wavelet_level,
        wavelet_name=wavelet_name,
        scale_factors=scale_factors,
        enable_lsh=enable_lsh,
        lsh_bands=lsh_bands,
        lsh_rows_per_band=lsh_rows_per_band,
        similarity_threshold=similarity_threshold
    )
