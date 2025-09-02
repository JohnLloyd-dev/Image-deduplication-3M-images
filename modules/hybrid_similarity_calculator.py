import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict, Any
import gc
import time

logger = logging.getLogger(__name__)

# Import our modules
try:
    from modules.enhanced_whash_deduplicator import EnhancedWHashDeduplicator
    from modules.structural_similarity import StructuralSimilarity
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    logger.warning("Enhanced modules not available, some features may be limited")


class HybridSimilarityCalculator:
    """
    Hybrid similarity calculator combining multiple similarity measures.
    
    Features:
    - WHash similarity for fast structural comparison
    - SSIM for perceptual similarity
    - Color similarity for color-based comparison
    - Weighted combination for optimal results
    - Caching for performance optimization
    """
    
    def __init__(self, 
                 whash_deduplicator: Optional[EnhancedWHashDeduplicator] = None,
                 ssim_calculator: Optional[StructuralSimilarity] = None,
                 weights: Optional[Dict[str, float]] = None,
                 thresholds: Optional[Dict[str, float]] = None,
                 enable_caching: bool = True,
                 cache_size: int = 10000):
        """
        Initialize hybrid similarity calculator.
        
        Args:
            whash_deduplicator: Enhanced WHash deduplicator instance
            ssim_calculator: Structural similarity calculator instance
            weights: Weights for different similarity measures
            thresholds: Thresholds for different similarity measures
            enable_caching: Enable similarity caching
            cache_size: Maximum cache size
        """
        self.whash_deduplicator = whash_deduplicator
        self.ssim_calculator = ssim_calculator
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        # Default weights for different similarity measures
        if weights is None:
            self.weights = {
                'structural': 0.6,  # Highest weight for structural similarity
                'color': 0.25,      # Medium weight for color
                'whash': 0.15       # Lower weight for WHash (fast but less accurate)
            }
        else:
            self.weights = weights
        
        # Default thresholds
        if thresholds is None:
            self.thresholds = {
                'global': 0.65,     # Overall similarity threshold
                'structural': 0.60, # SSIM threshold
                'color': 0.55,      # Color similarity threshold
                'whash': 0.60       # WHash threshold
            }
        else:
            self.thresholds = thresholds
        
        # Initialize cache
        if self.enable_caching:
            self.similarity_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
        else:
            self.similarity_cache = None
        
        # Performance tracking
        self.stats = {
            'total_comparisons': 0,
            'cached_comparisons': 0,
            'whash_comparisons': 0,
            'ssim_comparisons': 0,
            'color_comparisons': 0,
            'total_time': 0.0,
            'average_time': 0.0
        }
        
        logger.info(f"Hybrid similarity calculator initialized with weights: {self.weights}")
    
    def compute_hybrid_similarity(self, img1_path: str, img2_path: str) -> Dict[str, float]:
        """
        Compute comprehensive similarity between two images.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            
        Returns:
            Dictionary with similarity scores and overall score
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(img1_path, img2_path)
        if self.enable_caching and cache_key in self.similarity_cache:
            self.stats['cached_comparisons'] += 1
            self.stats['total_comparisons'] += 1
            return self.similarity_cache[cache_key]
        
        try:
            # Load images
            img1 = self._load_image(img1_path)
            img2 = self._load_image(img2_path)
            
            if img1 is None or img2 is None:
                result = {'overall': 0.0, 'error': 'Failed to load images'}
                self._cache_result(cache_key, result)
                return result
            
            # Compute individual similarity measures
            similarity_scores = {}
            
            # WHash similarity
            if self.whash_deduplicator:
                whash_similarity = self._compute_whash_similarity(img1, img2)
                similarity_scores['whash'] = whash_similarity
                self.stats['whash_comparisons'] += 1
            else:
                similarity_scores['whash'] = 0.0
            
            # SSIM similarity
            if self.ssim_calculator:
                ssim_score = self.ssim_calculator.compute_ssim(img1, img2)
                similarity_scores['structural'] = ssim_score
                self.stats['ssim_comparisons'] += 1
            else:
                similarity_scores['structural'] = 0.0
            
            # Color similarity
            color_similarity = self._compute_color_similarity(img1, img2)
            similarity_scores['color'] = color_similarity
            self.stats['color_comparisons'] += 1
            
            # Compute weighted overall score
            overall_score = self._compute_weighted_score(similarity_scores)
            similarity_scores['overall'] = overall_score
            
            # Update stats
            processing_time = time.time() - start_time
            self.stats['total_comparisons'] += 1
            self.stats['total_time'] += processing_time
            self.stats['average_time'] = self.stats['total_time'] / self.stats['total_comparisons']
            
            # Cache result
            self._cache_result(cache_key, similarity_scores)
            
            # Clean up
            del img1, img2
            gc.collect()
            
            return similarity_scores
            
        except Exception as e:
            logger.error(f"Hybrid similarity computation failed: {e}")
            result = {'overall': 0.0, 'error': str(e)}
            self._cache_result(cache_key, result)
            return result
    
    def _compute_whash_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute WHash similarity between two images.
        
        Args:
            img1: First image array
            img2: Second image array
            
        Returns:
            WHash similarity score
        """
        try:
            if self.whash_deduplicator is None:
                return 0.0
            
            # Compute multi-scale hashes
            hashes1 = self.whash_deduplicator._compute_multi_scale_hashes(img1)
            hashes2 = self.whash_deduplicator._compute_multi_scale_hashes(img2)
            
            # Compare hashes
            similarity = self.whash_deduplicator._compare_multi_scale_hashes(hashes1, hashes2)
            
            return similarity
            
        except Exception as e:
            logger.error(f"WHash similarity computation failed: {e}")
            return 0.0
    
    def _compute_color_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute color similarity between two images.
        
        Args:
            img1: First image array
            img2: Second image array
            
        Returns:
            Color similarity score
        """
        try:
            # Resize images for consistent comparison
            target_size = (64, 64)
            img1_resized = cv2.resize(img1, target_size, interpolation=cv2.INTER_AREA)
            img2_resized = cv2.resize(img2, target_size, interpolation=cv2.INTER_AREA)
            
            # Convert to multiple color spaces
            hsv1 = cv2.cvtColor(img1_resized, cv2.COLOR_RGB2HSV)
            hsv2 = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2HSV)
            
            lab1 = cv2.cvtColor(img1_resized, cv2.COLOR_RGB2LAB)
            lab2 = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2LAB)
            
            # Compute histograms
            h_hist1 = cv2.calcHist([hsv1], [0], None, [16], [0, 180]).flatten()
            s_hist1 = cv2.calcHist([hsv1], [1], None, [16], [0, 256]).flatten()
            v_hist1 = cv2.calcHist([hsv1], [2], None, [16], [0, 256]).flatten()
            
            h_hist2 = cv2.calcHist([hsv2], [0], None, [16], [0, 180]).flatten()
            s_hist2 = cv2.calcHist([hsv2], [1], None, [16], [0, 256]).flatten()
            v_hist2 = cv2.calcHist([hsv2], [2], None, [16], [0, 256]).flatten()
            
            # Normalize histograms
            h_hist1 = h_hist1 / (h_hist1.sum() + 1e-8)
            s_hist1 = s_hist1 / (s_hist1.sum() + 1e-8)
            v_hist1 = v_hist1 / (v_hist1.sum() + 1e-8)
            
            h_hist2 = h_hist2 / (h_hist2.sum() + 1e-8)
            s_hist2 = s_hist2 / (s_hist2.sum() + 1e-8)
            v_hist2 = v_hist2 / (v_hist2.sum() + 1e-8)
            
            # Compute cosine similarity for each channel
            h_similarity = self._cosine_similarity(h_hist1, h_hist2)
            s_similarity = self._cosine_similarity(s_hist1, s_hist2)
            v_similarity = self._cosine_similarity(v_hist1, v_hist2)
            
            # Weighted average of channel similarities
            color_similarity = (0.5 * h_similarity + 0.3 * s_similarity + 0.2 * v_similarity)
            
            return max(0.0, min(1.0, color_similarity))
            
        except Exception as e:
            logger.error(f"Color similarity computation failed: {e}")
            return 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Cosine similarity computation failed: {e}")
            return 0.0
    
    def _compute_weighted_score(self, similarity_scores: Dict[str, float]) -> float:
        """
        Compute weighted overall similarity score.
        
        Args:
            similarity_scores: Dictionary with individual similarity scores
            
        Returns:
            Weighted overall score
        """
        try:
            overall_score = 0.0
            total_weight = 0.0
            
            for measure, weight in self.weights.items():
                if measure in similarity_scores:
                    score = similarity_scores[measure]
                    overall_score += weight * score
                    total_weight += weight
            
            if total_weight > 0:
                return overall_score / total_weight
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Weighted score computation failed: {e}")
            return 0.0
    
    def _get_cache_key(self, img1_path: str, img2_path: str) -> str:
        """
        Generate cache key for two image paths.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            
        Returns:
            Cache key string
        """
        # Sort paths to ensure consistent key regardless of order
        sorted_paths = tuple(sorted([img1_path, img2_path]))
        return f"{hash(sorted_paths)}"
    
    def _cache_result(self, cache_key: str, result: Dict[str, float]):
        """
        Cache similarity result.
        
        Args:
            cache_key: Cache key
            result: Similarity result to cache
        """
        if not self.enable_caching or self.similarity_cache is None:
            return
        
        try:
            # Check cache size and evict if necessary
            if len(self.similarity_cache) >= self.cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self.similarity_cache))
                del self.similarity_cache[oldest_key]
            
            # Cache the result
            self.similarity_cache[cache_key] = result
            
        except Exception as e:
            logger.error(f"Caching failed: {e}")
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image with support for local and Azure paths.
        
        Args:
            image_path: Path to image (local or Azure blob)
            
        Returns:
            Loaded image array or None if failed
        """
        try:
            # Check if this is a relative Azure path (starts with Image_Dedup_Project/)
            if image_path.startswith('Image_Dedup_Project/'):
                return self._load_azure_image(image_path)
            
            # Try local loading for non-Azure paths
            if not image_path.startswith('http'):
                image = cv2.imread(image_path)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Try Azure loading for full URLs
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
            from modules.azure_utils import download_blob_to_memory, SAS_URL
            
            # Download image data
            image_data = download_blob_to_memory(image_path, SAS_URL)
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
    
    def is_similar(self, img1_path: str, img2_path: str) -> bool:
        """
        Check if two images are similar based on overall threshold.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            
        Returns:
            True if images are similar, False otherwise
        """
        similarity_scores = self.compute_hybrid_similarity(img1_path, img2_path)
        overall_score = similarity_scores.get('overall', 0.0)
        return overall_score >= self.thresholds['global']
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache metrics
        """
        if not self.enable_caching or self.similarity_cache is None:
            return {}
        
        return {
            'cache_size': len(self.similarity_cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = self.stats.copy()
        
        # Add cache stats
        cache_stats = self.get_cache_stats()
        stats.update(cache_stats)
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'total_comparisons': 0,
            'cached_comparisons': 0,
            'whash_comparisons': 0,
            'ssim_comparisons': 0,
            'color_comparisons': 0,
            'total_time': 0.0,
            'average_time': 0.0
        }
        
        if self.enable_caching:
            self.cache_hits = 0
            self.cache_misses = 0
    
    def clear_cache(self):
        """Clear similarity cache."""
        if self.enable_caching and self.similarity_cache is not None:
            self.similarity_cache.clear()
            logger.info("Similarity cache cleared")
    
    def release(self):
        """Release resources."""
        logger.info("Releasing hybrid similarity calculator resources...")
        self.clear_cache()
        self.reset_stats()
        gc.collect()


def create_hybrid_similarity_calculator(
    whash_deduplicator: Optional[EnhancedWHashDeduplicator] = None,
    ssim_calculator: Optional[StructuralSimilarity] = None,
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    enable_caching: bool = True,
    cache_size: int = 10000
) -> HybridSimilarityCalculator:
    """
    Factory function to create hybrid similarity calculator.
    
    Args:
        whash_deduplicator: Enhanced WHash deduplicator instance
        ssim_calculator: Structural similarity calculator instance
        weights: Weights for different similarity measures
        thresholds: Thresholds for different similarity measures
        enable_caching: Enable similarity caching
        cache_size: Maximum cache size
        
    Returns:
        Configured HybridSimilarityCalculator instance
    """
    return HybridSimilarityCalculator(
        whash_deduplicator=whash_deduplicator,
        ssim_calculator=ssim_calculator,
        weights=weights,
        thresholds=thresholds,
        enable_caching=enable_caching,
        cache_size=cache_size
    )
