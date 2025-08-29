import logging
import time
import gc
from typing import List, Optional, Tuple, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

# Import our enhanced modules
try:
    from modules.enhanced_whash_deduplicator import EnhancedWHashDeduplicator
    from modules.structural_similarity import StructuralSimilarity
    from modules.hybrid_similarity_calculator import HybridSimilarityCalculator
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    logger.warning("Enhanced modules not available, some features may be limited")


class AccuracyOptimizedDeduplicator:
    """
    Accuracy-optimized deduplicator with enhanced similarity measures.
    
    Features:
    - Multi-scale WHash for scale invariance
    - SSIM for perceptual similarity
    - Hybrid similarity calculation
    - Comprehensive duplicate verification
    - Performance optimization and caching
    """
    
    def __init__(self, 
                 hybrid_calculator: Optional[HybridSimilarityCalculator] = None,
                 whash_deduplicator: Optional[EnhancedWHashDeduplicator] = None,
                 ssim_calculator: Optional[StructuralSimilarity] = None,
                 enable_verification: bool = True,
                 verification_threshold: float = 0.65,
                 max_group_size: int = 1000,
                 enable_caching: bool = True):
        """
        Initialize accuracy-optimized deduplicator.
        
        Args:
            hybrid_calculator: Hybrid similarity calculator instance
            whash_deduplicator: Enhanced WHash deduplicator instance
            ssim_calculator: Structural similarity calculator instance
            enable_verification: Enable duplicate verification
            verification_threshold: Threshold for verification
            max_group_size: Maximum size for verification groups
            enable_caching: Enable similarity caching
        """
        self.hybrid_calculator = hybrid_calculator
        self.whash_deduplicator = whash_deduplicator
        self.ssim_calculator = ssim_calculator
        self.enable_verification = enable_verification
        self.verification_threshold = verification_threshold
        self.max_group_size = max_group_size
        self.enable_caching = enable_caching
        
        # Performance tracking
        self.stats = {
            'total_images_processed': 0,
            'total_groups_found': 0,
            'total_duplicates_found': 0,
            'whash_groups': 0,
            'verified_groups': 0,
            'total_processing_time': 0.0,
            'average_time_per_image': 0.0
        }
        
        logger.info("Accuracy-optimized deduplicator initialized")
    
    def find_duplicates(self, image_paths: List[str], 
                       progress_callback: Optional[callable] = None) -> List[List[str]]:
        """
        Find duplicate groups with enhanced accuracy.
        
        Args:
            image_paths: List of image paths to process
            progress_callback: Optional callback for progress reporting
            
        Returns:
            List of duplicate groups
        """
        start_time = time.time()
        logger.info(f"Starting accuracy-optimized deduplication for {len(image_paths)} images")
        
        try:
            # First pass: WHash grouping for fast pre-filtering
            if progress_callback:
                progress_callback("Stage 1: WHash pre-grouping...")
            
            whash_groups = self._whash_grouping(image_paths, progress_callback)
            self.stats['whash_groups'] = len(whash_groups)
            
            # Second pass: Verify duplicates with hybrid similarity
            if progress_callback:
                progress_callback("Stage 2: Duplicate verification...")
            
            duplicate_groups = []
            total_duplicates = 0
            
            for i, group in enumerate(whash_groups):
                if progress_callback:
                    progress_callback(f"Verifying group {i+1}/{len(whash_groups)} ({len(group)} images)")
                
                if len(group) == 1:
                    # Single image, no duplicates
                    duplicate_groups.append(group)
                else:
                    # Multiple images, verify duplicates
                    verified_group = self._verify_group_with_hybrid_similarity(
                        group, progress_callback
                    )
                    duplicate_groups.append(verified_group)
                    total_duplicates += len(verified_group) - 1  # Subtract 1 for the original
            
            # Update stats
            processing_time = time.time() - start_time
            self.stats['total_images_processed'] += len(image_paths)
            self.stats['total_groups_found'] += len(duplicate_groups)
            self.stats['total_duplicates_found'] += total_duplicates
            self.stats['verified_groups'] += len([g for g in duplicate_groups if len(g) > 1])
            self.stats['total_processing_time'] += processing_time
            self.stats['average_time_per_image'] = (
                self.stats['total_processing_time'] / 
                self.stats['total_images_processed']
            )
            
            logger.info(f"Accuracy-optimized deduplication completed: "
                       f"{len(duplicate_groups)} groups, {total_duplicates} duplicates "
                       f"in {processing_time:.2f}s")
            
            return duplicate_groups
            
        except Exception as e:
            logger.error(f"Accuracy-optimized deduplication failed: {e}")
            # Return single-image groups as fallback
            return [[path] for path in image_paths]
        finally:
            # Clean up
            gc.collect()
    
    def _whash_grouping(self, image_paths: List[str], 
                        progress_callback: Optional[callable] = None) -> List[List[str]]:
        """
        Group images using enhanced WHash with multi-scale hashing.
        
        Args:
            image_paths: List of image paths to group
            progress_callback: Optional callback for progress reporting
            
        Returns:
            List of image groups
        """
        if self.whash_deduplicator is None:
            # Fallback to single-image groups
            logger.warning("No WHash deduplicator available, using single-image groups")
            return [[path] for path in image_paths]
        
        try:
            return self.whash_deduplicator.group_by_whash(image_paths, progress_callback)
        except Exception as e:
            logger.error(f"WHash grouping failed: {e}, using single-image groups")
            return [[path] for path in image_paths]
    
    def _verify_group_with_hybrid_similarity(self, group: List[str], 
                                           progress_callback: Optional[callable] = None) -> List[str]:
        """
        Verify duplicates using hybrid similarity measures.
        
        Args:
            group: List of image paths to verify
            progress_callback: Optional callback for progress reporting
            
        Returns:
            Verified list of duplicate images
        """
        if len(group) <= 1:
            return group
        
        if self.hybrid_calculator is None:
            logger.warning("No hybrid calculator available, returning original group")
            return group
        
        try:
            # Use the first image as anchor
            anchor = group[0]
            duplicates = [anchor]
            
            # Limit group size for verification to avoid performance issues
            if len(group) > self.max_group_size:
                logger.warning(f"Group size {len(group)} exceeds maximum {self.max_group_size}, "
                             f"limiting verification to first {self.max_group_size} images")
                group = group[:self.max_group_size]
            
            # Verify each candidate against the anchor
            for i, candidate in enumerate(group[1:], 1):
                if progress_callback:
                    progress_callback(f"Verifying image {i}/{len(group)-1}")
                
                try:
                    # Compute hybrid similarity
                    similarity_scores = self.hybrid_calculator.compute_hybrid_similarity(
                        anchor, candidate
                    )
                    
                    overall_score = similarity_scores.get('overall', 0.0)
                    
                    if overall_score >= self.verification_threshold:
                        duplicates.append(candidate)
                        logger.debug(f"Duplicate verified: {candidate} (score: {overall_score:.3f})")
                    else:
                        logger.debug(f"Not a duplicate: {candidate} (score: {overall_score:.3f})")
                        
                except Exception as e:
                    logger.error(f"Failed to verify {candidate}: {e}")
                    # Include in duplicates by default if verification fails
                    duplicates.append(candidate)
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Group verification failed: {e}, returning original group")
            return group
    
    def _verify_group_with_individual_similarity(self, group: List[str], 
                                               progress_callback: Optional[callable] = None) -> List[str]:
        """
        Alternative verification method using individual similarity measures.
        
        Args:
            group: List of image paths to verify
            progress_callback: Optional callback for progress reporting
            
        Returns:
            Verified list of duplicate images
        """
        if len(group) <= 1:
            return group
        
        try:
            # Use the first image as anchor
            anchor = group[0]
            duplicates = [anchor]
            
            # Load anchor image once
            anchor_image = self._load_image(anchor)
            if anchor_image is None:
                logger.error(f"Failed to load anchor image: {anchor}")
                return group
            
            # Verify each candidate
            for i, candidate in enumerate(group[1:], 1):
                if progress_callback:
                    progress_callback(f"Verifying image {i}/{len(group)-1}")
                
                try:
                    candidate_image = self._load_image(candidate)
                    if candidate_image is None:
                        continue
                    
                    # Compute individual similarity measures
                    similarity_scores = {}
                    
                    # WHash similarity
                    if self.whash_deduplicator:
                        whash_similarity = self._compute_whash_similarity(
                            anchor_image, candidate_image
                        )
                        similarity_scores['whash'] = whash_similarity
                    
                    # SSIM similarity
                    if self.ssim_calculator:
                        ssim_score = self.ssim_calculator.compute_ssim(
                            anchor_image, candidate_image
                        )
                        similarity_scores['structural'] = ssim_score
                    
                    # Color similarity
                    color_similarity = self._compute_color_similarity(
                        anchor_image, candidate_image
                    )
                    similarity_scores['color'] = color_similarity
                    
                    # Compute overall score (simple average)
                    if similarity_scores:
                        overall_score = sum(similarity_scores.values()) / len(similarity_scores)
                    else:
                        overall_score = 0.0
                    
                    if overall_score >= self.verification_threshold:
                        duplicates.append(candidate)
                        logger.debug(f"Duplicate verified: {candidate} (score: {overall_score:.3f})")
                    else:
                        logger.debug(f"Not a duplicate: {candidate} (score: {overall_score:.3f})")
                    
                    # Clean up
                    del candidate_image
                    
                except Exception as e:
                    logger.error(f"Failed to verify {candidate}: {e}")
                    # Include in duplicates by default if verification fails
                    duplicates.append(candidate)
            
            # Clean up
            del anchor_image
            gc.collect()
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Individual similarity verification failed: {e}")
            return group
    
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
            import cv2
            import numpy as np
            
            # Resize images for consistent comparison
            target_size = (64, 64)
            img1_resized = cv2.resize(img1, target_size, interpolation=cv2.INTER_AREA)
            img2_resized = cv2.resize(img2, target_size, interpolation=cv2.INTER_AREA)
            
            # Convert to HSV color space
            hsv1 = cv2.cvtColor(img1_resized, cv2.COLOR_RGB2HSV)
            hsv2 = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2HSV)
            
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
            import numpy as np
            
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
    
    def _load_image(self, image_path: str):
        """
        Load image with support for local and Azure paths.
        
        Args:
            image_path: Path to image (local or Azure blob)
            
        Returns:
            Loaded image array or None if failed
        """
        try:
            import cv2
            
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
    
    def _load_azure_image(self, image_path: str):
        """
        Load image from Azure Blob Storage.
        
        Args:
            image_path: Azure blob URL
            
        Returns:
            Loaded image array or None if failed
        """
        try:
            import cv2
            
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
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'total_images_processed': 0,
            'total_groups_found': 0,
            'total_duplicates_found': 0,
            'whash_groups': 0,
            'verified_groups': 0,
            'total_processing_time': 0.0,
            'average_time_per_image': 0.0
        }
    
    def release(self):
        """Release resources."""
        logger.info("Releasing accuracy-optimized deduplicator resources...")
        self.reset_stats()
        gc.collect()


def create_accuracy_optimized_deduplicator(
    hybrid_calculator: Optional[HybridSimilarityCalculator] = None,
    whash_deduplicator: Optional[EnhancedWHashDeduplicator] = None,
    ssim_calculator: Optional[StructuralSimilarity] = None,
    enable_verification: bool = True,
    verification_threshold: float = 0.65,
    max_group_size: int = 1000,
    enable_caching: bool = True
) -> AccuracyOptimizedDeduplicator:
    """
    Factory function to create accuracy-optimized deduplicator.
    
    Args:
        hybrid_calculator: Hybrid similarity calculator instance
        whash_deduplicator: Enhanced WHash deduplicator instance
        ssim_calculator: Structural similarity calculator instance
        enable_verification: Enable duplicate verification
        verification_threshold: Threshold for verification
        max_group_size: Maximum size for verification groups
        enable_caching: Enable similarity caching
        
    Returns:
        Configured AccuracyOptimizedDeduplicator instance
    """
    return AccuracyOptimizedDeduplicator(
        hybrid_calculator=hybrid_calculator,
        whash_deduplicator=whash_deduplicator,
        ssim_calculator=ssim_calculator,
        enable_verification=enable_verification,
        verification_threshold=verification_threshold,
        max_group_size=max_group_size,
        enable_caching=enable_caching
    )
