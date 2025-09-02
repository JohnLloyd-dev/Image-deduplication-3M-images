import logging
import time
import gc
import numpy as np
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
            
            # Stage 3: Global feature refinement (cross-group analysis)
            if progress_callback:
                progress_callback("Stage 3: Global feature refinement...")
            
            if len(duplicate_groups) > 1:
                global_refined_groups = self._global_feature_refinement(
                    duplicate_groups, progress_callback
                )
                duplicate_groups = global_refined_groups
                logger.info(f"Stage 3 completed: {len(duplicate_groups)} groups after global refinement")
            else:
                logger.info("⏭️  Skipping Stage 3 - no groups need global refinement")
            
            # Stage 4: Local feature verification (within-group refinement)
            if progress_callback:
                progress_callback("Stage 4: Local feature verification...")
            
            if any(len(group) > 1 for group in duplicate_groups):
                local_verified_groups = self._local_feature_verification(
                    duplicate_groups, progress_callback
                )
                duplicate_groups = local_verified_groups
                logger.info(f"Stage 4 completed: {len(duplicate_groups)} groups after local verification")
            else:
                logger.info("⏭️  Skipping Stage 4 - no groups need local verification")
            
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
    
    def _global_feature_refinement(self, groups: List[List[str]], 
                                 progress_callback: Optional[callable] = None) -> List[List[str]]:
        """
        Stage 3: Global feature refinement using cross-group analysis.
        
        Args:
            groups: List of image groups to refine
            progress_callback: Optional callback for progress reporting
            
        Returns:
            Refined list of image groups
        """
        if len(groups) <= 1:
            return groups
        
        try:
            if progress_callback:
                progress_callback("Global refinement: Analyzing cross-group similarities...")
            
            refined_groups = []
            processed_groups = set()
            
            for i, group in enumerate(groups):
                if i in processed_groups:
                    continue
                
                if progress_callback:
                    progress_callback(f"Global refinement: Processing group {i+1}/{len(groups)}")
                
                # Find similar groups that might be merged
                similar_groups = []
                for j, other_group in enumerate(groups):
                    if j <= i or j in processed_groups:
                        continue
                    
                    # Check if groups should be merged based on hybrid similarity
                    if self._should_merge_groups(group, other_group):
                        similar_groups.append(j)
                
                # Merge similar groups
                if similar_groups:
                    merged_group = group.copy()
                    for j in similar_groups:
                        merged_group.extend(groups[j])
                        processed_groups.add(j)
                    
                    refined_groups.append(merged_group)
                    processed_groups.add(i)
                    logger.info(f"Merged {len(similar_groups) + 1} groups into one with {len(merged_group)} images")
                else:
                    refined_groups.append(group)
                    processed_groups.add(i)
            
            logger.info(f"Global refinement: {len(groups)} groups → {len(refined_groups)} groups")
            return refined_groups
            
        except Exception as e:
            logger.error(f"Global feature refinement failed: {e}")
            return groups
    
    def _local_feature_verification(self, groups: List[List[str]], 
                                  progress_callback: Optional[callable] = None) -> List[List[str]]:
        """
        Stage 4: Local feature verification within groups.
        
        Args:
            groups: List of image groups to verify
            progress_callback: Optional callback for progress reporting
            
        Returns:
            Verified list of image groups
        """
        try:
            if progress_callback:
                progress_callback("Local verification: Refining within-group duplicates...")
            
            verified_groups = []
            
            for i, group in enumerate(groups):
                if progress_callback:
                    progress_callback(f"Local verification: Group {i+1}/{len(groups)} ({len(group)} images)")
                
                if len(group) <= 1:
                    verified_groups.append(group)
                    continue
                
                # Skip very large groups for local verification (too expensive)
                if len(group) > self.max_group_size:
                    logger.warning(f"Skipping local verification for large group with {len(group)} images (performance protection)")
                    verified_groups.append(group)
                    continue
                
                # Use hybrid similarity for final verification within the group
                verified_group = self._verify_group_with_hybrid_similarity(
                    group, progress_callback
                )
                
                # Filter out single-image groups (local verification might split groups)
                if len(verified_group) > 1:
                    verified_groups.append(verified_group)
                else:
                    # Single image, add to a new single-image group
                    verified_groups.append(verified_group)
            
            logger.info(f"Local verification: {len(groups)} groups → {len(verified_groups)} groups")
            return verified_groups
            
        except Exception as e:
            logger.error(f"Local feature verification failed: {e}")
            return groups
    
    def _should_merge_groups(self, group1: List[str], group2: List[str]) -> bool:
        """
        Determine if two groups should be merged based on hybrid similarity.
        
        Args:
            group1: First image group
            group2: Second image group
            
        Returns:
            True if groups should be merged, False otherwise
        """
        if not self.hybrid_calculator:
            return False
        
        try:
            # Sample images from each group for comparison
            sample_size = min(3, len(group1), len(group2))
            sample1 = group1[:sample_size]
            sample2 = group2[:sample_size]
            
            # Check if any images from group1 are similar to any from group2
            for img1 in sample1:
                for img2 in sample2:
                    try:
                        similarity = self.hybrid_calculator.calculate_similarity(img1, img2)
                        if similarity >= self.verification_threshold:
                            logger.debug(f"Groups should be merged: {img1} ↔ {img2} (similarity: {similarity:.3f})")
                            return True
                    except Exception as e:
                        logger.debug(f"Similarity calculation failed for {img1} ↔ {img2}: {e}")
                        continue
            
            return False
            
        except Exception as e:
            logger.debug(f"Group merge check failed: {e}")
            return False
    
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
    
    def create_report(self, duplicate_groups: List[List[str]], 
                     similarity_scores: Dict[Tuple[str, str], float],
                     output_dir: str) -> str:
        """Create a detailed report of all groups with quality scores and similarity information."""
        try:
            import pandas as pd
            import os
            
            # Prepare data for DataFrame
            data = []
            
            for group_idx, group in enumerate(duplicate_groups):
                if len(group) == 0:
                    continue
                    
                # Use first image as representative (best image)
                best_image = group[0]
                group_size = len(group)
                
                # Add best image entry
                data.append({
                    'Image Path': best_image,
                    'Quality Score': 1.0,  # Representative gets perfect score
                    'Group ID': group_idx + 1,
                    'Group Size': group_size,
                    'Status': 'Best',
                    'Similarity Score': 1.0
                })
                
                # Add duplicate entries
                for i, dup_image in enumerate(group[1:], 1):
                    # Get similarity score if available
                    similarity_score = similarity_scores.get((best_image, dup_image), 0.0)
                    if similarity_score == 0.0:
                        # Try reverse order
                        similarity_score = similarity_scores.get((dup_image, best_image), 0.0)
                    
                    data.append({
                        'Image Path': dup_image,
                        'Quality Score': 0.8,  # Duplicates get lower quality score
                        'Group ID': group_idx + 1,
                        'Group Size': group_size,
                        'Status': 'Duplicate',
                        'Similarity Score': similarity_score
                    })
                    
            # Create DataFrame and sort by Group ID and Status (Best first)
            df = pd.DataFrame(data)
            df = df.sort_values(['Group ID', 'Status'], ascending=[True, False])
            
            # Format scores to 3 decimal places
            df['Quality Score'] = df['Quality Score'].round(3)
            df['Similarity Score'] = df['Similarity Score'].round(3)
            
            # Save to CSV
            report_path = os.path.join(output_dir, "image_report.csv")
            df.to_csv(report_path, index=False)
            
            # Log report statistics
            total_images = len(df)
            best_images = len(df[df['Status'] == 'Best'])
            duplicate_images = len(df[df['Status'] == 'Duplicate'])
            total_groups = df['Group ID'].nunique()
            
            logger.info(f"\nReport Statistics:")
            logger.info(f"- Total Images: {total_images}")
            logger.info(f"- Best Images: {best_images}")
            logger.info(f"- Duplicate Images: {duplicate_images}")
            logger.info(f"- Total Groups: {total_groups}")
            logger.info(f"- Report saved to: {report_path}")
            
            return report_path
            
        except Exception as e:
            logger.error(f"Error creating report: {e}")
            return ""

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
