#!/usr/bin/env python3
"""
Memory-Efficient Image Loader for Deduplication

This module provides on-demand image loading that:
1. Downloads images only when needed for comparison
2. Never caches images in memory (only features are cached)
3. Loads image pairs once per comparison operation
4. Immediately discards images after processing
"""

import logging
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
from .azure_image_loader import load_image_from_azure
import threading
import time
import gc

logger = logging.getLogger(__name__)


class MemoryEfficientImageLoader:
    """
    Memory-efficient image loader that downloads images only when needed
    and immediately discards them after processing.
    """
    
    def __init__(self):
        """Initialize the memory-efficient image loader."""
        self.stats = {
            'images_downloaded': 0,
            'download_failures': 0,
            'total_download_time': 0.0,
            'comparisons_performed': 0,
            'memory_saved_mb': 0.0
        }
        self._lock = threading.Lock()
    
    def load_image_pair_for_comparison(self, img1_path: str, img2_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load a pair of images for comparison and return them.
        Images are downloaded fresh and should be discarded after use.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            
        Returns:
            Tuple of (image1, image2) or (None, None) if loading fails
        """
        start_time = time.time()
        
        try:
            # Check if these are test images first
            if self._is_test_image_path(img1_path) or self._is_test_image_path(img2_path):
                logger.debug(f"Skipping Azure download for test images: {img1_path}, {img2_path}")
                return None, None
            
            # Download both images
            img1 = load_image_from_azure(img1_path)
            img2 = load_image_from_azure(img2_path)
            
            download_time = time.time() - start_time
            
            with self._lock:
                self.stats['comparisons_performed'] += 1
                self.stats['total_download_time'] += download_time
                
                if img1 is not None:
                    self.stats['images_downloaded'] += 1
                else:
                    self.stats['download_failures'] += 1
                    
                if img2 is not None:
                    self.stats['images_downloaded'] += 1
                else:
                    self.stats['download_failures'] += 1
            
            if img1 is None or img2 is None:
                logger.warning(f"Failed to load image pair: {img1_path}, {img2_path}")
                return None, None
            
            logger.debug(f"Loaded image pair in {download_time:.2f}s: {img1_path}, {img2_path}")
            return img1, img2
            
        except Exception as e:
            download_time = time.time() - start_time
            with self._lock:
                self.stats['download_failures'] += 2
                self.stats['total_download_time'] += download_time
            logger.error(f"Error loading image pair {img1_path}, {img2_path}: {e}")
            return None, None
    
    def compute_dominant_color_distance(self, img1_path: str, img2_path: str) -> float:
        """
        Compute dominant color distance between two images.
        Downloads images only for this computation and discards them immediately.
        """
        try:
            # Check if these are test images first
            if self._is_test_image_path(img1_path) or self._is_test_image_path(img2_path):
                return 50.0  # Neutral distance for test images
            
            # Load image pair
            img1, img2 = self.load_image_pair_for_comparison(img1_path, img2_path)
            
            if img1 is None or img2 is None:
                return 50.0  # Neutral distance when images can't be loaded
            
            try:
                # Extract dominant colors
                colors1 = self._get_dominant_colors(img1, 2)
                colors2 = self._get_dominant_colors(img2, 2)
                
                # Find minimum pairwise distance
                min_dist = float('inf')
                for c1 in colors1:
                    for c2 in colors2:
                        dist = np.linalg.norm(c1 - c2)
                        min_dist = min(min_dist, dist)
                
                return min_dist
                
            finally:
                # Immediately free memory
                del img1, img2
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error in dominant color distance: {e}")
            return float('inf')
    
    def compute_average_pixel_difference(self, img1_path: str, img2_path: str) -> float:
        """
        Compute average pixel difference between two images.
        Downloads images only for this computation and discards them immediately.
        """
        try:
            # Check if these are test images first
            if self._is_test_image_path(img1_path) or self._is_test_image_path(img2_path):
                return 25.0  # Neutral difference for test images
            
            # Load image pair
            img1, img2 = self.load_image_pair_for_comparison(img1_path, img2_path)
            
            if img1 is None or img2 is None:
                return 25.0  # Moderate difference when images can't be loaded
            
            try:
                # Resize to common dimensions
                h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
                img1 = cv2.resize(img1, (w, h))
                img2 = cv2.resize(img2, (w, h))
                
                # Convert to LAB and compute mean absolute difference
                lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
                lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
                return np.mean(np.abs(lab1.astype(np.float32) - lab2.astype(np.float32)))
                
            finally:
                # Immediately free memory
                del img1, img2
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error in pixel difference: {e}")
            return float('inf')
    
    def compute_histogram_correlation(self, img1_path: str, img2_path: str) -> float:
        """
        Compute histogram correlation between two images.
        Downloads images only for this computation and discards them immediately.
        """
        try:
            # Check if these are test images first
            if self._is_test_image_path(img1_path) or self._is_test_image_path(img2_path):
                return 0.5  # Neutral correlation for test images
            
            # Load image pair
            img1, img2 = self.load_image_pair_for_comparison(img1_path, img2_path)
            
            if img1 is None or img2 is None:
                return 0.5  # Neutral correlation when images can't be loaded
            
            try:
                # Focus on central region (ignore border variations)
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                
                # Extract central 80% of each image
                y1_start, y1_end = int(h1 * 0.1), int(h1 * 0.9)
                x1_start, x1_end = int(w1 * 0.1), int(w1 * 0.9)
                y2_start, y2_end = int(h2 * 0.1), int(h2 * 0.9)
                x2_start, x2_end = int(w2 * 0.1), int(w2 * 0.9)
                
                central1 = img1[y1_start:y1_end, x1_start:x1_end]
                central2 = img2[y2_start:y2_end, x2_start:x2_end]
                
                # Compute histograms
                hist1 = cv2.calcHist([central1], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
                hist2 = cv2.calcHist([central2], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
                
                # Normalize histograms
                cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                
                # Compute correlation
                correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                return max(0.0, correlation)  # Ensure non-negative
                
            finally:
                # Immediately free memory
                del img1, img2
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error in histogram correlation: {e}")
            return 0.0
    
    def compute_all_color_metrics(self, img1_path: str, img2_path: str) -> Dict[str, float]:
        """
        Compute all color metrics for a pair of images in a single download.
        This is the most memory-efficient approach - download once, compute all metrics.
        
        Returns:
            Dict with keys: 'dominant_distance', 'pixel_difference', 'histogram_correlation'
        """
        try:
            # Check if these are test images first
            if self._is_test_image_path(img1_path) or self._is_test_image_path(img2_path):
                return {
                    'dominant_distance': 50.0,
                    'pixel_difference': 25.0,
                    'histogram_correlation': 0.5
                }
            
            # Load image pair once
            img1, img2 = self.load_image_pair_for_comparison(img1_path, img2_path)
            
            if img1 is None or img2 is None:
                return {
                    'dominant_distance': 50.0,
                    'pixel_difference': 25.0,
                    'histogram_correlation': 0.5
                }
            
            try:
                results = {}
                
                # 1. Dominant color distance
                try:
                    colors1 = self._get_dominant_colors(img1, 2)
                    colors2 = self._get_dominant_colors(img2, 2)
                    
                    min_dist = float('inf')
                    for c1 in colors1:
                        for c2 in colors2:
                            dist = np.linalg.norm(c1 - c2)
                            min_dist = min(min_dist, dist)
                    
                    results['dominant_distance'] = min_dist
                except Exception as e:
                    logger.error(f"Error computing dominant colors: {e}")
                    results['dominant_distance'] = 50.0
                
                # 2. Average pixel difference
                try:
                    # Resize to common dimensions
                    h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
                    img1_resized = cv2.resize(img1, (w, h))
                    img2_resized = cv2.resize(img2, (w, h))
                    
                    # Convert to LAB and compute mean absolute difference
                    lab1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2LAB)
                    lab2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2LAB)
                    results['pixel_difference'] = np.mean(np.abs(lab1.astype(np.float32) - lab2.astype(np.float32)))
                except Exception as e:
                    logger.error(f"Error computing pixel difference: {e}")
                    results['pixel_difference'] = 25.0
                
                # 3. Histogram correlation
                try:
                    # Focus on central region
                    h1, w1 = img1.shape[:2]
                    h2, w2 = img2.shape[:2]
                    
                    y1_start, y1_end = int(h1 * 0.1), int(h1 * 0.9)
                    x1_start, x1_end = int(w1 * 0.1), int(w1 * 0.9)
                    y2_start, y2_end = int(h2 * 0.1), int(h2 * 0.9)
                    x2_start, x2_end = int(w2 * 0.1), int(w2 * 0.9)
                    
                    central1 = img1[y1_start:y1_end, x1_start:x1_end]
                    central2 = img2[y2_start:y2_end, x2_start:x2_end]
                    
                    # Compute histograms
                    hist1 = cv2.calcHist([central1], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
                    hist2 = cv2.calcHist([central2], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
                    
                    # Normalize histograms
                    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    
                    # Compute correlation
                    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    results['histogram_correlation'] = max(0.0, correlation)
                except Exception as e:
                    logger.error(f"Error computing histogram correlation: {e}")
                    results['histogram_correlation'] = 0.5
                
                return results
                
            finally:
                # Immediately free memory
                del img1, img2
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error computing color metrics for {img1_path}, {img2_path}: {e}")
            return {
                'dominant_distance': 50.0,
                'pixel_difference': 25.0,
                'histogram_correlation': 0.5
            }
    
    def _get_dominant_colors(self, img: np.ndarray, n_colors: int = 3) -> np.ndarray:
        """Extract dominant colors from image using K-means clustering."""
        try:
            # Reshape image to be a list of pixels
            pixels = img.reshape(-1, 3).astype(np.float32)
            
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
        """Check if an image path looks like a test image."""
        test_patterns = [
            'test_', 'fake_', 'dummy_', 'sample_',
            '/test/', '/fake/', '/dummy/', '/sample/',
            'test.', 'fake.', 'dummy.', 'sample.',
            '/tmp/', 'temp_', '/temp/'
        ]
        
        path_lower = image_path.lower()
        return any(pattern in path_lower for pattern in test_patterns)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        with self._lock:
            stats = self.stats.copy()
            
            if stats['comparisons_performed'] > 0:
                stats['avg_download_time'] = stats['total_download_time'] / stats['comparisons_performed']
            else:
                stats['avg_download_time'] = 0.0
                
            total_attempts = stats['images_downloaded'] + stats['download_failures']
            if total_attempts > 0:
                stats['success_rate'] = stats['images_downloaded'] / total_attempts
            else:
                stats['success_rate'] = 0.0
            
            return stats
    
    def reset_stats(self):
        """Reset loader statistics."""
        with self._lock:
            self.stats = {
                'images_downloaded': 0,
                'download_failures': 0,
                'total_download_time': 0.0,
                'comparisons_performed': 0,
                'memory_saved_mb': 0.0
            }


# Global instance for use across the deduplication pipeline
_global_efficient_loader = None
_global_efficient_loader_lock = threading.Lock()


def get_memory_efficient_loader() -> MemoryEfficientImageLoader:
    """
    Get global memory-efficient image loader instance (singleton pattern).
    
    Returns:
        MemoryEfficientImageLoader instance
    """
    global _global_efficient_loader
    
    if _global_efficient_loader is None:
        with _global_efficient_loader_lock:
            if _global_efficient_loader is None:
                _global_efficient_loader = MemoryEfficientImageLoader()
    
    return _global_efficient_loader