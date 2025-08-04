#!/usr/bin/env python3
"""
Azure Image Loader for Deduplication Pipeline

This module provides on-demand image loading from Azure Blob Storage
for the deduplication pipeline. Images are downloaded only when needed
and immediately discarded after processing to save memory.
"""

import logging
import numpy as np
import cv2
from typing import Optional, Dict, Any
from .azure_utils import AzureBlobManager, SAS_URL
import threading
import time

logger = logging.getLogger(__name__)


class AzureImageLoader:
    """
    On-demand image loader for Azure Blob Storage.
    
    This class provides thread-safe, memory-efficient image loading
    for the deduplication pipeline. Images are downloaded fresh each
    time they're needed and not cached locally.
    """
    
    def __init__(self, sas_url: str = None, max_retries: int = 3):
        """
        Initialize Azure image loader.
        
        Args:
            sas_url: Azure SAS URL. If None, uses default from azure_utils
            max_retries: Maximum number of retry attempts for failed downloads
        """
        self.sas_url = sas_url or SAS_URL
        self.max_retries = max_retries
        self._blob_manager = None
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'images_downloaded': 0,
            'download_failures': 0,
            'total_download_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    @property
    def blob_manager(self) -> AzureBlobManager:
        """Get or create blob manager (thread-safe lazy initialization)."""
        if self._blob_manager is None:
            with self._lock:
                if self._blob_manager is None:
                    self._blob_manager = AzureBlobManager(
                        sas_url=self.sas_url,
                        max_retries=self.max_retries
                    )
        return self._blob_manager
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an image from Azure Blob Storage.
        
        Args:
            image_path: Path/name of the image in Azure blob storage
            
        Returns:
            numpy.ndarray: Image as OpenCV format (BGR) or None if failed
        """
        start_time = time.time()
        
        try:
            # Download image from Azure
            image = self.blob_manager.download_as_cv2(image_path)
            
            download_time = time.time() - start_time
            
            if image is not None:
                self.stats['images_downloaded'] += 1
                self.stats['total_download_time'] += download_time
                logger.debug(f"Successfully downloaded {image_path} in {download_time:.2f}s")
                return image
            else:
                self.stats['download_failures'] += 1
                logger.warning(f"Failed to download {image_path}")
                return None
                
        except Exception as e:
            download_time = time.time() - start_time
            self.stats['download_failures'] += 1
            self.stats['total_download_time'] += download_time
            logger.error(f"Error downloading {image_path}: {e}")
            return None
    
    def load_images_batch(self, image_paths: list) -> Dict[str, Optional[np.ndarray]]:
        """
        Load multiple images in batch (still downloads one by one for memory efficiency).
        
        Args:
            image_paths: List of image paths to download
            
        Returns:
            Dict mapping image_path -> image array (or None if failed)
        """
        results = {}
        
        for path in image_paths:
            results[path] = self.load_image(path)
            
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get download statistics."""
        stats = self.stats.copy()
        
        if stats['images_downloaded'] > 0:
            stats['avg_download_time'] = stats['total_download_time'] / stats['images_downloaded']
        else:
            stats['avg_download_time'] = 0.0
            
        stats['success_rate'] = (
            stats['images_downloaded'] / 
            (stats['images_downloaded'] + stats['download_failures'])
            if (stats['images_downloaded'] + stats['download_failures']) > 0 
            else 0.0
        )
        
        return stats
    
    def reset_stats(self):
        """Reset download statistics."""
        self.stats = {
            'images_downloaded': 0,
            'download_failures': 0,
            'total_download_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }


# Global instance for use across the deduplication pipeline
_global_loader = None
_global_loader_lock = threading.Lock()


def get_azure_image_loader(sas_url: str = None) -> AzureImageLoader:
    """
    Get global Azure image loader instance (singleton pattern).
    
    Args:
        sas_url: Azure SAS URL. If None, uses default
        
    Returns:
        AzureImageLoader instance
    """
    global _global_loader
    
    if _global_loader is None:
        with _global_loader_lock:
            if _global_loader is None:
                _global_loader = AzureImageLoader(sas_url=sas_url)
    
    return _global_loader


def load_image_from_azure(image_path: str) -> Optional[np.ndarray]:
    """
    Convenience function to load a single image from Azure.
    
    Args:
        image_path: Path/name of the image in Azure blob storage
        
    Returns:
        numpy.ndarray: Image as OpenCV format (BGR) or None if failed
    """
    loader = get_azure_image_loader()
    return loader.load_image(image_path)


def load_images_from_azure(image_paths: list) -> Dict[str, Optional[np.ndarray]]:
    """
    Convenience function to load multiple images from Azure.
    
    Args:
        image_paths: List of image paths to download
        
    Returns:
        Dict mapping image_path -> image array (or None if failed)
    """
    loader = get_azure_image_loader()
    return loader.load_images_batch(image_paths)