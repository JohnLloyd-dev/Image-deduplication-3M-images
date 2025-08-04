import os
import json
import gzip
import numpy as np
import threading
from typing import Dict, Optional, Any
from io import BytesIO
import logging
from collections import defaultdict
import pickle

logger = logging.getLogger(__name__)

class BoundedFeatureCache:
    """Bounded feature cache for storing and retrieving image features."""
    
    def __init__(self, cache_dir: str = "features", max_size: int = 10000):
        """
        Initialize the feature cache.
        
        Args:
            cache_dir: Directory to store feature files
            max_size: Maximum number of features to keep in memory
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.memory_cache = {}
        self.lock = threading.Lock()
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"Initialized BoundedFeatureCache with cache_dir={cache_dir}, max_size={max_size}")
    
    def put_features(self, azure_path: str, features: Dict[str, Any]) -> bool:
        """
        Store features for an Azure blob path.
        
        Args:
            azure_path: Azure blob path (e.g., "folder/image.jpg")
            features: Dictionary containing extracted features
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert Azure path to safe filename
            safe_filename = self._azure_path_to_filename(azure_path)
            feature_path = os.path.join(self.cache_dir, safe_filename)
            
            # Compress and save features
            compressed_data = self._compress_features(features)
            
            with open(feature_path, 'wb') as f:
                f.write(compressed_data)
            
            # Add to memory cache
            with self.lock:
                if len(self.memory_cache) >= self.max_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self.memory_cache))
                    del self.memory_cache[oldest_key]
                
                self.memory_cache[azure_path] = features
            
            logger.debug(f"Stored features for {azure_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing features for {azure_path}: {str(e)}")
            return False
    
    def get_features(self, azure_path: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve features for an Azure blob path.
        
        Args:
            azure_path: Azure blob path
            
        Returns:
            Features dictionary if found, None otherwise
        """
        try:
            # Check memory cache first
            with self.lock:
                if azure_path in self.memory_cache:
                    return self.memory_cache[azure_path]
            
            # Check disk cache
            safe_filename = self._azure_path_to_filename(azure_path)
            feature_path = os.path.join(self.cache_dir, safe_filename)
            
            if os.path.exists(feature_path):
                with open(feature_path, 'rb') as f:
                    compressed_data = f.read()
                
                features = self._decompress_features(compressed_data)
                
                # Add to memory cache
                with self.lock:
                    if len(self.memory_cache) >= self.max_size:
                        oldest_key = next(iter(self.memory_cache))
                        del self.memory_cache[oldest_key]
                    self.memory_cache[azure_path] = features
                
                return features
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving features for {azure_path}: {str(e)}")
            return None
    
    def get_all_features(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all features from disk cache.
        
        Returns:
            Dictionary mapping Azure paths to feature dictionaries
        """
        features = {}
        
        try:
            # Scan cache directory
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.npy.gz'):
                    try:
                        # Convert filename back to Azure path
                        azure_path = self._filename_to_azure_path(filename)
                        feature_path = os.path.join(self.cache_dir, filename)
                        
                        with open(feature_path, 'rb') as f:
                            compressed_data = f.read()
                        
                        features[azure_path] = self._decompress_features(compressed_data)
                        
                    except Exception as e:
                        logger.warning(f"Error loading features from {filename}: {str(e)}")
                        continue
            
            logger.info(f"Loaded {len(features)} features from disk cache")
            return features
            
        except Exception as e:
            logger.error(f"Error loading all features: {str(e)}")
            return {}
    
    def get_feature_path(self, azure_path: str) -> str:
        """
        Get the file path for a feature.
        
        Args:
            azure_path: Azure blob path
            
        Returns:
            Full path to the feature file
        """
        safe_filename = self._azure_path_to_filename(azure_path)
        return os.path.join(self.cache_dir, safe_filename)
    
    def has_features(self, azure_path: str) -> bool:
        """
        Check if features exist for an Azure blob path.
        
        Args:
            azure_path: Azure blob path
            
        Returns:
            True if features exist, False otherwise
        """
        # Check memory cache first
        with self.lock:
            if azure_path in self.memory_cache:
                return True
        
        # Check disk cache
        safe_filename = self._azure_path_to_filename(azure_path)
        feature_path = os.path.join(self.cache_dir, safe_filename)
        return os.path.exists(feature_path)
    
    def clear(self):
        """Clear all cached features."""
        with self.lock:
            self.memory_cache.clear()
        
        # Clear disk cache
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.npy.gz'):
                    os.remove(os.path.join(self.cache_dir, filename))
            logger.info("Cleared all cached features")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    def _azure_path_to_filename(self, azure_path: str) -> str:
        """Convert Azure blob path to safe filename."""
        # Replace slashes with underscores and add extension
        safe_name = azure_path.replace('/', '_').replace('\\', '_')
        return f"{safe_name}.npy.gz"
    
    def _filename_to_azure_path(self, filename: str) -> str:
        """Convert safe filename back to Azure blob path."""
        # Remove extension and replace underscores with slashes
        base_name = filename.replace('.npy.gz', '')
        # This is a simplified conversion - in practice, you might need a more sophisticated mapping
        return base_name.replace('_', '/')
    
    def _compress_features(self, features: Dict[str, Any]) -> bytes:
        """Compress features using gzip and pickle."""
        try:
            # Serialize features
            serialized = pickle.dumps(features)
            
            # Compress with gzip
            buffer = BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
                f.write(serialized)
            
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error compressing features: {str(e)}")
            raise
    
    def _decompress_features(self, compressed_data: bytes) -> Dict[str, Any]:
        """Decompress features using gzip and pickle."""
        try:
            # Decompress with gzip
            buffer = BytesIO(compressed_data)
            with gzip.GzipFile(fileobj=buffer, mode='rb') as f:
                serialized = f.read()
            
            # Deserialize features
            return pickle.loads(serialized)
            
        except Exception as e:
            logger.error(f"Error decompressing features: {str(e)}")
            raise

def load_compressed_features(feature_cache: BoundedFeatureCache, azure_path: str) -> Optional[Dict[str, Any]]:
    """
    Load compressed features from the feature cache.
    
    Args:
        feature_cache: BoundedFeatureCache instance
        azure_path: Azure blob path
        
    Returns:
        Features dictionary if found, None otherwise
    """
    return feature_cache.get_features(azure_path) 