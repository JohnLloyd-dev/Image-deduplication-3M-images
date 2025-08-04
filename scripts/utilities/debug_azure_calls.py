#!/usr/bin/env python3
"""
Debug Azure Calls - Find out why Azure is still being called
"""

import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.deduplication import HierarchicalDeduplicator
from modules.feature_cache import BoundedFeatureCache
import tempfile

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def test_color_similarity():
    """Test color similarity with test images to see if Azure is called."""
    
    # Create a temporary cache
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = BoundedFeatureCache(cache_dir=temp_dir, max_size=10)
        analyzer = HierarchicalDeduplicator(feature_cache=cache)
    
    # Test with obvious test image paths
    test_paths = [
        "test_image_001.jpg",
        "test_image_002.jpg",
        "fake_image_001.jpg",
        "/tmp/temp_image.jpg"
    ]
    
    print("🔍 Testing color similarity with test image paths...")
    
    for i, path1 in enumerate(test_paths):
        for j, path2 in enumerate(test_paths):
            if i < j:  # Only test unique pairs
                print(f"\n📊 Testing: {path1} vs {path2}")
                
                # Test the detection function first
                is_test1 = analyzer._is_test_image_path(path1)
                is_test2 = analyzer._is_test_image_path(path2)
                print(f"   Test detection: {path1} = {is_test1}, {path2} = {is_test2}")
                
                # Now test the similarity computation
                try:
                    similarity = analyzer.compute_color_similarity(path1, path2)
                    print(f"   Similarity: {similarity:.3f}")
                except Exception as e:
                    print(f"   Error: {e}")

if __name__ == "__main__":
    test_color_similarity()