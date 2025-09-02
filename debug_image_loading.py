#!/usr/bin/env python3
"""
Debug script to test image loading from Azure paths.
"""

import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.azure_utils import list_blobs_from_azure, download_blob_to_memory, SAS_URL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_azure_image_loading():
    """Test Azure image loading with a few sample paths."""
    
    print("🔍 Testing Azure image loading...")
    
    # Test 1: List some blobs to see what's available
    print("\n📋 Listing available blobs...")
    try:
        blobs = list_blobs_from_azure(SAS_URL, use_cache=False, force_refresh=True)
        print(f"Found {len(blobs)} total blobs")
        
        # Show first 10 blobs
        for i, blob in enumerate(blobs[:10]):
            print(f"  {i+1}. {blob}")
            
        # Test 2: Try to download a few images
        print(f"\n📥 Testing image downloads...")
        test_count = 0
        success_count = 0
        
        for blob in blobs[:5]:  # Test first 5 images
            test_count += 1
            print(f"\n  Testing {blob}...")
            
            try:
                image_data = download_blob_to_memory(blob, SAS_URL)
                if image_data:
                    print(f"    ✅ Success: {len(image_data)} bytes")
                    success_count += 1
                else:
                    print(f"    ❌ Failed: No data returned")
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                
        print(f"\n📊 Results: {success_count}/{test_count} images loaded successfully")
        
    except Exception as e:
        print(f"❌ Error listing blobs: {e}")

if __name__ == "__main__":
    test_azure_image_loading()

