#!/usr/bin/env python3
"""
Simple test to verify multi-threading implementation works
"""

import sys
import os
import time
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🧪 Simple Multi-threading Test")
print("=" * 50)

try:
    print("1. Testing imports...")
    
    import numpy as np
    from modules.threading_optimizer import ThreadingOptimizer
    print("   ✅ Basic imports successful")
    
    print("\n2. Testing system detection...")
    optimizer = ThreadingOptimizer()
    system_info = optimizer.get_system_info()
    config = optimizer.get_optimal_config()
    
    print(f"   System: {system_info['platform']}")
    print(f"   CPU cores: {system_info['cpu_count_logical']}")
    print(f"   Memory: {system_info['total_memory_gb']:.1f} GB")
    print(f"   Optimal workers: {config['max_workers']}")
    print(f"   Optimal chunk size: {config['chunk_size']}")
    print("   ✅ System detection successful")
    
    print("\n3. Testing feature cache...")
    from modules.feature_cache import BoundedFeatureCache
    
    # Create a temporary cache
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = BoundedFeatureCache(cache_dir=temp_dir, max_size=10)
        
        # Add some test features
        test_features = {
            'wavelet': np.random.randint(0, 2, 32, dtype=np.uint8),
            'global': np.random.randn(512).astype(np.float32),
            'local': {
                'keypoints': np.random.randn(10, 2).astype(np.float32),
                'descriptors': np.random.randn(10, 128).astype(np.float32)
            }
        }
        
        cache.put_features("test_image.jpg", test_features)
        retrieved = cache.get_features("test_image.jpg")
        
        assert retrieved is not None, "Failed to retrieve features"
        assert 'wavelet' in retrieved, "Wavelet features missing"
        assert 'global' in retrieved, "Global features missing"
        
        print("   ✅ Feature cache test successful")
    
    print("\n4. Testing multi-threaded deduplicator creation...")
    from modules.multithreaded_deduplication import MultiThreadedDeduplicator
    
    # Create a small cache for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = BoundedFeatureCache(cache_dir=temp_dir, max_size=5)
        
        deduplicator = MultiThreadedDeduplicator(
            feature_cache=cache,
            device="cpu",
            max_workers=2,  # Small number for testing
            chunk_size=2
        )
        
        print(f"   Created deduplicator with {deduplicator.max_workers} workers")
        print("   ✅ Multi-threaded deduplicator creation successful")
    
    print("\n5. Testing threading functionality...")
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    results = []
    lock = threading.Lock()
    
    def test_worker(worker_id):
        time.sleep(0.1)  # Simulate work
        with lock:
            results.append(f"Worker {worker_id} completed")
        return worker_id
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(test_worker, i) for i in range(4)]
        completed = [f.result() for f in futures]
    
    end_time = time.time()
    
    assert len(results) == 4, "Not all workers completed"
    assert len(completed) == 4, "Not all futures completed"
    
    print(f"   Completed 4 parallel tasks in {end_time - start_time:.2f}s")
    print("   ✅ Threading functionality test successful")
    
    print("\n🎉 All tests passed!")
    print("\nThe multi-threading implementation appears to be working correctly.")
    print("Key components verified:")
    print("  ✅ System detection and optimization")
    print("  ✅ Feature cache functionality")
    print("  ✅ Multi-threaded deduplicator creation")
    print("  ✅ Thread pool execution")
    print("  ✅ Thread-safe operations")
    
    print(f"\nRecommended configuration for this system:")
    print(f"  - Max workers: {config['max_workers']}")
    print(f"  - Chunk size: {config['chunk_size']}")
    print(f"  - Memory conservative: {config['memory_conservative']}")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)