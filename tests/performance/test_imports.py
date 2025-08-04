#!/usr/bin/env python3
"""
Test imports for multi-threading implementation
"""

import sys
import os

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

try:
    print("Testing basic imports...")
    import numpy as np
    print("✅ numpy imported successfully")
    
    import threading
    print("✅ threading imported successfully")
    
    from concurrent.futures import ThreadPoolExecutor
    print("✅ ThreadPoolExecutor imported successfully")
    
    print("\nTesting project imports...")
    
    # Add project root to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from modules.feature_cache import BoundedFeatureCache
    print("✅ BoundedFeatureCache imported successfully")
    
    from modules.memory_efficient_deduplication import MemoryEfficientDeduplicator
    print("✅ MemoryEfficientDeduplicator imported successfully")
    
    from modules.multithreaded_deduplication import MultiThreadedDeduplicator
    print("✅ MultiThreadedDeduplicator imported successfully")
    
    from modules.threading_optimizer import ThreadingOptimizer
    print("✅ ThreadingOptimizer imported successfully")
    
    print("\n🎉 All imports successful!")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    
    optimizer = ThreadingOptimizer()
    system_info = optimizer.get_system_info()
    config = optimizer.get_optimal_config()
    
    print(f"System detected: {system_info['cpu_count_logical']} cores, {system_info['total_memory_gb']:.1f}GB RAM")
    print(f"Optimal config: {config['max_workers']} workers, {config['chunk_size']} chunk size")
    
    print("✅ Basic functionality test passed!")
    
except Exception as e:
    print(f"❌ Import/functionality test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)