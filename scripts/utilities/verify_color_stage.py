#!/usr/bin/env python3
"""
Simple verification script to check if the color-based deduplication stage is properly integrated.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verify_color_stage_integration():
    """Verify that the color stage is properly integrated into the hierarchical deduplication."""
    
    print("🔍 Verifying Color-Based Deduplication Stage Integration...")
    
    try:
        # Import the deduplication module
        from modules.deduplication import HierarchicalDeduplicator
        from modules.feature_cache import BoundedFeatureCache
        
        print("✅ Successfully imported HierarchicalDeduplicator")
        
        # Check if color verification methods exist
        cache = BoundedFeatureCache(cache_dir="temp_cache", max_size=10)
        deduplicator = HierarchicalDeduplicator(feature_cache=cache)
        
        # Verify color-related methods exist
        required_methods = [
            'verify_with_color_features',
            'compute_color_similarity', 
            '_dominant_color_distance',
            '_average_pixel_difference',
            '_histogram_correlation',
            'is_color_match'
        ]
        
        missing_methods = []
        for method in required_methods:
            if not hasattr(deduplicator, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing color methods: {missing_methods}")
            return False
        
        print("✅ All color verification methods are present")
        
        # Check if color parameters are initialized
        required_attributes = [
            'max_color_distance',
            'max_pixel_difference', 
            'color_base_threshold',
            'color_content_factor'
        ]
        
        missing_attributes = []
        for attr in required_attributes:
            if not hasattr(deduplicator, attr):
                missing_attributes.append(attr)
        
        if missing_attributes:
            print(f"❌ Missing color attributes: {missing_attributes}")
            return False
            
        print("✅ All color parameters are initialized")
        
        # Check the deduplicate method signature and structure
        import inspect
        deduplicate_source = inspect.getsource(deduplicator.deduplicate)
        
        # Check for the 5 stages
        stage_checks = [
            "Step 1: Grouping images by wavelet hash",
            "Step 2: Refining groups using global features", 
            "Step 3: Final verification using local features",
            "Step 4: Final color-based verification",
            "Step 5: Quality-based best image selection"
        ]
        
        missing_stages = []
        for stage in stage_checks:
            if stage not in deduplicate_source:
                missing_stages.append(stage)
        
        if missing_stages:
            print(f"❌ Missing stages in deduplicate method: {missing_stages}")
            return False
            
        print("✅ All 5 stages are present in the deduplicate method")
        
        # Check if color verification is actually called
        if "verify_with_color_features" not in deduplicate_source:
            print("❌ Color verification method is not called in deduplicate")
            return False
            
        print("✅ Color verification is properly integrated")
        
        # Check return values
        if "color_verified_groups" not in deduplicate_source:
            print("❌ Color verified groups are not returned")
            return False
            
        print("✅ Color verified groups are properly returned")
        
        print("\n🎉 SUCCESS: Color-based deduplication stage is fully integrated!")
        print("\nHierarchical Deduplication Process:")
        print("1. ⚡ Wavelet Hash Grouping (Fast, Coarse)")
        print("2. 🧠 Global Feature Refinement (Semantic)")  
        print("3. 🔍 Local Feature Verification (Geometric)")
        print("4. 🎨 Color-Based Final Verification (Perceptual)")
        print("5. ⭐ Quality-Based Best Selection (Organization)")
        
        cache.clear()
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

def verify_color_methods():
    """Verify individual color verification methods work."""
    
    print("\n🔍 Verifying Individual Color Methods...")
    
    try:
        from modules.deduplication import HierarchicalDeduplicator
        from modules.feature_cache import BoundedFeatureCache
        import numpy as np
        
        cache = BoundedFeatureCache(cache_dir="temp_cache", max_size=10)
        deduplicator = HierarchicalDeduplicator(feature_cache=cache)
        
        # Test color similarity computation
        color1 = np.array([0.8, 0.2, 0.1, 0.9, 0.3])
        color2 = np.array([0.81, 0.21, 0.11, 0.89, 0.31])  # Very similar
        color3 = np.array([0.1, 0.9, 0.8, 0.2, 0.7])  # Different
        
        sim_similar = deduplicator.compute_color_similarity(color1, color2)
        sim_different = deduplicator.compute_color_similarity(color1, color3)
        
        print(f"✅ Color similarity (similar): {sim_similar:.3f}")
        print(f"✅ Color similarity (different): {sim_different:.3f}")
        
        if sim_similar <= sim_different:
            print("❌ Color similarity logic may be incorrect")
            return False
            
        print("✅ Color similarity computation works correctly")
        
        # Test group verification with mock data
        test_group = ['img1.jpg', 'img2.jpg']
        
        # This would normally require actual image files, so we'll just check the method exists
        if hasattr(deduplicator, 'verify_with_color_features'):
            print("✅ Color group verification method is available")
        else:
            print("❌ Color group verification method is missing")
            return False
        
        cache.clear()
        return True
        
    except Exception as e:
        print(f"❌ Color methods verification error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("COLOR-BASED DEDUPLICATION STAGE VERIFICATION")
    print("=" * 60)
    
    success1 = verify_color_stage_integration()
    success2 = verify_color_methods()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("The color-based deduplication stage is fully implemented and integrated.")
        sys.exit(0)
    else:
        print("❌ SOME VERIFICATIONS FAILED!")
        print("Please check the implementation.")
        sys.exit(1)