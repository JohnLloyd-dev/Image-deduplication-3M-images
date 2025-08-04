import unittest
import numpy as np
from src.feature_extraction.feature_fusion import FeatureFuser

class TestFeatureFusion(unittest.TestCase):
    def setUp(self):
        self.fuser = FeatureFuser()
        
    def test_normalize_features(self):
        # Test feature normalization with multiple samples
        features = {
            'global': np.array([[1.0], [2.0], [3.0]]),  # Reshape to 2D array
            'local': np.array([[4.0], [5.0], [6.0]]),   # Reshape to 2D array
            'color': np.array([0.1, 0.2, 0.3])
        }
        
        normalized = self.fuser.normalize_features(features)
        
        # Check that color features remain unchanged
        np.testing.assert_array_equal(normalized['color'], features['color'])
        
        # Check that other features are normalized
        for feature_type in ['global', 'local']:
            self.assertAlmostEqual(np.mean(normalized[feature_type]), 0, places=5)
            self.assertAlmostEqual(np.std(normalized[feature_type]), 1, places=5)
            
    def test_fuse_features(self):
        # Test feature fusion
        features = {
            'global': np.array([1.0, 2.0, 3.0]),
            'local': np.array([4.0, 5.0, 6.0]),
            'color': np.array([0.1, 0.2, 0.3])
        }
        
        fused = self.fuser.fuse_features(features)
        self.assertIsNotNone(fused)
        self.assertEqual(len(fused), 9)  # 3 features * 3 dimensions
        
    def test_update_weights(self):
        # Test weight updates
        new_weights = {
            'global': 0.5,
            'local': 0.3,
            'color': 0.2
        }
        
        self.fuser.update_weights(new_weights)
        self.assertEqual(self.fuser.feature_weights, new_weights)
        
    def test_invalid_weights(self):
        # Test invalid weight updates
        invalid_weights = {
            'global': 0.5,
            'local': 0.3,
            'invalid': 0.2
        }
        
        with self.assertRaises(ValueError):
            self.fuser.update_weights(invalid_weights)
            
        invalid_sum = {
            'global': 0.5,
            'local': 0.5,
            'color': 0.5
        }
        
        with self.assertRaises(ValueError):
            self.fuser.update_weights(invalid_sum)

if __name__ == '__main__':
    unittest.main() 