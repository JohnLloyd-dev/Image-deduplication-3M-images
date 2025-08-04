import unittest
import os
import numpy as np
import cv2
from src.feature_extraction.color_features import ColorFeatureExtractor

class TestColorFeatures(unittest.TestCase):
    def setUp(self):
        self.extractor = ColorFeatureExtractor(hsv_bins=32, lab_weight=2.0)
        # Create a test image directory if it doesn't exist
        self.test_dir = "test_images"
        os.makedirs(self.test_dir, exist_ok=True)
        
    def test_hsv_histogram(self):
        # Test with a sample image
        test_image = os.path.join(self.test_dir, "test_image.jpg")
        if not os.path.exists(test_image):
            self.skipTest("Test image not found")
            
        hist = self.extractor.compute_hsv_histogram(test_image)
        self.assertIsNotNone(hist)
        self.assertEqual(hist.shape[0], 32 * 32)  # hsv_bins * hsv_bins
        self.assertTrue(np.all(hist >= 0))
        self.assertTrue(np.all(hist <= 1))
        
    def test_lab_color_stats(self):
        test_image = os.path.join(self.test_dir, "test_image.jpg")
        if not os.path.exists(test_image):
            self.skipTest("Test image not found")
            
        stats = self.extractor.compute_lab_color_stats(test_image)
        self.assertIsNotNone(stats)
        self.assertIn('l_mean', stats)
        self.assertIn('a_mean', stats)
        self.assertIn('b_mean', stats)
        
    def test_color_similarity(self):
        # Create properly formatted histograms for OpenCV
        hist1 = np.ones((32, 32), dtype=np.float32) / (32 * 32)
        hist2 = np.ones((32, 32), dtype=np.float32) / (32 * 32)
        
        features1 = {
            'hsv_hist': hist1.reshape(-1),  # Flatten for storage
            'lab_stats': {'l_mean': 50, 'a_mean': 0, 'b_mean': 0}
        }
        features2 = {
            'hsv_hist': hist2.reshape(-1),  # Flatten for storage
            'lab_stats': {'l_mean': 50, 'a_mean': 0, 'b_mean': 0}
        }
        
        similarity = self.extractor.compute_color_similarity(features1, features2)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
    def test_invalid_image(self):
        # Test with non-existent image
        hist = self.extractor.compute_hsv_histogram("nonexistent.jpg")
        self.assertIsNone(hist)
        
        stats = self.extractor.compute_lab_color_stats("nonexistent.jpg")
        self.assertIsNone(stats)

if __name__ == '__main__':
    unittest.main() 