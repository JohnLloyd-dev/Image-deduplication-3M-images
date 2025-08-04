import unittest
import os
import time
import numpy as np
import cv2
from src.pipeline.main_pipeline import DeduplicationPipeline
from src.utils.performance_monitor import PerformanceMonitor

class TestPerformance(unittest.TestCase):
    def setUp(self):
        self.pipeline = DeduplicationPipeline()
        self.test_dir = "test_images"
        self.output_dir = "test_results"
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate test images if they don't exist
        self._generate_test_images()
        
    def _generate_test_images(self):
        """Generate test images for testing."""
        # Create a simple test image
        test_image = os.path.join(self.test_dir, "test_image.jpg")
        if not os.path.exists(test_image):
            # Create a 100x100 RGB image with a gradient
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            for i in range(100):
                img[i, :, 0] = i  # Red channel gradient
                img[:, i, 1] = i  # Green channel gradient
            cv2.imwrite(test_image, img)
            
        # Create some duplicate images
        for i in range(5):
            dup_image = os.path.join(self.test_dir, f"duplicate_{i}.jpg")
            if not os.path.exists(dup_image):
                # Create slightly modified version of test image
                img = cv2.imread(test_image)
                # Add some noise
                noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)
                cv2.imwrite(dup_image, img)
                
    def test_pipeline_performance(self):
        """Test the overall pipeline performance."""
        # Skip if no test images
        if not os.listdir(self.test_dir):
            self.skipTest("No test images available")
            
        start_time = time.time()
        self.pipeline.run(self.test_dir, self.output_dir)
        total_time = time.time() - start_time
        
        # Get performance metrics
        metrics = self.pipeline.perf_monitor.get_metrics()
        
        # Log performance results
        print("\nPerformance Results:")
        print(f"Total processing time: {total_time:.2f} seconds")
        for stage, time_taken in metrics.items():
            print(f"{stage}: {time_taken:.2f} seconds")
            
        # Basic performance assertions
        self.assertGreater(len(metrics), 0)
        self.assertLess(total_time, 600)  # Should complete within 10 minutes
        
    def test_feature_extraction_performance(self):
        """Test feature extraction performance."""
        test_image = os.path.join(self.test_dir, "test_image.jpg")
        if not os.path.exists(test_image):
            self.skipTest("Test image not found")
            
        # Test color feature extraction
        start_time = time.time()
        color_features = self.pipeline.color_extractor.extract_color_features(test_image)
        color_time = time.time() - start_time
        
        self.assertIsNotNone(color_features)
        self.assertLess(color_time, 1.0)  # Should complete within 1 second
        
    def test_duplicate_detection_performance(self):
        """Test duplicate detection performance."""
        # Create properly formatted histograms
        hist1 = np.ones((32, 32), dtype=np.float32) / (32 * 32)
        hist2 = np.ones((32, 32), dtype=np.float32) / (32 * 32)
        
        # Create test features
        features1 = {
            'hsv_hist': hist1.reshape(-1),  # Flatten for storage
            'lab_stats': {'l_mean': 50, 'a_mean': 0, 'b_mean': 0}
        }
        features2 = {
            'hsv_hist': hist2.reshape(-1),  # Flatten for storage
            'lab_stats': {'l_mean': 51, 'a_mean': 1, 'b_mean': 1}
        }
        
        # Test similarity computation
        start_time = time.time()
        similarity = self.pipeline.color_extractor.compute_color_similarity(features1, features2)
        similarity_time = time.time() - start_time
        
        self.assertIsInstance(similarity, float)
        self.assertLess(similarity_time, 0.1)  # Should complete within 100ms

if __name__ == '__main__':
    unittest.main() 