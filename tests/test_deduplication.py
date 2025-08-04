import os
import sys
import logging
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import torch
from typing import List, Dict, Tuple

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from modules.feature_extraction import FeatureExtractor
from modules.deduplication import Deduplicator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deduplication.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestDeduplication:
    def __init__(self):
        """Initialize test environment."""
        self.test_dir = "test_images"
        self.results_dir = "test_results"
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize feature extractor and deduplicator
        self.feature_extractor = FeatureExtractor()
        self.deduplicator = Deduplicator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
    def validate_image(self, img_path: str) -> bool:
        """Validate image file."""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                return False
                
            # Check image dimensions
            if img.shape[0] < 32 or img.shape[1] < 32:
                logger.warning(f"Image too small: {img_path}")
                return False
                
            # Check image channels
            if len(img.shape) != 3 or img.shape[2] != 3:
                logger.warning(f"Invalid image format: {img_path}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating image {img_path}: {e}")
            return False
            
    def extract_features_batch(self, image_files: List[Path], batch_size: int = 32) -> Dict[str, Dict]:
        """Extract features in batches for better memory management."""
        features = {}
        total_batches = (len(image_files) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(image_files))
            batch_files = image_files[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
            
            for img_path in tqdm(batch_files, desc=f"Batch {batch_idx + 1}"):
                try:
                    # Validate image
                    if not self.validate_image(img_path):
                        continue
                        
                    # Read and preprocess image
                    img = cv2.imread(str(img_path))
                    
                    # Extract features
                    img_features = self.feature_extractor.extract_features(img)
                    
                    # Validate extracted features
                    if all(v is not None for v in img_features.values()):
                        features[str(img_path)] = img_features
                    else:
                        logger.warning(f"Failed to extract some features from: {img_path}")
                        
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    continue
                    
            # Clear CUDA cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return features
        
    def run_deduplication(self):
        """Run deduplication on test images."""
        try:
            # Get all image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files.extend(list(Path(self.test_dir).glob(f'*{ext}')))
            
            if not image_files:
                logger.error(f"No images found in {self.test_dir}")
                return False
                
            logger.info(f"Found {len(image_files)} images")
            
            # Extract features in batches
            logger.info("Extracting features...")
            features = self.extract_features_batch(image_files)
            
            if not features:
                logger.error("No features extracted from images")
                return False
                
            logger.info(f"Successfully extracted features from {len(features)} images")
            
            # Run deduplication
            logger.info("Running deduplication...")
            duplicate_groups, similarity_scores = self.deduplicator.deduplicate(
                list(features.keys()),
                features,
                self.results_dir
            )
            
            if not duplicate_groups:
                logger.info("No duplicates found")
                return True
                
            # Validate results
            total_duplicates = sum(len(group) - 1 for group in duplicate_groups)
            logger.info(f"Found {len(duplicate_groups)} groups with {total_duplicates} duplicates")
            
            # Create summary report
            self.create_summary_report(duplicate_groups, similarity_scores)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in deduplication: {e}")
            return False
            
        finally:
            # Clean up
            self.feature_extractor.release()
            self.deduplicator.release()
            
    def create_summary_report(self, duplicate_groups: List[List[str]], 
                            similarity_scores: Dict[Tuple[str, str], float]):
        """Create a summary report of the deduplication results."""
        try:
            # Prepare data for DataFrame
            data = []
            for group_idx, group in enumerate(duplicate_groups):
                best_image = group[0]  # First image is the best one
                for dup_idx, dup_image in enumerate(group[1:], 1):
                    similarity = similarity_scores.get((best_image, dup_image), 0.0)
                    data.append({
                        'Group': group_idx + 1,
                        'Best Image': os.path.basename(best_image),
                        'Duplicate Image': os.path.basename(dup_image),
                        'Similarity Score': f"{similarity:.3f}",
                        'Best Path': best_image,
                        'Duplicate Path': dup_image
                    })
                    
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.results_dir, f"deduplication_summary_{timestamp}.csv")
            df.to_csv(report_path, index=False)
            
            # Create summary statistics
            stats = {
                'Total Groups': len(duplicate_groups),
                'Total Duplicates': sum(len(group) - 1 for group in duplicate_groups),
                'Average Group Size': np.mean([len(group) for group in duplicate_groups]),
                'Max Group Size': max(len(group) for group in duplicate_groups),
                'Min Group Size': min(len(group) for group in duplicate_groups),
                'Average Similarity': np.mean([s for s in similarity_scores.values()])
            }
            
            # Save statistics
            stats_path = os.path.join(self.results_dir, f"deduplication_stats_{timestamp}.txt")
            with open(stats_path, 'w') as f:
                for key, value in stats.items():
                    f.write(f"{key}: {value:.2f}\n")
                    
            logger.info(f"Summary report saved to {report_path}")
            logger.info(f"Statistics saved to {stats_path}")
            
        except Exception as e:
            logger.error(f"Error creating summary report: {e}")
            
def main():
    """Run the test."""
    test = TestDeduplication()
    success = test.run_deduplication()
    
    if success:
        logger.info("Test completed successfully")
        logger.info(f"Results saved in {test.results_dir}")
        logger.info("Check the following files:")
        logger.info(f"1. CSV report: {test.results_dir}/deduplication_summary_*.csv")
        logger.info(f"2. Statistics: {test.results_dir}/deduplication_stats_*.txt")
        logger.info(f"3. Best images: {test.results_dir}/best/")
        logger.info(f"4. Duplicate images: {test.results_dir}/duplicates/")
    else:
        logger.error("Test failed")
        
if __name__ == "__main__":
    main() 