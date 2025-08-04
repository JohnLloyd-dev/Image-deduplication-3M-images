import os
import sys
from pathlib import Path
from src.pipeline.main_pipeline import DeduplicationPipeline
from src.utils.logger import get_logger

def test_pipeline(image_dir: str, output_dir: str):
    """
    Test the deduplication pipeline on a directory of images.
    
    Args:
        image_dir: Directory containing test images
        output_dir: Directory to save results
    """
    logger = get_logger()
    logger.info(f"Starting test pipeline on images in {image_dir}")
    
    try:
        # Validate input directory
        if not os.path.exists(image_dir):
            raise ValueError(f"Input directory does not exist: {image_dir}")
            
        # Initialize and run pipeline
        pipeline = DeduplicationPipeline()
        pipeline.run(image_dir, output_dir)
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python test_pipeline.py <image_directory> <output_directory>")
        sys.exit(1)
        
    image_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(image_dir):
        print(f"Image directory '{image_dir}' does not exist")
        sys.exit(1)
        
    test_pipeline(image_dir, output_dir) 