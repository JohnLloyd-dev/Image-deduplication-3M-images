#!/usr/bin/env python3
"""
Copy Images to Azure Based on Deduplication Report

This script reads the deduplication CSV report and copies images to Azure
destination directory based on their status (Best vs Duplicate).
"""

import os
import sys
import time
import pandas as pd
import logging
from typing import List, Dict, Tuple
from pathlib import Path

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Import configuration
try:
    from azure_copy_config import *
except ImportError:
    # Default values if config file not found
    AZURE_SAS_URL = None
    AZURE_DESTINATION_BASE = "JohnLloyd_test/result/"
    REPORT_FILE_PATH = "deduplication_report.csv"
    COPY_BEST_IMAGES = True
    COPY_DUPLICATE_IMAGES = True
    BATCH_SIZE = 50
    LOG_LEVEL = "INFO"
    LOG_TO_FILE = True
    LOG_FILE_NAME = "azure_copy.log"
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 2
    MAX_CONCURRENT_COPIES = 10
    RATE_LIMIT_DELAY = 0.1

from azure_utils import AzureBlobManager, SAS_URL

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
handlers = [logging.StreamHandler(sys.stdout)]
if LOG_TO_FILE:
    handlers.append(logging.FileHandler(LOG_FILE_NAME))

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)


class AzureImageCopier:
    """Copy images to Azure based on deduplication report."""
    
    def __init__(self, sas_url: str = None, destination_base: str = None):
        """
        Initialize the Azure image copier.
        
        Args:
            sas_url: Azure SAS URL. If None, uses default from azure_utils
            destination_base: Base destination path in Azure
        """
        self.sas_url = sas_url or AZURE_SAS_URL or SAS_URL
        self.destination_base = (destination_base or AZURE_DESTINATION_BASE).rstrip('/') + '/'
        self.azure_manager = AzureBlobManager(self.sas_url)
        
        # Create destination directories
        self.best_dir = self.destination_base + "best/"
        self.duplicate_dir = self.destination_base + "duplicates/"
        
        logger.info(f"Initialized Azure Image Copier")
        logger.info(f"Destination base: {self.destination_base}")
        logger.info(f"Best images dir: {self.best_dir}")
        logger.info(f"Duplicate images dir: {self.duplicate_dir}")
    
    def read_report(self, report_path: str) -> Tuple[List[str], List[str]]:
        """
        Read the deduplication CSV report and extract image paths.
        
        Args:
            report_path: Path to the CSV report file
            
        Returns:
            Tuple of (best_images, duplicate_images) lists
        """
        try:
            logger.info(f"Reading report from: {report_path}")
            
            if not os.path.exists(report_path):
                raise FileNotFoundError(f"Report file not found: {report_path}")
            
            # Read CSV with pandas
            df = pd.read_csv(report_path)
            
            # Validate required columns
            required_columns = ['Image Path', 'Status']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in report: {missing_columns}")
            
            # Extract image paths by status
            best_images = df[df['Status'] == 'Best']['Image Path'].tolist()
            duplicate_images = df[df['Status'] == 'Duplicate']['Image Path'].tolist()
            
            logger.info(f"Report analysis:")
            logger.info(f"  - Total rows: {len(df)}")
            logger.info(f"  - Best images: {len(best_images)}")
            logger.info(f"  - Duplicate images: {len(duplicate_images)}")
            logger.info(f"  - Unique statuses: {df['Status'].unique().tolist()}")
            
            return best_images, duplicate_images
            
        except Exception as e:
            logger.error(f"Error reading report: {e}")
            raise
    
    def copy_images_to_azure(self, image_paths: List[str], destination_dir: str, 
                            image_type: str = "images") -> Dict[str, bool]:
        """
        Copy a list of images to Azure destination directory.
        
        Args:
            image_paths: List of source image paths in Azure
            destination_dir: Destination directory in Azure
            image_type: Description of image type for logging
            
        Returns:
            Dictionary mapping image paths to copy success status
        """
        if not image_paths:
            logger.info(f"No {image_type} to copy")
            return {}
        
        logger.info(f"Starting to copy {len(image_paths)} {image_type} to {destination_dir}")
        
        results = {}
        successful_copies = 0
        failed_copies = 0
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                # Construct destination path
                filename = os.path.basename(image_path)
                dest_blob = destination_dir + filename
                
                # Copy blob within Azure
                success = self.azure_manager.copy_blob(image_path, dest_blob)
                
                if success:
                    results[image_path] = True
                    successful_copies += 1
                    logger.info(f"✓ Copied {filename} ({i}/{len(image_paths)})")
                else:
                    results[image_path] = False
                    failed_copies += 1
                    logger.error(f"✗ Failed to copy {filename}")
                
                # Progress update based on batch size
                if i % BATCH_SIZE == 0:
                    logger.info(f"Progress: {i}/{len(image_paths)} {image_type} processed")
                
                # Rate limiting to respect Azure limits
                if RATE_LIMIT_DELAY > 0:
                    time.sleep(RATE_LIMIT_DELAY)
                
            except Exception as e:
                results[image_path] = False
                failed_copies += 1
                logger.error(f"✗ Error copying {image_path}: {e}")
        
        # Summary
        logger.info(f"Copy operation completed for {image_type}:")
        logger.info(f"  - Successful: {successful_copies}/{len(image_paths)}")
        logger.info(f"  - Failed: {failed_copies}/{len(image_paths)}")
        
        return results
    
    def copy_all_images(self, report_path: str) -> Dict[str, Dict[str, bool]]:
        """
        Copy all images from the report to Azure destination.
        
        Args:
            report_path: Path to the deduplication CSV report
            
        Returns:
            Dictionary with copy results for best and duplicate images
        """
        try:
            # Read the report
            best_images, duplicate_images = self.read_report(report_path)
            
            # Copy best images (if enabled)
            best_results = {}
            if COPY_BEST_IMAGES and best_images:
                logger.info("=" * 60)
                logger.info("COPYING BEST IMAGES")
                logger.info("=" * 60)
                best_results = self.copy_images_to_azure(
                    best_images, 
                    self.best_dir, 
                    "best images"
                )
            else:
                logger.info("Skipping best images copy (disabled or no images)")
            
            # Copy duplicate images (if enabled)
            duplicate_results = {}
            if COPY_DUPLICATE_IMAGES and duplicate_images:
                logger.info("=" * 60)
                logger.info("COPYING DUPLICATE IMAGES")
                logger.info("=" * 60)
                duplicate_results = self.copy_images_to_azure(
                    duplicate_images, 
                    self.duplicate_dir, 
                    "duplicate images"
                )
            else:
                logger.info("Skipping duplicate images copy (disabled or no images)")
            
            # Overall summary
            total_best = len(best_images)
            total_duplicate = len(duplicate_images)
            successful_best = sum(1 for success in best_results.values() if success)
            successful_duplicate = sum(1 for success in duplicate_results.values() if success)
            
            logger.info("=" * 60)
            logger.info("FINAL SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Best Images: {successful_best}/{total_best} copied to {self.best_dir}")
            logger.info(f"Duplicate Images: {successful_duplicate}/{total_duplicate} copied to {self.duplicate_dir}")
            logger.info(f"Total Success Rate: {(successful_best + successful_duplicate)}/{(total_best + total_duplicate)}")
            
            return {
                'best_images': best_results,
                'duplicate_images': duplicate_results
            }
            
        except Exception as e:
            logger.error(f"Error in copy_all_images: {e}")
            raise


def main():
    """Main function to run the Azure image copy process."""
    try:
        # Check if report file exists
        if not os.path.exists(REPORT_FILE_PATH):
            logger.error(f"Report file not found: {REPORT_FILE_PATH}")
            logger.info("Please ensure the deduplication report CSV file exists.")
            logger.info("You can run the deduplication pipeline first to generate it.")
            logger.info(f"Expected report file: {REPORT_FILE_PATH}")
            return False
        
        # Initialize copier
        copier = AzureImageCopier()
        
        # Copy all images
        results = copier.copy_all_images(REPORT_FILE_PATH)
        
        logger.info("🎉 Azure image copy process completed successfully!")
        return True
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Process interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Azure copy process failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
