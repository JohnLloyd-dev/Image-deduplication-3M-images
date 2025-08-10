import os
import sys
import logging
import time
import json
import torch
import numpy as np
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty, Full
import threading
from datetime import datetime
import multiprocessing as mp
import gzip
from io import BytesIO
from collections import defaultdict, deque
from modules.feature_cache import BoundedFeatureCache

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.azure_utils import (
    list_blobs_from_azure,
    download_blob_from_azure,
    copy_blob_within_azure,
    download_blob_to_memory,
    AzureBlobManager,
    validate_sas_url
)
from modules.feature_extraction import FeatureExtractor
from modules.deduplication import Deduplicator
from modules.memory_efficient_deduplication import MemoryEfficientDeduplicator
from modules.multithreaded_deduplication import MultiThreadedDeduplicator
from modules.threading_optimizer import create_optimized_deduplicator
from modules.distributed_processor import DistributedProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Azure configuration
NUM_AZURE_CONNECTIONS = 10
SAS_URLS = [
    "https://azwtewebsitecache.blob.core.windows.net/webvia?sp=rcwl&st=2025-05-05T17:40:16Z&se=2025-11-05T18:40:16Z&spr=https&sv=2024-11-04&sr=c&sig=6eTcYmq%2BeauVioFmi1bxh%2Bd4gDjvNdq54EufmpPSKYY%3D"
] * NUM_AZURE_CONNECTIONS

# Target directory in Azure container
TARGET_DIR = "Image_Dedup_Project/TestEquity/CompleteImageDataset/"

# Worker configuration
MIN_DOWNLOAD_WORKERS = 20  # Reduced from 100
MAX_DOWNLOAD_WORKERS = 40  # Reduced from 200
NUM_PROCESS_WORKERS = 8    # Reduced from 48
BATCH_SIZE = 32           # Keep batch size the same for GPU efficiency

# Rate limiting settings
MAX_REQUESTS_PER_SECOND = 120

class TokenBucketRateLimiter:
    def __init__(self, rate_per_sec, min_rate=100, max_rate=150, window_sec=300):
        self.rate = rate_per_sec
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.window_sec = window_sec
        self.bucket_capacity = max_rate * window_sec
        self.tokens = self.bucket_capacity
        self.lock = threading.Lock()
        self.last_refill = time.time()

    def set_rate(self, new_rate):
        with self.lock:
            self.rate = max(self.min_rate, min(self.max_rate, new_rate))

    def get_rate(self):
        with self.lock:
            return self.rate

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.rate
        self.tokens = min(self.tokens + refill_amount, self.bucket_capacity)
        self.last_refill = now

    def wait(self):
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                # Not enough tokens, calculate wait time
                needed = 1 - self.tokens
                wait_time = needed / self.rate
            time.sleep(min(wait_time, 1.0))

# Global stop event for graceful shutdown
stop_event = threading.Event()

# Add token bucket limiter for download workers only
download_token_bucket = TokenBucketRateLimiter(rate_per_sec=35, min_rate=30, max_rate=60, window_sec=60)

class ProgressTracker:
    """Tracks progress of the pipeline."""
    
    def __init__(self, total_images: int):
        self.total_images = total_images
        self.downloaded = 0
        self.processed = 0
        self.saved = 0
        self.lock = threading.Lock()
        self.processed_files = set()
        
    def update_downloaded(self):
        with self.lock:
            self.downloaded += 1
            logger.info(f"Downloaded {self.downloaded}/{self.total_images} images")
                
    def update_processed(self):
        with self.lock:
            self.processed += 1
            logger.info(f"Processed {self.processed}/{self.total_images} images")
                
    def update_saved(self):
        with self.lock:
            self.saved += 1
            logger.info(f"Saved {self.saved}/{self.total_images} features")
    
    def add_processed_file(self, file_path: str):
        with self.lock:
            self.processed_files.add(file_path)
            
    def is_processed(self, file_path: str) -> bool:
        with self.lock:
            return file_path in self.processed_files
                
    def get_summary(self):
        return {
            'total_images': self.total_images,
            'downloaded': self.downloaded,
            'processed': self.processed,
            'saved': self.saved,
            'processed_files': list(self.processed_files)
        }
    
    def save_progress(self, progress_file: str):
        with self.lock:
            progress_data = {
                'total_images': self.total_images,
                'downloaded': self.downloaded,
                'processed': self.processed,
                'saved': self.saved,
                'processed_files': list(self.processed_files),
                'timestamp': datetime.now().isoformat()
            }
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f)
    
    @classmethod
    def load_progress(cls, progress_file: str):
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                data = json.load(f)
                tracker = cls(data['total_images'])
                tracker.downloaded = data['downloaded']
                tracker.processed = data['processed']
                tracker.saved = data['saved']
                tracker.processed_files = set(data['processed_files'])
                return tracker
        return None

class FeatureCache:
    """In-memory cache for compressed features."""
    def __init__(self):
        self.features = defaultdict(BytesIO)
        self.lock = threading.Lock()
        
    def store(self, key: str, compressed_data: bytes):
        """Store compressed features in memory."""
        with self.lock:
            self.features[key].write(compressed_data)
            self.features[key].seek(0)
            
    def get(self, key: str) -> Optional[bytes]:
        """Get compressed features from memory."""
        with self.lock:
            if key in self.features:
                self.features[key].seek(0)
                return self.features[key].read()
        return None
        
    def clear(self):
        """Clear all stored features."""
        with self.lock:
            self.features.clear()

class AzureConnectionPool:
    """Manages a pool of Azure connections"""
    def __init__(self, sas_urls: List[str]):
        self.connections = [AzureBlobManager(url) for url in sas_urls]
        self.current_index = 0
        self.lock = threading.Lock()
        
    def get_connection(self) -> AzureBlobManager:
        """Get next connection in round-robin fashion"""
        with self.lock:
            connection = self.connections[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.connections)
            return connection

def shutdown_sequence(
    download_queue: Queue,
    process_queue: Queue,
    save_queue: Queue,
    download_threads: List[threading.Thread],
    process_threads: List[threading.Thread],
    save_worker_thread: threading.Thread
):
    """Graceful shutdown sequence for all workers."""
    logger.info("Initiating shutdown sequence")
    stop_event.set()
    
    # Drain queues
    for q in [download_queue, process_queue, save_queue]:
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except Empty:
                break
    
    # Send termination signals
    for _ in range(len(download_threads)):
        try:
            download_queue.put(None, timeout=5)
        except Full:
            logger.warning("Download queue full during shutdown")
            
    for _ in range(len(process_threads)):
        try:
            process_queue.put(None, timeout=5)
        except Full:
            logger.warning("Process queue full during shutdown")
            
    try:
        save_queue.put(None, timeout=10)
    except Full:
        logger.warning("Save queue full during shutdown")
    
    # Wait for threads to finish
    for t in download_threads:
        t.join(timeout=10.0)
        if t.is_alive():
            logger.warning(f"Download worker {t.name} did not terminate")
            
    for t in process_threads:
        t.join(timeout=15.0)
        if t.is_alive():
            logger.warning(f"Process worker {t.name} did not terminate")
            
    save_worker_thread.join(timeout=20.0)
    if save_worker_thread.is_alive():
        logger.error("Save worker did not terminate")
    
    logger.info("Shutdown sequence completed")

def download_worker_azure(
    download_queue: Queue,
    process_queue: Queue,
    progress_tracker: ProgressTracker,
    connection_pool: AzureConnectionPool
):
    """Download worker that handles Azure blob downloads with retry logic"""
    max_retries = 3
    retry_delay = 5  # seconds
    
    while not stop_event.is_set():
        try:
            item = download_queue.get(block=True, timeout=5)
            
            if item is None:
                download_queue.task_done()
                break
                
            blob_name = item['name']
            retry_count = item.get('retry_count', 0)
            
            try:
                # Get connection from pool
                azure_manager = connection_pool.get_connection()
                
                try:
                    # Download blob
                    image_data = azure_manager.download_as_cv2(blob_name)
                    if image_data is not None:
                        process_queue.put({
                            'azure_path': blob_name,
                            'image_data': image_data
                        })
                        progress_tracker.update_downloaded()
                    else:
                        raise Exception("Download returned no data")
                        
                except Exception as e:
                    if retry_count < max_retries:
                        logger.warning(f"Download failed for {blob_name}, retrying ({retry_count + 1}/{max_retries}): {str(e)}")
                        time.sleep(retry_delay)
                        download_queue.put({'name': blob_name, 'retry_count': retry_count + 1})
                    else:
                        logger.error(f"Download failed for {blob_name} after {max_retries} retries: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Error in download worker: {str(e)}")
                
            finally:
                download_queue.task_done()
                
        except Empty:
            continue
        except Exception as e:
            logger.error(f"Fatal error in download worker: {str(e)}")

def process_worker(
    process_queue: Queue,
    save_queue: Queue,
    progress_tracker: ProgressTracker,
    distributed_processor: DistributedProcessor
):
    """Simple process worker that processes one image at a time"""
    try:
        while not stop_event.is_set():
            try:
                item = process_queue.get(block=True, timeout=5)
                
                if item is None:
                    process_queue.task_done()
                    break
                
                try:
                    # Ensure we're passing the image data correctly
                    image_data = item['image_data']
                    azure_path = item['azure_path']
                    
                    # Process the image - pass as a list of dictionaries
                    results = distributed_processor.process_images([{
                        'image_data': image_data,
                        'azure_path': azure_path
                    }])
                    
                    for res in results:
                        if res is not None:
                            save_queue.put({
                                'azure_path': res['azure_path'],
                                'features': res['features']
                            })
                            progress_tracker.update_processed()
                finally:
                    process_queue.task_done()
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
    except Exception as e:
        logger.error(f"Fatal error in process worker: {str(e)}")

def save_worker(
    save_queue: Queue,
    progress_tracker: ProgressTracker,
    feature_cache: BoundedFeatureCache
):
    """Simple save worker that saves one feature at a time"""
    processed_files = set()
    
    try:
        while not stop_event.is_set():
            try:
                item = save_queue.get(timeout=5)
                if item is None:
                    break
                    
                try:
                    if item['azure_path'] in processed_files:
                        save_queue.task_done()
                        continue
                        
                    # Save features
                    feature_cache.put_features(item['azure_path'], item['features'])
                    processed_files.add(item['azure_path'])
                    progress_tracker.add_processed_file(item['azure_path'])
                    progress_tracker.update_saved()
                    
                    # Verify feature was saved
                    feature_path = feature_cache.get_feature_path(item['azure_path'])
                    if not os.path.exists(feature_path):
                        logger.error(f"Failed to save features for {item['azure_path']}")
                    else:
                        logger.info(f"Successfully saved features for {item['azure_path']}")
                        
                except Exception as e:
                    logger.error(f"Error saving features for {item['azure_path']}: {str(e)}")
                finally:
                    save_queue.task_done()
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Save worker error: {str(e)}")
                time.sleep(1)
    except Exception as e:
        logger.error(f"Fatal error in save worker: {str(e)}")
    finally:
        try:
            progress_file = f"pipeline_progress_{TARGET_DIR.replace('/', '_')}.json"
            progress_tracker.save_progress(progress_file)
        except Exception as e:
            logger.error(f"Error saving final progress: {str(e)}")

def load_features_from_disk(features_dir: str) -> Dict[str, np.ndarray]:
    """Load all features from disk."""
    features = {}
    try:
        for filename in os.listdir(features_dir):
            if filename.endswith('.npy'):
                feature_path = os.path.join(features_dir, filename)
                features[filename.replace('_', '/').replace('.npy', '')] = np.load(feature_path)
        return features
    except Exception as e:
        logger.error(f"Error loading features from disk: {str(e)}")
        return {}

def run_pipeline():
    """Complete pipeline with proper step connections"""
    distributed_processor = None
    feature_cache = None
    deduplicator = None
    report_path = None
    
    try:
        mp.set_start_method('spawn', force=True)
        
        connection_pool = AzureConnectionPool(SAS_URLS)
        
        logger.info(f"Getting list of images from Azure directory: {TARGET_DIR}")
        try:
            blob_list = list_blobs_from_azure(SAS_URLS[0])
            if not blob_list:
                logger.error(f"No images found in Azure directory: {TARGET_DIR}")
                return
                
            blob_list = [blob for blob in blob_list if blob.startswith(TARGET_DIR)]
            if not blob_list:
                logger.error(f"No images found in Azure directory: {TARGET_DIR}")
                return
                
            logger.info(f"Found {len(blob_list)} images in Azure directory")
        except Exception as e:
            logger.error(f"Failed to list blobs from Azure: {str(e)}")
            return
            
        progress_file = f"pipeline_progress_{TARGET_DIR.replace('/', '_')}.json"
        progress_tracker = ProgressTracker.load_progress(progress_file)
        if progress_tracker is None:
            progress_tracker = ProgressTracker(len(blob_list))
        
        # Check if we need to process new images or just run deduplication
        remaining_blobs = [blob for blob in blob_list if not progress_tracker.is_processed(blob)]
        logger.info(f"Remaining blobs to process: {len(remaining_blobs)}")
        
        # Initialize feature cache to check existing features
        feature_cache = BoundedFeatureCache()
        existing_features = feature_cache.get_all_features()
        logger.info(f"Existing features found: {len(existing_features)}")
        
        # Step 1-3: Download, Extract, Save (only if needed)
        if remaining_blobs:
            logger.info("Starting download and feature extraction pipeline...")
            
            num_gpus = torch.cuda.device_count()
            distributed_processor = DistributedProcessor(num_gpus=num_gpus)
            
            download_queue = Queue(maxsize=200)
            process_queue = Queue(maxsize=100)
            save_queue = Queue(maxsize=1000)
            
            download_threads = []
            process_threads = []
            save_threads = []
            
            current_download_workers = min(MIN_DOWNLOAD_WORKERS, len(remaining_blobs))
            for i in range(current_download_workers):
                t = threading.Thread(
                    target=download_worker_azure,
                    args=(download_queue, process_queue, progress_tracker, connection_pool),
                    name=f"DownloadWorker-{i}"
                )
                t.daemon = True
                t.start()
                download_threads.append(t)
                
            for i in range(NUM_PROCESS_WORKERS):
                t = threading.Thread(
                    target=process_worker,
                    args=(process_queue, save_queue, progress_tracker, distributed_processor),
                    name=f"ProcessWorker-{i}"
                )
                t.daemon = True
                t.start()
                process_threads.append(t)
                
            save_worker_thread = threading.Thread(
                target=save_worker,
                args=(save_queue, progress_tracker, feature_cache),
                name="SaveWorker"
            )
            save_worker_thread.daemon = True
            save_worker_thread.start()
            save_threads.append(save_worker_thread)
            
            # Queue all blobs
            for blob_name in remaining_blobs:
                try:
                    download_queue.put({'name': blob_name, 'retry_count': 0})
                except Exception as e:
                    logger.error(f"Error queueing blob {blob_name}: {str(e)}")
                    continue
            
            try:
                # Wait for all queues to be processed with timeout
                download_queue.join()
                logger.info("Download queue completed")
                
                # Signal process workers to finish
                for _ in range(len(process_threads)):
                    process_queue.put(None)
                process_queue.join()
                logger.info("Process queue completed")
                
                # Signal save worker to finish
                save_queue.put(None)
                save_queue.join()
                logger.info("Save queue completed")
                
            except Exception as e:
                logger.error(f"Processing interrupted: {str(e)}")
                raise
            finally:
                # Graceful shutdown
                shutdown_sequence(
                    download_queue,
                    process_queue,
                    save_queue,
                    download_threads,
                    process_threads,
                    save_worker_thread
                )
                
                # Save final progress
                progress_tracker.save_progress(progress_file)
                
                if distributed_processor is not None:
                    distributed_processor.shutdown()
        else:
            logger.info("No new images to process, using existing features")
        
        # Step 4: Deduplication
        logger.info("Starting deduplication process...")
        try:
            logger.info("Loading features from disk for deduplication...")
            all_features = feature_cache.get_all_features()
            logger.info(f"Total features available for deduplication (from cache): {len(all_features)}")

            # If cache is empty, try loading from disk using blob list
            if len(all_features) == 0:
                logger.info("No features found in cache, loading from disk using blob list...")
                all_features = {}
                for blob in blob_list:
                    feature = feature_cache.get_features(blob)
                    if feature is not None:
                        all_features[blob] = feature
                logger.info(f"Total features loaded from disk using blob list: {len(all_features)}")

            if len(all_features) == 0:
                logger.error("No features available for deduplication!")
                return {
                    'summary': progress_tracker.get_summary(),
                    'blob_list': blob_list,
                    'deduplication_report': None,
                    'error': 'No features available for deduplication'
                }

            # Use auto-optimized multi-threaded memory-efficient deduplicator for maximum performance
            logger.info("Initializing auto-optimized multi-threaded memory-efficient deduplicator...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            deduplicator = create_optimized_deduplicator(
                feature_cache=feature_cache,
                device=device
            )
            
            # Create output directory for deduplication results
            output_dir = os.path.join("deduplication_results", TARGET_DIR.replace('/', '_'))
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info("Running multi-threaded memory-efficient deduplication algorithm...")
            logger.info(f"Processing {len(all_features)} images with multi-threaded staged approach")
            logger.info(f"Using {deduplicator.max_workers} worker threads for parallel processing")
            
            # Progress callback for UI updates
            def progress_callback(stage_info, progress_percent):
                logger.info(f"Multi-threaded deduplication progress: {stage_info} ({progress_percent:.1f}%)")
            
            duplicate_groups, similarity_scores = deduplicator.deduplicate_multithreaded(
                image_paths=list(all_features.keys()),
                output_dir=output_dir,
                progress_callback=progress_callback
            )
            logger.info("Multi-threaded memory-efficient deduplication algorithm finished.")
            
            # Create deduplication report
            logger.info("Creating deduplication report...")
            report_path = deduplicator.create_report(
                duplicate_groups=duplicate_groups,
                similarity_scores=similarity_scores,
                output_dir=output_dir
            )
            
            logger.info(f"Deduplication completed. Found {len(duplicate_groups)} duplicate groups")
            logger.info(f"Deduplication report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Deduplication failed: {str(e)}", exc_info=True)
            return {
                'summary': progress_tracker.get_summary(),
                'blob_list': blob_list,
                'deduplication_report': None,
                'error': f'Deduplication failed: {str(e)}'
            }
        
        # Step 5: Copy images to Azure output directory based on report
        logger.info("Starting Azure copy process based on deduplication report...")
        try:
            if report_path and os.path.exists(report_path):
                # Read the CSV deduplication report to determine which images to copy
                import pandas as pd
                df = pd.read_csv(report_path)
                
                # Extract both best and duplicate images for copying
                best_images_df = df[df['Status'] == 'Best']
                duplicate_images_df = df[df['Status'] == 'Duplicate']
                
                best_images = set(best_images_df['Image Path'].tolist())
                duplicate_images = set(duplicate_images_df['Image Path'].tolist())
                
                logger.info(f"Found {len(best_images)} best images and {len(duplicate_images)} duplicate images to copy to Azure")
                
                # Define output directories in Azure
                azure_base_dir = f"{TARGET_DIR.rstrip('/')}_deduplicated/"
                azure_best_dir = azure_base_dir + "best/"
                azure_duplicate_dir = azure_base_dir + "duplicates/"
                
                # Copy best images to Azure best directory
                best_copied_count = 0
                logger.info(f"Copying {len(best_images)} best images to {azure_best_dir}")
                for image_path in best_images:
                    try:
                        # Construct source and destination paths
                        source_blob = image_path
                        dest_blob = azure_best_dir + os.path.basename(image_path)
                        
                        # Copy blob within Azure
                        azure_manager = connection_pool.get_connection()
                        success = azure_manager.copy_blob(source_blob, dest_blob)
                        
                        if success:
                            best_copied_count += 1
                            if best_copied_count % 50 == 0:
                                logger.info(f"Copied {best_copied_count}/{len(best_images)} best images to Azure")
                        else:
                            logger.warning(f"Failed to copy best image {source_blob} to {dest_blob}")
                            
                    except Exception as e:
                        logger.error(f"Error copying best image {image_path}: {str(e)}")
                
                # Copy duplicate images to Azure duplicates directory
                duplicate_copied_count = 0
                logger.info(f"Copying {len(duplicate_images)} duplicate images to {azure_duplicate_dir}")
                for image_path in duplicate_images:
                    try:
                        # Construct source and destination paths
                        source_blob = image_path
                        dest_blob = azure_duplicate_dir + os.path.basename(image_path)
                        
                        # Copy blob within Azure
                        azure_manager = connection_pool.get_connection()
                        success = azure_manager.copy_blob(source_blob, dest_blob)
                        
                        if success:
                            duplicate_copied_count += 1
                            if duplicate_copied_count % 50 == 0:
                                logger.info(f"Copied {duplicate_copied_count}/{len(duplicate_images)} duplicate images to Azure")
                        else:
                            logger.warning(f"Failed to copy duplicate image {source_blob} to {dest_blob}")
                            
                    except Exception as e:
                        logger.error(f"Error copying duplicate image {image_path}: {str(e)}")
                
                total_copied = best_copied_count + duplicate_copied_count
                total_images = len(best_images) + len(duplicate_images)
                logger.info(f"Azure copy completed. Successfully copied {total_copied}/{total_images} images")
                logger.info(f"  - Best images: {best_copied_count}/{len(best_images)} copied to {azure_best_dir}")
                logger.info(f"  - Duplicate images: {duplicate_copied_count}/{len(duplicate_images)} copied to {azure_duplicate_dir}")
                
                # Save copy results
                copy_results = {
                    'total_best_images': len(best_images),
                    'total_duplicate_images': len(duplicate_images),
                    'total_images': total_images,
                    'best_images_copied': best_copied_count,
                    'duplicate_images_copied': duplicate_copied_count,
                    'total_successfully_copied': total_copied,
                    'azure_best_directory': azure_best_dir,
                    'azure_duplicate_directory': azure_duplicate_dir,
                    'azure_base_directory': azure_base_dir,
                    'timestamp': datetime.now().isoformat()
                }
                
                copy_results_path = os.path.join(output_dir, 'azure_copy_results.json')
                with open(copy_results_path, 'w') as f:
                    json.dump(copy_results, f, indent=2)
                
                logger.info(f"Azure copy results saved to: {copy_results_path}")
                
            else:
                logger.error("Deduplication report not found or invalid")
                
        except Exception as e:
            logger.error(f"Azure copy process failed: {str(e)}", exc_info=True)
        
        summary = progress_tracker.get_summary()
        logger.info(f"Pipeline completed successfully: {summary}")
        
        return {
            'summary': summary,
            'blob_list': blob_list,
            'deduplication_report': report_path,
            'azure_base_directory': azure_base_dir if 'azure_base_dir' in locals() else None,
            'azure_best_directory': azure_best_dir if 'azure_best_dir' in locals() else None,
            'azure_duplicate_directory': azure_duplicate_dir if 'azure_duplicate_dir' in locals() else None,
            'copy_results': copy_results if 'copy_results' in locals() else None
        }
        
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}", exc_info=True)
        return {
            'summary': progress_tracker.get_summary() if 'progress_tracker' in locals() else {},
            'blob_list': blob_list if 'blob_list' in locals() else [],
            'deduplication_report': None,
            'error': str(e)
        }
    finally:
        if distributed_processor is not None:
            distributed_processor.shutdown()
        if deduplicator is not None:
            deduplicator.release()

if __name__ == "__main__":
    run_pipeline() 