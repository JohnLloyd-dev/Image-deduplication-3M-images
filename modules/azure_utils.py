# (No code changes needed, just move this file into the modules/ folder.) 

import os
import time
import logging
import threading
import json
import pickle
from azure.storage.blob import ContainerClient, BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, ServiceRequestError, HttpResponseError
from datetime import datetime
from typing import List, Optional, Dict, Any, Generator
import numpy as np
from PIL import Image
import io
import requests
from urllib.parse import urlparse, parse_qs
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import urllib.parse

# Configure logging to filter out Azure HTTP requests
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Comprehensive list of image extensions
ALL_IMAGE_EXTS = (
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", 
    ".gif", ".jfif", ".pnm", ".ppm", ".pgm", ".pbm", 
    ".heic", ".avif", ".ico", ".svg", ".raw", ".cr2", 
    ".nef", ".arw", ".dng", ".raf", ".rw2", ".pef", 
    ".srw", ".orf", ".x3f", ".mrw", ".mef", ".iiq"
)

# Rate limiting settings - 3000 requests per second = 180000 requests per minute
RATE_LIMIT_DELAY = 0.0  # No delay between requests
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 10.0

# Cache file for blob list
BLOB_LIST_CACHE = 'azure_blob_list.json'

# SAS URL for Azure Blob Storage
SAS_URL = "https://azwtewebsitecache.blob.core.windows.net/webvia?sp=rcwl&st=2025-05-05T17:40:16Z&se=2025-11-05T18:40:16Z&spr=https&sv=2024-11-04&sr=c&sig=6eTcYmq%2BeauVioFmi1bxh%2Bd4gDjvNdq54EufmpPSKYY%3D"

class TokenBucketRateLimiter:
    def __init__(self, rate_per_sec, min_rate=20, max_rate=40, window_sec=300):
        self.rate = rate_per_sec
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.window_sec = window_sec
        self.bucket_capacity = max_rate * window_sec  # 1800 tokens
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

class AzureBlobManager:
    def __init__(self, sas_url: str, max_retries: int = 3):
        self.container_client = ContainerClient.from_container_url(sas_url)
        self.max_retries = max_retries
        self.rate_limiter = TokenBucketRateLimiter(rate_per_sec=35, min_rate=30, max_rate=60, window_sec=60)
        
    def download_as_cv2(self, blob_name: str) -> Optional[np.ndarray]:
        """Download blob and return as cv2 image with rate limiting."""
        for attempt in range(self.max_retries):
            try:                # Apply rate limiting
                self.rate_limiter.wait()
                
                blob_client = self.container_client.get_blob_client(blob_name)
                data = blob_client.download_blob().readall()
                
                # Convert to numpy array
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    logger.error(f"Failed to decode image: {blob_name}")
                    continue
                    
                return img
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to download {blob_name} after {self.max_retries} attempts: {str(e)}")
                    # If Azure throttling detected, reduce rate more aggressively
                    if hasattr(e, 'status_code') and e.status_code == 429:
                        current_rate = self.rate_limiter.get_rate()
                        self.rate_limiter.set_rate(current_rate - 2)
                    elif 'throttle' in str(e).lower() or '429' in str(e):
                        current_rate = self.rate_limiter.get_rate()
                        self.rate_limiter.set_rate(current_rate - 2)
                time.sleep(1)  # Wait before retry
                
        return None

def save_blob_list(blob_list, cache_file=BLOB_LIST_CACHE):
    """Save the list of blobs to a cache file."""
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'blobs': blob_list
            }, f, indent=2)
        logger.info(f"Saved {len(blob_list)} blobs to cache file")
    except Exception as e:
        logger.error(f"Failed to save blob list: {str(e)}")

def load_blob_list(cache_file=BLOB_LIST_CACHE):
    """Load the list of blobs from cache file if it exists."""
    try:
        if not os.path.exists(cache_file):
            logger.info("No cache file found, will fetch from Azure")
            return None
            
        with open(cache_file, 'r') as f:
            data = json.load(f)
            
        logger.info(f"Successfully loaded {len(data['blobs'])} blobs from cache file")
        return data['blobs']
    except Exception as e:
        logger.error(f"Failed to load blob list from cache: {str(e)}")
        return None

def log_thread_timing(func):
    """Decorator to log thread-specific timing information."""
    def wrapper(*args, **kwargs):
        thread_name = threading.current_thread().name
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"[Thread-{thread_name}] {func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"[Thread-{thread_name}] {func.__name__} failed after {duration:.2f}s: {str(e)}")
            raise
    return wrapper

def retry_on_azure_error(func):
    """Simple retry decorator for Azure operations."""
    def wrapper(*args, **kwargs):
        thread_name = threading.current_thread().name
        last_exception = None
        total_retry_time = 0
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (ServiceRequestError, HttpResponseError) as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                    total_retry_time += delay
                    logger.warning(f"[Thread-{thread_name}] Retry {attempt + 1}/{MAX_RETRIES} in {delay:.1f}s")
                    time.sleep(delay)
        if total_retry_time > 0:
            logger.error(f"[Thread-{thread_name}] Failed after {total_retry_time:.1f}s of retries")
        raise last_exception
    return wrapper

@log_thread_timing
@retry_on_azure_error
def download_images_from_azure(sas_url, local_dir, exts=ALL_IMAGE_EXTS):
    """Download images from Azure Blob Storage with retries and rate limiting."""
    try:
        thread_name = threading.current_thread().name
        container_client = ContainerClient.from_container_url(sas_url)
        count = 0
        total_size = 0
        
        # First, get the total count of images
        all_blobs = list(container_client.list_blobs())
        total_images = sum(1 for blob in all_blobs if blob.name.lower().endswith(exts))
        logger.info(f"[Thread-{thread_name}] Found {total_images} images to process")
        
        for blob in all_blobs:
            if blob.name.lower().endswith(exts):
                blob_start_time = time.time()
                rate_limiter.wait()
                try:
                    blob_client = container_client.get_blob_client(blob)
                    file_path = os.path.join(local_dir, os.path.basename(blob.name))
                    with open(file_path, "wb") as f:
                        data = blob_client.download_blob().readall()
                        f.write(data)
                    count += 1
                    total_size += len(data)
                    blob_duration = time.time() - blob_start_time
                    logger.info(f"[Thread-{thread_name}] Downloaded {os.path.basename(blob.name)} ({len(data)/1024/1024:.1f}MB) in {blob_duration:.2f}s - Progress: {count}/{total_images}")
                except ResourceNotFoundError as e:
                    logger.error(f"[Thread-{thread_name}] Blob not found: {blob.name}")
                    continue
                except (ServiceRequestError, HttpResponseError) as e:
                    logger.error(f"[Thread-{thread_name}] Azure error: {str(e)}")
                    raise
                except Exception as e:
                    logger.error(f"[Thread-{thread_name}] Unexpected error: {str(e)}")
                    continue
        
        logger.info(f"[Thread-{thread_name}] Completed downloading {count} images ({total_size/1024/1024:.1f}MB)")
        return count
    except Exception as e:
        logger.error(f"[Thread-{thread_name}] Failed to download images: {str(e)}")
        raise

def get_container_name_from_sas(sas_url: str) -> str:
    """Extract container name from SAS URL."""
    parsed_url = urlparse(sas_url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) >= 1:
        return path_parts[0]
    raise ValueError("Could not extract container name from SAS URL")

def download_blob_to_memory(blob_name: str, sas_url: str) -> bytes:
    """Download a blob directly to memory without writing to disk."""
    try:
        # Construct full URL for the blob
        parsed_url = urlparse(sas_url)
        container_name = get_container_name_from_sas(sas_url)
        blob_url = f"{parsed_url.scheme}://{parsed_url.netloc}/{container_name}/{blob_name}{parsed_url.query}"
        
        # Download blob
        response = requests.get(blob_url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Error downloading blob {blob_name} to memory: {str(e)}")
        raise

def validate_sas_url(sas_url: str) -> str:
    """Validate and normalize SAS URL."""
    try:
        # Parse the URL
        parsed = urllib.parse.urlparse(sas_url)
        
        # Ensure the signature is properly encoded
        query_params = urllib.parse.parse_qs(parsed.query)
        
        # Reconstruct the URL with proper encoding
        new_query = urllib.parse.urlencode(query_params, doseq=True)
        
        # Reconstruct the full URL
        normalized_url = urllib.parse.urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))
        
        return normalized_url
    except Exception as e:
        logger.error(f"Error validating SAS URL: {e}")
        raise ValueError("Invalid SAS URL format")

def is_image_file(filename: str) -> bool:
    """Check if a file is an image based on its extension."""
    return filename.lower().endswith(ALL_IMAGE_EXTS)

def list_blobs_from_azure(sas_url: str = None, use_cache: bool = True, force_refresh: bool = False) -> List[str]:
    """List all blobs in the Azure container with permanent caching support."""
    # Try to load from cache first if not forcing refresh
    if use_cache and not force_refresh:
        cached_blobs = load_blob_list()
        if cached_blobs is not None:
            logger.info("Using cached blob list")
            return cached_blobs

    try:
        # Use the Azure SDK's built-in function directly
        container_client = ContainerClient.from_container_url(sas_url or SAS_URL)
        blobs = []
        logger.info("Starting to list blobs from Azure container...")
        
        for blob in container_client.list_blobs():
            if is_image_file(blob.name):
                blobs.append(blob.name)
                if len(blobs) % 100 == 0:  # Log progress every 100 files
                    logger.info(f"Found {len(blobs)} image files so far...")
        
        logger.info(f"Completed listing blobs. Total image files found: {len(blobs)}")
        
        # Save to cache
        if blobs:
            save_blob_list(blobs)
            
        return blobs
    except Exception as e:
        logger.error(f"Error listing blobs: {e}")
        return []

def download_blob_from_azure(blob_name: str) -> Optional[np.ndarray]:
    """Download a blob from Azure and convert to numpy array."""
    start_time = time.time()
    logger.info(f"Starting download of {blob_name}")
    
    try:
        container_client = ContainerClient.from_container_url(SAS_URL)
        blob_client = container_client.get_blob_client(blob_name)
        
        # Download blob
        logger.debug(f"Downloading blob content for {blob_name}")
        stream = blob_client.download_blob()
        data = stream.readall()
        download_time = time.time() - start_time
        logger.info(f"Downloaded {len(data)/1024/1024:.2f}MB for {blob_name} in {download_time:.2f}s")
        
        # Convert to numpy array
        logger.debug(f"Converting {blob_name} to numpy array")
        np_array = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error(f"Failed to decode image: {blob_name}")
            return None
            
        total_time = time.time() - start_time
        logger.info(f"Successfully processed {blob_name} ({img.shape[1]}x{img.shape[0]} pixels) in {total_time:.2f}s")
        return img
        
    except ResourceNotFoundError:
        logger.error(f"Blob not found: {blob_name}")
        return None
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Error processing {blob_name} after {total_time:.2f}s: {str(e)}")
        return None

def copy_blob_within_azure(source_blob_name: str, destination_blob_name: str) -> bool:
    """Copy a blob within the same container."""
    try:
        container_client = ContainerClient.from_container_url(SAS_URL)
        source_blob = container_client.get_blob_client(source_blob_name)
        target_blob = container_client.get_blob_client(destination_blob_name)
        
        # Start copy operation
        target_blob.start_copy_from_url(source_blob.url)
        return True
        
    except Exception as e:
        logger.error(f"Error copying blob {source_blob_name} to {destination_blob_name}: {e}")
        return False

def download_images_from_azure(
    blob_names: List[str],
    blob_service_client: BlobServiceClient,
    container_name: str,
    max_retries: int = 3
) -> List[Dict]:
    """Download multiple images from Azure."""
    results = []
    total_start_time = time.time()
    logger.info(f"Starting batch download of {len(blob_names)} images")
    
    for idx, blob_name in enumerate(blob_names, 1):
        start_time = time.time()
        logger.info(f"Processing image {idx}/{len(blob_names)}: {blob_name}")
        
        try:
            image_data = download_blob_from_azure(blob_name)
            if image_data is not None:
                results.append({
                    'azure_path': blob_name,
                    'image_data': image_data
                })
                process_time = time.time() - start_time
                logger.info(f"Successfully processed {blob_name} in {process_time:.2f}s")
            else:
                logger.warning(f"Failed to process {blob_name}")
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Error processing {blob_name} after {process_time:.2f}s: {str(e)}")
    
    total_time = time.time() - total_start_time
    success_rate = (len(results) / len(blob_names)) * 100
    logger.info(f"Batch download completed in {total_time:.2f}s. Success rate: {success_rate:.1f}% ({len(results)}/{len(blob_names)})")
    return results

class AzureBlobManager:
    def __init__(self, sas_url: str = None, max_concurrent_downloads: int = 3000, max_retries: int = 3):
        """Initialize the Azure Blob Manager.
        
        Args:
            sas_url: The SAS URL for the Azure Blob Storage container
            max_concurrent_downloads: Maximum number of concurrent downloads (default: 3000)
            max_retries: Maximum number of retry attempts for failed downloads
        """
        self.sas_url = sas_url or SAS_URL
        self.container_client = ContainerClient.from_container_url(self.sas_url)
        self.max_concurrent_downloads = max_concurrent_downloads
        self.rate_limiter = threading.Semaphore(max_concurrent_downloads)
        self.container_name = self._extract_container_name(self.sas_url)
        self.max_retries = max_retries
        self._blob_cache = {}
        self._cache_lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_concurrent_downloads,
            thread_name_prefix="azure_downloader"
        )
        logger.info(f"Initialized AzureBlobManager for container: {self.container_name} with {max_concurrent_downloads} concurrent downloads")
        
    def _extract_container_name(self, sas_url: str) -> str:
        """Extract container name from SAS URL."""
        try:
            # Example: "https://account.blob.core.windows.net/container?sp=rcwl..."
            parts = sas_url.split('?')[0].split('/')
            container = parts[-1] if parts[-1] != '' else parts[-2]
            if not container:
                raise ValueError("Could not extract container name from SAS URL")
            return container
        except Exception as e:
            logger.error(f"Error extracting container name: {e}")
            raise

    def list_blobs(self, prefix: str = "", use_cache: bool = True, force_refresh: bool = False) -> List[str]:
        """List blobs efficiently with permanent caching."""
        cache_file = f"blob_cache_{self.container_name}.pkl"
        
        # Try to load from cache first if not forcing refresh
        if use_cache and not force_refresh and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    if 'blobs' in data and 'prefix' in data and data['prefix'] == prefix:
                        logger.info(f"Loaded {len(data['blobs'])} blobs from cache for prefix '{prefix}'")
                        return data['blobs']
            except Exception as e:
                logger.error(f"Error loading blob cache: {e}")

        # List blobs from Azure
        blobs = []
        try:
            # Using the container client to list blobs with pagination
            blob_list = self.container_client.list_blobs(name_starts_with=prefix)
            for blob in blob_list:
                if is_image_file(blob.name):
                    blobs.append(blob.name)
                    if len(blobs) % 1000 == 0:
                        logger.info(f"Found {len(blobs)} images so far...")
            logger.info(f"Total images found: {len(blobs)}")
        except Exception as e:
            logger.error(f"Error listing blobs: {str(e)}")
            return []
        
        # Save to cache
        if blobs:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'blobs': blobs,
                        'prefix': prefix
                    }, f)
            except Exception as e:
                logger.error(f"Error saving blob cache: {e}")
            
        return blobs

    def download_to_memory(self, blob_name: str) -> bytes:
        """Download a blob to memory (bytes) with rate limiting and retries."""
        for attempt in range(self.max_retries):
            try:
                with self.rate_limiter:
                    blob_client = self.container_client.get_blob_client(blob_name)
                    stream = blob_client.download_blob()
                    return stream.readall()
            except ResourceNotFoundError:
                logger.error(f"Blob not found: {blob_name}")
                raise
            except (ServiceRequestError, ConnectionError) as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to download {blob_name} after {self.max_retries} attempts: {e}")
                    raise
                delay = min(2 ** attempt, 5)  # Reduced max delay to 5 seconds for faster retries
                logger.warning(f"Retry {attempt + 1}/{self.max_retries} for {blob_name}: {e} - Waiting {delay}s")
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Error downloading blob {blob_name}: {e}")
                raise

    @lru_cache(maxsize=1000)
    def get_blob_properties(self, blob_name: str) -> Optional[Dict[str, Any]]:
        """Get blob properties with caching."""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            return blob_client.get_blob_properties()
        except Exception as e:
            logger.error(f"Error getting properties for {blob_name}: {e}")
            return None

    def download_as_cv2(self, blob_name: str) -> Optional[np.ndarray]:
        """Download a blob and convert directly to OpenCV image."""
        start_time = time.time()
        logger.info(f"Starting download of {blob_name}")
        
        try:
            img_bytes = self.download_to_memory(blob_name)
            download_time = time.time() - start_time
            logger.info(f"Downloaded {len(img_bytes)/1024/1024:.2f}MB for {blob_name} in {download_time:.2f}s")
            
            np_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error(f"Failed to decode image: {blob_name}")
                return None
                
            total_time = time.time() - start_time
            logger.info(f"Successfully processed {blob_name} ({img.shape[1]}x{img.shape[0]} pixels) in {total_time:.2f}s")
            return img
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Error processing {blob_name} after {total_time:.2f}s: {str(e)}")
            return None

    def prefetch(self, blob_name: str, cache_dir: str) -> bool:
        """Pre-download blob to local cache for faster access."""
        cache_path = os.path.join(cache_dir, blob_name.replace('/', '_'))
        if not os.path.exists(cache_path):
            try:
                data = self.download_to_memory(blob_name)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    f.write(data)
                return True
            except Exception as e:
                logger.error(f"Prefetch failed for {blob_name}: {e}")
                return False
        return True

    def download_blob_to_file(self, blob_name: str, file_path: str) -> bool:
        """Download a blob to a local file."""
        for attempt in range(self.max_retries):
            try:
                with self.rate_limiter:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "wb") as f:
                        blob_client = self.container_client.get_blob_client(blob_name)
                        stream = blob_client.download_blob()
                        stream.readinto(f)
                    return True
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Error downloading blob {blob_name} to {file_path}: {e}")
                    return False
                logger.warning(f"Retry {attempt + 1}/{self.max_retries} for {blob_name}: {e}")
                time.sleep(2 ** attempt)

    def copy_blob(self, source_blob_name: str, destination_blob_name: str) -> bool:
        """Copy a blob within the same container."""
        for attempt in range(self.max_retries):
            try:
                with self.rate_limiter:
                    source_blob = self.container_client.get_blob_client(source_blob_name)
                    target_blob = self.container_client.get_blob_client(destination_blob_name)
                    
                    # Start copy operation
                    target_blob.start_copy_from_url(source_blob.url)
                    logger.info(f"Successfully initiated copy from {source_blob_name} to {destination_blob_name}")
                    return True
                    
            except ResourceNotFoundError:
                logger.error(f"Source blob not found: {source_blob_name}")
                return False
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Error copying blob {source_blob_name} to {destination_blob_name}: {e}")
                    return False
                logger.warning(f"Retry {attempt + 1}/{self.max_retries} for copy {source_blob_name}: {e}")
                time.sleep(2 ** attempt)
        
        return False

    def batch_download(self, blob_names: List[str], max_workers: int = None) -> Generator[tuple[str, Optional[np.ndarray]], None, None]:
        """Download multiple blobs concurrently with optimized threading."""
        if max_workers is None:
            # Calculate optimal number of workers based on system
            cpu_count = os.cpu_count() or 4
            # Increased maximum workers to 1000
            max_workers = min(1000, max(cpu_count * 50, self.max_concurrent_downloads))
        
        logger.info(f"Starting batch download with {max_workers} workers")
        start_time = time.time()
        completed = 0
        total = len(blob_names)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a dictionary to track futures
            future_to_blob = {
                executor.submit(self.download_as_cv2, blob_name): blob_name 
                for blob_name in blob_names
            }
            
            # Process completed futures as they come in
            for future in as_completed(future_to_blob):
                blob_name = future_to_blob[future]
                completed += 1
                try:
                    image = future.result()
                    if image is not None:
                        if completed % 10 == 0:  # Reduced logging frequency for better performance
                            logger.info(f"Progress: {completed}/{total} - Successfully downloaded {blob_name}")
                    else:
                        logger.warning(f"Progress: {completed}/{total} - Failed to download {blob_name}")
                    yield blob_name, image
                except Exception as e:
                    logger.error(f"Progress: {completed}/{total} - Error processing {blob_name}: {str(e)}")
                    yield blob_name, None
                
                # Log progress every 5% or 50 items for better performance
                if completed % max(1, total // 20) == 0 or completed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total - completed) / rate if rate > 0 else 0
                    logger.info(f"Progress: {completed}/{total} ({completed/total*100:.1f}%) - "
                              f"Rate: {rate:.1f} items/s - "
                              f"Est. remaining time: {remaining/60:.1f} minutes")

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
