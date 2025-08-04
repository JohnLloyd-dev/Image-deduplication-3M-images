import multiprocessing as mp
import numpy as np
import torch
from typing import Dict, List, Optional
import logging
from modules.feature_extraction import FeatureExtractor
from modules.feature_cache import BoundedFeatureCache

logger = logging.getLogger(__name__)

class GPUWorker:
    """Process for GPU-based feature extraction."""
    
    def __init__(self, device_id: int, input_queue: mp.Queue, output_queue: mp.Queue):
        self.device = f'cuda:{device_id}'
        self.feature_extractor = FeatureExtractor(device=self.device)
        self.input_queue = input_queue
        self.output_queue = output_queue
        logger.info(f"GPU Worker {device_id} initialized on {self.device}")
        
    def run(self):
        """Main worker loop."""
        try:
            while True:
                # Get batch from input queue
                batch = self.input_queue.get()
                if batch is None:  # Poison pill
                    break
                    
                try:
                    # Process each image in the batch
                    features = []
                    for image in batch['images']:
                        feature_dict = self.feature_extractor.extract_features(image)
                        features.append(feature_dict)
                    
                    # Send results
                    self.output_queue.put({
                        'batch_id': batch['batch_id'],
                        'features': features,
                        'paths': batch['paths']
                    })
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    self.output_queue.put({
                        'batch_id': batch['batch_id'],
                        'error': str(e)
                    })
                    
        except Exception as e:
            logger.error(f"GPU Worker error: {str(e)}")
        finally:
            logger.info(f"GPU Worker {self.device} shutting down")

class DistributedProcessor:
    """Manages distributed processing using multiprocessing."""
    
    def __init__(self, num_gpus: int = None):
        # Get available GPUs
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
            
        self.num_gpus = num_gpus
        logger.info(f"Initializing distributed processor with {num_gpus} GPUs")
        
        # Initialize queues
        self.input_queues = [mp.Queue() for _ in range(num_gpus)]
        self.output_queue = mp.Queue()
        
        # Start worker processes
        self.workers = []
        for i in range(num_gpus):
            worker = GPUWorker(i, self.input_queues[i], self.output_queue)
            process = mp.Process(target=worker.run)
            process.start()
            self.workers.append(process)
            
        # Track active batches
        self.active_batches = {}
        self.next_batch_id = 0
        
    def process_images(self, images: List[Dict]) -> List[Dict]:
        """Process a list of images using distributed workers."""
        try:
            # Split images among workers
            batch_size = len(images) // self.num_gpus
            if batch_size == 0:
                batch_size = 1
                
            # Create batches
            batches = []
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                batches.append(batch)
                
            # Process batches
            for batch in batches:
                batch_id = self.next_batch_id
                self.next_batch_id += 1
                
                # Prepare batch data
                batch_data = {
                    'batch_id': batch_id,
                    'images': [item['image_data'] for item in batch],
                    'paths': [item['azure_path'] for item in batch]
                }
                
                # Send to worker
                worker_idx = batch_id % self.num_gpus
                self.input_queues[worker_idx].put(batch_data)
                self.active_batches[batch_id] = batch
                
            # Collect results
            results = []
            while self.active_batches:
                try:
                    result = self.output_queue.get(timeout=1)
                    batch_id = result['batch_id']
                    
                    if 'error' in result:
                        logger.error(f"Batch {batch_id} failed: {result['error']}")
                        continue
                        
                    # Process results
                    batch = self.active_batches.pop(batch_id)
                    for path, feature in zip(result['paths'], result['features']):
                        # Add to results
                        results.append({
                            'azure_path': path,
                            'features': feature
                        })
                        
                except mp.queues.Empty:
                    continue
                    
            return results
            
        except Exception as e:
            logger.error(f"Distributed processing error: {str(e)}")
            raise
            
    def shutdown(self):
        """Shutdown worker processes."""
        try:
            # Send poison pills
            for queue in self.input_queues:
                queue.put(None)
                
            # Wait for workers to finish
            for worker in self.workers:
                worker.join()
                
            # Clean up queues
            for queue in self.input_queues:
                queue.close()
            self.output_queue.close()
            
        except Exception as e:
            logger.error(f"Error shutting down workers: {str(e)}")
            
    def get_stats(self) -> Dict:
        """Get statistics about the distributed processing."""
        return {
            'num_gpus': self.num_gpus
        } 