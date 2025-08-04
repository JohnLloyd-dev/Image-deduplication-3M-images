import time
import threading
import logging
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)

class TokenBucketRateLimiter:
    """Rate limiter with dynamic rate adjustment based on system performance."""
    
    def __init__(self, initial_rate: float = 30.0, max_rate: float = 100.0):
        self.rate = initial_rate  # tokens per second
        self.max_rate = max_rate
        self.tokens = initial_rate
        self.last_update = time.time()
        self.lock = threading.Lock()
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)  # Keep last 100 processing times
        self.queue_sizes = deque(maxlen=100)  # Keep last 100 queue sizes
        self.last_adjustment = time.time()
        self.adjustment_interval = 5.0  # Adjust rate every 5 seconds
        
    def wait(self):
        """Wait for a token to become available."""
        with self.lock:
            now = time.time()
            
            # Update tokens
            time_passed = now - self.last_update
            self.tokens = min(self.max_rate, self.tokens + time_passed * self.rate)
            self.last_update = now
            
            # Check if we need to wait
            if self.tokens < 1.0:
                sleep_time = (1.0 - self.tokens) / self.rate
                time.sleep(sleep_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0
                
    def set_rate(self, new_rate: float):
        """Set a new rate, respecting max_rate."""
        with self.lock:
            self.rate = min(self.max_rate, max(0.1, new_rate))
            logger.info(f"Rate adjusted to {self.rate:.1f} requests/second")
            
    def record_processing_time(self, processing_time: float):
        """Record the time taken to process an item."""
        self.processing_times.append(processing_time)
        
    def record_queue_size(self, size: int):
        """Record the current queue size."""
        self.queue_sizes.append(size)
        
    def adjust_rate(self, target_processing_time: float = 0.1):
        """Adjust rate based on system performance."""
        now = time.time()
        if now - self.last_adjustment < self.adjustment_interval:
            return
            
        with self.lock:
            # Calculate average processing time
            if self.processing_times:
                avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            else:
                avg_processing_time = target_processing_time
                
            # Calculate average queue size
            if self.queue_sizes:
                avg_queue_size = sum(self.queue_sizes) / len(self.queue_sizes)
            else:
                avg_queue_size = 0
                
            # Adjust rate based on performance
            if avg_processing_time > target_processing_time * 1.5:
                # Processing is too slow, reduce rate
                new_rate = self.rate * 0.9
            elif avg_processing_time < target_processing_time * 0.5 and avg_queue_size < 10:
                # Processing is fast and queue is small, increase rate
                new_rate = min(self.max_rate, self.rate * 1.1)
            else:
                # Performance is good, maintain current rate
                new_rate = self.rate
                
            # Apply new rate
            self.set_rate(new_rate)
            self.last_adjustment = now
            
            # Log performance metrics
            logger.info(
                f"Performance metrics - "
                f"Avg processing time: {avg_processing_time:.3f}s, "
                f"Avg queue size: {avg_queue_size:.1f}, "
                f"Current rate: {self.rate:.1f} req/s"
            )
            
    def get_stats(self) -> dict:
        """Get current rate limiter statistics."""
        with self.lock:
            return {
                'current_rate': self.rate,
                'max_rate': self.max_rate,
                'available_tokens': self.tokens,
                'avg_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
                'avg_queue_size': sum(self.queue_sizes) / len(self.queue_sizes) if self.queue_sizes else 0
            } 