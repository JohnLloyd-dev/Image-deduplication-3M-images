#!/usr/bin/env python3
"""
Threading Optimizer

This module automatically detects optimal threading configuration
for the multi-threaded deduplication pipeline based on system resources.
"""

import os
import sys
import logging
import platform
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class ThreadingOptimizer:
    """Automatically optimize threading configuration for the system."""
    
    def __init__(self):
        self.system_info = self._detect_system_info()
        self.optimal_config = self._calculate_optimal_config()
    
    def _detect_system_info(self) -> Dict:
        """Detect system hardware and capabilities."""
        
        info = {
            'cpu_count_physical': 1,
            'cpu_count_logical': 1,
            'total_memory_gb': 4,
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'architecture': platform.machine()
        }
        
        # Detect CPU cores
        try:
            import psutil
            info['cpu_count_physical'] = psutil.cpu_count(logical=False) or 1
            info['cpu_count_logical'] = psutil.cpu_count(logical=True) or 1
            
            # Detect memory
            memory_info = psutil.virtual_memory()
            info['total_memory_gb'] = memory_info.total / (1024**3)
            
            # Detect CPU frequency
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                info['cpu_base_freq'] = cpu_freq.current
                info['cpu_max_freq'] = cpu_freq.max
            
        except ImportError:
            # Fallback to os.cpu_count()
            info['cpu_count_logical'] = os.cpu_count() or 1
            info['cpu_count_physical'] = info['cpu_count_logical']
            logger.warning("psutil not available, using basic CPU detection")
        
        return info
    
    def _calculate_optimal_config(self) -> Dict:
        """Calculate optimal threading configuration."""
        
        cpu_physical = self.system_info['cpu_count_physical']
        cpu_logical = self.system_info['cpu_count_logical']
        memory_gb = self.system_info['total_memory_gb']
        
        # Base configuration on system class
        if cpu_logical <= 2:
            # Low-end system (laptops, old computers)
            config_class = "low_end"
            max_workers = 2
            chunk_size = 3
            memory_conservative = True
            
        elif cpu_logical <= 4:
            # Entry-level system (basic desktops)
            config_class = "entry_level"
            max_workers = cpu_logical
            chunk_size = 5
            memory_conservative = True
            
        elif cpu_logical <= 8:
            # Mid-range system (modern desktops, workstations)
            config_class = "mid_range"
            max_workers = cpu_logical
            chunk_size = 8
            memory_conservative = False
            
        elif cpu_logical <= 16:
            # High-end system (gaming PCs, workstations)
            config_class = "high_end"
            max_workers = min(12, cpu_logical)  # Don't use all cores
            chunk_size = 12
            memory_conservative = False
            
        else:
            # Server-class system (high-end workstations, servers)
            config_class = "server_class"
            max_workers = min(16, cpu_logical)  # Cap at 16 for diminishing returns
            chunk_size = 15
            memory_conservative = False
        
        # Adjust for memory constraints
        if memory_gb < 8:
            # Low memory system
            max_workers = min(max_workers, 4)
            chunk_size = min(chunk_size, 5)
            memory_conservative = True
            
        elif memory_gb < 16:
            # Moderate memory system
            max_workers = min(max_workers, 8)
            chunk_size = min(chunk_size, 10)
        
        # Platform-specific adjustments
        if self.system_info['platform'] == 'Windows':
            # Windows has higher thread overhead
            max_workers = max(1, max_workers - 1)
            
        elif self.system_info['platform'] == 'Darwin':  # macOS
            # macOS has efficient threading but thermal constraints
            if cpu_logical > 8:
                max_workers = min(max_workers, 8)  # Thermal throttling prevention
        
        return {
            'config_class': config_class,
            'max_workers': max_workers,
            'chunk_size': chunk_size,
            'memory_conservative': memory_conservative,
            'reasoning': self._generate_reasoning(config_class, max_workers, chunk_size, memory_conservative)
        }
    
    def _generate_reasoning(self, config_class: str, max_workers: int, 
                          chunk_size: int, memory_conservative: bool) -> str:
        """Generate human-readable reasoning for the configuration."""
        
        cpu_logical = self.system_info['cpu_count_logical']
        memory_gb = self.system_info['total_memory_gb']
        
        reasoning = f"System class: {config_class.replace('_', ' ').title()}\n"
        reasoning += f"- {cpu_logical} logical CPU cores detected\n"
        reasoning += f"- {memory_gb:.1f}GB total memory available\n"
        reasoning += f"- Configured for {max_workers} worker threads\n"
        reasoning += f"- Batch size: {chunk_size} groups per thread\n"
        
        if memory_conservative:
            reasoning += "- Memory-conservative mode enabled\n"
        
        # Add specific recommendations
        if config_class == "low_end":
            reasoning += "- Optimized for basic systems with limited resources\n"
        elif config_class == "entry_level":
            reasoning += "- Balanced configuration for entry-level systems\n"
        elif config_class == "mid_range":
            reasoning += "- Performance-optimized for modern desktop systems\n"
        elif config_class == "high_end":
            reasoning += "- High-performance configuration for powerful systems\n"
        elif config_class == "server_class":
            reasoning += "- Server-optimized configuration with controlled resource usage\n"
        
        return reasoning
    
    def get_optimal_config(self) -> Dict:
        """Get the optimal threading configuration."""
        return self.optimal_config.copy()
    
    def get_system_info(self) -> Dict:
        """Get detected system information."""
        return self.system_info.copy()
    
    def print_system_analysis(self):
        """Print detailed system analysis and recommendations."""
        
        print("=" * 60)
        print("THREADING OPTIMIZER - SYSTEM ANALYSIS")
        print("=" * 60)
        
        print(f"\n🖥️  System Information:")
        print(f"   Platform: {self.system_info['platform']}")
        print(f"   Architecture: {self.system_info['architecture']}")
        print(f"   Python Version: {self.system_info['python_version']}")
        print(f"   Physical CPU Cores: {self.system_info['cpu_count_physical']}")
        print(f"   Logical CPU Cores: {self.system_info['cpu_count_logical']}")
        print(f"   Total Memory: {self.system_info['total_memory_gb']:.1f} GB")
        
        if 'cpu_base_freq' in self.system_info:
            print(f"   CPU Base Frequency: {self.system_info['cpu_base_freq']:.0f} MHz")
            print(f"   CPU Max Frequency: {self.system_info['cpu_max_freq']:.0f} MHz")
        
        print(f"\n⚙️  Optimal Configuration:")
        config = self.optimal_config
        print(f"   Configuration Class: {config['config_class'].replace('_', ' ').title()}")
        print(f"   Max Workers: {config['max_workers']}")
        print(f"   Chunk Size: {config['chunk_size']}")
        print(f"   Memory Conservative: {config['memory_conservative']}")
        
        print(f"\n💡 Reasoning:")
        for line in config['reasoning'].split('\n'):
            if line.strip():
                print(f"   {line}")
        
        print(f"\n🚀 Expected Performance:")
        self._print_performance_estimates()
        
        print(f"\n📋 Usage Example:")
        print(f"   from modules.multithreaded_deduplication import MultiThreadedDeduplicator")
        print(f"   from modules.threading_optimizer import ThreadingOptimizer")
        print(f"   ")
        print(f"   optimizer = ThreadingOptimizer()")
        print(f"   config = optimizer.get_optimal_config()")
        print(f"   ")
        print(f"   deduplicator = MultiThreadedDeduplicator(")
        print(f"       feature_cache=cache,")
        print(f"       max_workers=config['max_workers'],")
        print(f"       chunk_size=config['chunk_size']")
        print(f"   )")
    
    def _print_performance_estimates(self):
        """Print estimated performance improvements."""
        
        max_workers = self.optimal_config['max_workers']
        
        # Estimate speedup (accounting for Amdahl's law and overhead)
        if max_workers == 1:
            speedup = 1.0
            efficiency = 100
        else:
            # Assume 80% of work is parallelizable
            parallel_fraction = 0.8
            overhead_factor = 0.9  # 10% overhead
            
            speedup = 1 / ((1 - parallel_fraction) + (parallel_fraction / max_workers))
            speedup *= overhead_factor
            efficiency = (speedup / max_workers) * 100
        
        print(f"   Estimated Speedup: {speedup:.1f}x")
        print(f"   Threading Efficiency: {efficiency:.0f}%")
        
        # Performance estimates for different dataset sizes
        base_times = {1000: 20, 5000: 100, 10000: 200, 50000: 1000}  # Single-threaded estimates
        
        print(f"   Processing Time Estimates:")
        for images, single_time in base_times.items():
            multi_time = single_time / speedup
            print(f"     {images:,} images: {single_time:.0f}s → {multi_time:.0f}s ({speedup:.1f}x faster)")


def get_optimal_threading_config() -> Tuple[int, int, bool]:
    """
    Get optimal threading configuration for the current system.
    
    Returns:
        Tuple of (max_workers, chunk_size, memory_conservative)
    """
    optimizer = ThreadingOptimizer()
    config = optimizer.get_optimal_config()
    
    return (
        config['max_workers'],
        config['chunk_size'], 
        config['memory_conservative']
    )


def create_optimized_deduplicator(feature_cache, device="cpu"):
    """
    Create a multi-threaded deduplicator with optimal configuration.
    
    Args:
        feature_cache: Feature cache instance
        device: Processing device ("cpu" or "cuda")
        
    Returns:
        Optimally configured MultiThreadedDeduplicator
    """
    from modules.multithreaded_deduplication import MultiThreadedDeduplicator
    
    max_workers, chunk_size, memory_conservative = get_optimal_threading_config()
    
    logger.info(f"🚀 Creating optimized multi-threaded deduplicator:")
    logger.info(f"   - Max workers: {max_workers}")
    logger.info(f"   - Chunk size: {chunk_size}")
    logger.info(f"   - Memory conservative: {memory_conservative}")
    logger.info(f"   - Device: {device}")
    
    return MultiThreadedDeduplicator(
        feature_cache=feature_cache,
        device=device,
        max_workers=max_workers,
        chunk_size=chunk_size
    )


if __name__ == "__main__":
    # Run system analysis
    optimizer = ThreadingOptimizer()
    optimizer.print_system_analysis()