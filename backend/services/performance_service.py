#!/usr/bin/env python3
"""
Performance optimization service for Stock Prediction API
Handles caching, response optimization, and model inference acceleration
"""

import asyncio
import time
import json
import hashlib
from typing import Any, Dict, List, Optional, Callable
from functools import wraps, lru_cache
from datetime import datetime, timedelta
import logging

import redis
import pandas as pd
import numpy as np
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import gzip
import asyncio
from contextlib import asynccontextmanager

from ..core.config import get_settings

logger = logging.getLogger(__name__)

class CacheManager:
    """Advanced caching manager with Redis and in-memory fallback"""
    
    def __init__(self, redis_url: Optional[str] = None, default_ttl: int = 3600):
        self.settings = get_settings()
        self.default_ttl = default_ttl
        self.redis_client = None
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'errors': 0
        }
        
        # Use Redis URL from config if not provided
        redis_url = redis_url or self.settings.redis_url
        
        try:
            if redis_url:
                self.redis_client = redis.from_url(
                    redis_url, 
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                self.redis_client.ping()
                logger.info(f"Redis cache initialized successfully at {redis_url}")
            else:
                logger.warning("No Redis URL configured, using memory cache only")
        except Exception as e:
            logger.warning(f"Redis not available, using memory cache: {e}")
            self.redis_client = None
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = f"{prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for caching with compression"""
        try:
            # Handle pandas DataFrames
            if isinstance(data, pd.DataFrame):
                serialized = data.to_json(orient='records', date_format='iso')
                return gzip.compress(serialized.encode())
            
            # Handle numpy arrays
            elif isinstance(data, np.ndarray):
                serialized = pickle.dumps(data)
                return gzip.compress(serialized)
            
            # Handle regular objects
            else:
                serialized = json.dumps(data, default=str)
                return gzip.compress(serialized.encode())
                
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return gzip.compress(pickle.dumps(data))
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize cached data"""
        try:
            decompressed = gzip.decompress(data)
            
            # Try JSON first
            try:
                return json.loads(decompressed.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fall back to pickle
                return pickle.loads(decompressed)
                
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            # Try Redis first
            if self.redis_client:
                try:
                    data = await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.get, key
                    )
                    if data:
                        self.cache_stats['hits'] += 1
                        return self._deserialize_data(data)
                except Exception as e:
                    logger.error(f"Redis get error: {e}")
            
            # Fall back to memory cache
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if entry['expires'] > datetime.now():
                    self.cache_stats['hits'] += 1
                    return entry['data']
                else:
                    del self.memory_cache[key]
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_stats['errors'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or self.default_ttl
            serialized_data = self._serialize_data(value)
            
            # Try Redis first
            if self.redis_client:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.setex, key, ttl, serialized_data
                    )
                    self.cache_stats['sets'] += 1
                    return True
                except Exception as e:
                    logger.error(f"Redis set error: {e}")
            
            # Fall back to memory cache
            self.memory_cache[key] = {
                'data': value,
                'expires': datetime.now() + timedelta(seconds=ttl)
            }
            
            # Cleanup old entries
            if len(self.memory_cache) > 1000:
                self._cleanup_memory_cache()
            
            self.cache_stats['sets'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self.cache_stats['errors'] += 1
            return False
    
    def _cleanup_memory_cache(self):
        """Clean up expired entries from memory cache"""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if entry['expires'] <= now
        ]
        for key in expired_keys:
            del self.memory_cache[key]
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.redis_client:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.delete, key
                )
            
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis connection health"""
        redis_status = {
            'available': False,
            'latency_ms': None,
            'memory_usage': None,
            'error': None
        }
        
        if self.redis_client:
            try:
                start_time = time.time()
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.ping
                )
                redis_status['latency_ms'] = round((time.time() - start_time) * 1000, 2)
                redis_status['available'] = True
                
                # Get Redis memory info
                info = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.info, 'memory'
                )
                redis_status['memory_usage'] = info.get('used_memory_human', 'Unknown')
                
            except Exception as e:
                redis_status['error'] = str(e)
                logger.error(f"Redis health check failed: {e}")
        
        return redis_status
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            'hit_rate': round(hit_rate, 2),
            'memory_cache_size': len(self.memory_cache),
            'redis_available': self.redis_client is not None,
            'default_ttl': self.default_ttl
        }

class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        self.request_times = []
        
    def cache_result(self, ttl: int = 3600, key_prefix: str = "api"):
        """Decorator for caching function results"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self.cache._generate_key(f"{key_prefix}:{func.__name__}", *args, **kwargs)
                
                # Try to get from cache
                cached_result = await self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                await self.cache.set(cache_key, result, ttl)
                return result
            
            return wrapper
        return decorator
    
    def async_executor(self, executor_type: str = "thread"):
        """Decorator for running CPU-intensive tasks in executor"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                executor = self.thread_pool if executor_type == "thread" else self.process_pool
                
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(executor, func, *args, **kwargs)
                return result
            
            return wrapper
        return decorator
    
    def measure_performance(self, func: Callable):
        """Decorator for measuring function performance"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                self.request_times.append(execution_time)
                
                # Keep only last 1000 measurements
                if len(self.request_times) > 1000:
                    self.request_times = self.request_times[-1000:]
                
                logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
                raise
        
        return wrapper
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.request_times:
            return {"message": "No performance data available"}
        
        times = np.array(self.request_times)
        return {
            'total_requests': len(times),
            'avg_response_time': round(np.mean(times), 3),
            'median_response_time': round(np.median(times), 3),
            'min_response_time': round(np.min(times), 3),
            'max_response_time': round(np.max(times), 3),
            'p95_response_time': round(np.percentile(times, 95), 3),
            'p99_response_time': round(np.percentile(times, 99), 3)
        }

class ModelInferenceOptimizer:
    """Optimize ML model inference performance"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.model_cache = {}
        self.batch_queue = asyncio.Queue(maxsize=100)
        self.batch_processor_task = None
        
    async def start_batch_processor(self):
        """Start batch processing for model inference"""
        if self.batch_processor_task is None:
            self.batch_processor_task = asyncio.create_task(self._batch_processor())
    
    async def stop_batch_processor(self):
        """Stop batch processing"""
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
            self.batch_processor_task = None
    
    async def _batch_processor(self):
        """Process batched inference requests"""
        batch = []
        batch_timeout = 0.1  # 100ms timeout
        
        while True:
            try:
                # Wait for requests with timeout
                try:
                    request = await asyncio.wait_for(
                        self.batch_queue.get(), timeout=batch_timeout
                    )
                    batch.append(request)
                except asyncio.TimeoutError:
                    pass
                
                # Process batch if we have requests or timeout occurred
                if batch and (len(batch) >= 10 or time.time() - batch[0]['timestamp'] > batch_timeout):
                    await self._process_batch(batch)
                    batch = []
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                batch = []  # Clear batch on error
    
    async def _process_batch(self, batch: List[Dict]):
        """Process a batch of inference requests"""
        try:
            # Group by model type
            model_batches = {}
            for request in batch:
                model_type = request['model_type']
                if model_type not in model_batches:
                    model_batches[model_type] = []
                model_batches[model_type].append(request)
            
            # Process each model type batch
            for model_type, requests in model_batches.items():
                try:
                    # Extract features for batch processing
                    features_batch = [req['features'] for req in requests]
                    
                    # Run batch inference
                    predictions = await self._run_batch_inference(model_type, features_batch)
                    
                    # Set results for each request
                    for request, prediction in zip(requests, predictions):
                        request['future'].set_result(prediction)
                        
                except Exception as e:
                    logger.error(f"Batch inference error for {model_type}: {e}")
                    # Set error for all requests in this batch
                    for request in requests:
                        request['future'].set_exception(e)
                        
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
    
    async def _run_batch_inference(self, model_type: str, features_batch: List[np.ndarray]) -> List[Any]:
        """Run batch inference for a specific model type"""
        # This would be implemented based on your specific models
        # For now, return dummy predictions
        return [{'prediction': i, 'confidence': 0.8} for i in range(len(features_batch))]
    
    async def predict_async(self, model_type: str, features: np.ndarray) -> Any:
        """Async prediction with batching support"""
        # Check cache first
        cache_key = self.cache._generate_key(f"prediction:{model_type}", features.tobytes())
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Create future for result
        future = asyncio.Future()
        
        # Add to batch queue
        request = {
            'model_type': model_type,
            'features': features,
            'future': future,
            'timestamp': time.time()
        }
        
        try:
            await self.batch_queue.put(request)
            result = await future
            
            # Cache result
            await self.cache.set(cache_key, result, ttl=300)  # 5 minutes
            return result
            
        except asyncio.QueueFull:
            # Fall back to direct prediction if queue is full
            logger.warning("Batch queue full, falling back to direct prediction")
            return await self._run_batch_inference(model_type, [features])

class ResponseOptimizer:
    """Optimize API response formatting and compression"""
    
    @staticmethod
    def optimize_dataframe_response(df: pd.DataFrame, max_rows: int = 1000) -> Dict[str, Any]:
        """Optimize DataFrame for API response"""
        # Limit rows
        if len(df) > max_rows:
            df = df.tail(max_rows)
        
        # Convert to efficient format
        result = {
            'data': df.to_dict('records'),
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        return result
    
    @staticmethod
    def compress_response(data: Any, compression_threshold: int = 1024) -> JSONResponse:
        """Compress response if it exceeds threshold"""
        json_str = json.dumps(data, default=str)
        
        if len(json_str) > compression_threshold:
            # Use gzip compression
            compressed_data = gzip.compress(json_str.encode())
            
            return JSONResponse(
                content=data,
                headers={
                    'Content-Encoding': 'gzip',
                    'Content-Length': str(len(compressed_data))
                }
            )
        
        return JSONResponse(content=data)
    
    @staticmethod
    def paginate_response(data: List[Any], page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        """Paginate large responses"""
        total_items = len(data)
        total_pages = (total_items + page_size - 1) // page_size
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        return {
            'data': data[start_idx:end_idx],
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_items': total_items,
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            }
        }

# Global instances - initialized with config settings
settings = get_settings()
cache_manager = CacheManager(default_ttl=settings.cache_ttl)
performance_optimizer = PerformanceOptimizer(cache_manager)
model_optimizer = ModelInferenceOptimizer(cache_manager)
response_optimizer = ResponseOptimizer()

# Context manager for performance optimization
@asynccontextmanager
async def performance_context():
    """Context manager for performance optimization lifecycle"""
    try:
        await model_optimizer.start_batch_processor()
        logger.info("Performance optimization started")
        yield {
            'cache': cache_manager,
            'performance': performance_optimizer,
            'model': model_optimizer,
            'response': response_optimizer
        }
    finally:
        await model_optimizer.stop_batch_processor()
        logger.info("Performance optimization stopped")

# Utility functions
async def get_performance_metrics() -> Dict[str, Any]:
    """Get comprehensive performance metrics"""
    return {
        'cache_stats': cache_manager.get_stats(),
        'performance_stats': performance_optimizer.get_performance_stats(),
        'timestamp': datetime.now().isoformat()
    }

async def clear_all_caches() -> bool:
    """Clear all caches"""
    try:
        if cache_manager.redis_client:
            await asyncio.get_event_loop().run_in_executor(
                None, cache_manager.redis_client.flushdb
            )
        
        cache_manager.memory_cache.clear()
        logger.info("All caches cleared")
        return True
    except Exception as e:
        logger.error(f"Error clearing caches: {e}")
        return False

# Decorators for easy use
def cached(ttl: int = 3600, key_prefix: str = "api"):
    """Convenience decorator for caching"""
    return performance_optimizer.cache_result(ttl=ttl, key_prefix=key_prefix)

def measured():
    """Convenience decorator for performance measurement"""
    return performance_optimizer.measure_performance

def async_task(executor_type: str = "thread"):
    """Convenience decorator for async execution"""
    return performance_optimizer.async_executor(executor_type=executor_type)