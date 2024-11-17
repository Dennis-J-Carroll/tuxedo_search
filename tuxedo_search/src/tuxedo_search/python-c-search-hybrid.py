#!/usr/bin/env python3
"""
High-Performance Search Engine using Python-C Hybrid Approach
"""
from libc.stdlib cimport malloc, free
from libc.string cimport strcpy, strlen
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from collections import deque
import xxhash
from functools import lru_cache
from cachetools import TTLCache
from psutil import cpu_count
from prometheus_client import Histogram, Counter, Gauge
from contextlib import contextmanager

# Cython wrapper for the C search engine
cdef extern from "search_engine.h":
    ctypedef struct SearchEngine:
        pass
    
    ctypedef struct SearchResult:
        char* doc_id
        float score
        char* snippet
    
    SearchEngine* create_search_engine()
    void destroy_search_engine(SearchEngine* engine)
    int index_document(SearchEngine* engine, const char* content, const char* doc_id)
    SearchResult* search(SearchEngine* engine, const char* query, int* num_results)
    void free_search_results(SearchResult* results)

# Add SIMD support for vector operations
cdef extern from "search_engine_simd.h":
    void vector_similarity_simd(
        const float* vec1,
        const float* vec2,
        size_t dim,
        float* result
    )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_similarity(self, np.ndarray[np.float32_t, ndim=1] vec1,
                               np.ndarray[np.float32_t, ndim=1] vec2):
        """Optimized vector similarity computation"""
        cdef float result = 0
        vector_similarity_simd(
            <float*>vec1.data,
            <float*>vec2.data,
            vec1.shape[0],
            &result
        )
        return result

cdef class PySearchEngine:
    """Cython wrapper class for the C search engine"""
    cdef SearchEngine* _engine
    
    def __cinit__(self):
        self._engine = create_search_engine()
        if self._engine is NULL:
            raise MemoryError("Failed to create search engine")
    
    def __dealloc__(self):
        if self._engine is not NULL:
            destroy_search_engine(self._engine)
    
    def index_document(self, str content, str doc_id) -> int:
        """Index a single document"""
        return index_document(
            self._engine,
            content.encode('utf-8'),
            doc_id.encode('utf-8')
        )
    
    def search(self, str query) -> List[Dict]:
        """Perform a search query"""
        cdef int num_results = 0
        cdef SearchResult* results = search(
            self._engine,
            query.encode('utf-8'),
            &num_results
        )
        
        if results is NULL:
            return []
        
        try:
            return [
                {
                    'doc_id': results[i].doc_id.decode('utf-8'),
                    'score': results[i].score,
                    'snippet': results[i].snippet.decode('utf-8')
                }
                for i in range(num_results)
            ]
        finally:
            free_search_results(results)

class SearchEngineError(Exception):
    """Custom exception for search engine errors"""
    pass

class SearchEngine:
    """Enhanced search engine with better error handling"""
    
    def __init__(self):
        try:
            self.engine = PySearchEngine()
        except Exception as e:
            raise SearchEngineError(f"Failed to initialize search engine: {e}")
            
    async def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Enhanced search with error handling"""
        try:
            results = await self._execute_search(query, limit)
            return self._process_results(results)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchEngineError(f"Search operation failed: {e}")

class BatchProcessor:
    """Efficient batch document processing"""
    
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.queue = asyncio.Queue()
        
    async def process_batch(self, documents: List[Dict]) -> None:
        """Process documents in optimized batches"""
        batches = [
            documents[i:i + self.batch_size]
            for i in range(0, len(documents), self.batch_size)
        ]
        
        async with ThreadPoolExecutor() as executor:
            tasks = [
                self._process_batch(batch, executor)
                for batch in batches
            ]
            await asyncio.gather(*tasks)

class EnhancedMetrics:
    """Comprehensive performance monitoring"""
    
    def __init__(self):
        self.metrics = {
            'latency': Histogram('search_latency_seconds', buckets=(
                0.001, 0.005, 0.01, 0.05, 0.1, 0.5
            )),
            'cache_hits': Counter('cache_hits_total'),
            'simd_operations': Counter('simd_operations_total'),
            'memory_usage': Gauge('memory_usage_bytes')
        }
        
    @contextmanager
    def measure_operation(self, operation_name: str):
        """Measure operation performance"""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.metrics['latency'].observe(duration)
            self._update_memory_metrics()

# FastAPI integration
app = FastAPI(title="High-Performance Search API")

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

class Document(BaseModel):
    id: str
    content: str

class BulkIndexRequest(BaseModel):
    documents: List[Document]

# Initialize search engine
search_engine = SearchEngine()

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Enhanced search endpoint with validation"""
    try:
        results = await search_engine.search(
            query=request.query,
            limit=min(request.limit, 100)  # Enforce reasonable limits
        )
        return SearchResponse(
            results=results,
            metadata={
                'total_hits': len(results),
                'query_time_ms': search_engine.metrics.get_last_latency()
            }
        )
    except SearchEngineError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/index")
async def index_document(document: Document):
    """Index a single document"""
    success = search_engine.engine.index_document(document.content, document.id)
    return {"success": success == 0}

@app.post("/bulk-index")
async def bulk_index(request: BulkIndexRequest):
    """Bulk index multiple documents"""
    docs = [{"id": doc.id, "content": doc.content} for doc in request.documents]
    search_engine.bulk_index(docs)
    return {"success": True}

# Example usage
if __name__ == "__main__":
    # Create engine
    engine = SearchEngine()
    
    # Example documents
    documents = [
        {
            "id": "doc1",
            "content": "Python and C hybrid programming for high performance"
        },
        {
            "id": "doc2",
            "content": "Fast search engines using SIMD optimizations"
        }
    ]
    
    # Index documents
    engine.bulk_index(documents)
    
    # Example search
    results = engine.search("python performance")
    print("Search Results:", results)
    
    # Run API server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

class OptimizedMemoryPool:
    """Enhanced memory pool with size-based optimization"""
    
    def __init__(self):
        self.pools = {
            'tiny': deque(maxlen=10000),    # < 1KB
            'small': deque(maxlen=5000),    # < 10KB
            'medium': deque(maxlen=1000),   # < 100KB
            'large': deque(maxlen=100)      # >= 100KB
        }
        self.size_thresholds = {
            'tiny': 1024,
            'small': 10240,
            'medium': 102400
        }
        
    def get_buffer(self, size: int) -> memoryview:
        """Get memory buffer with optimal size"""
        pool_type = self._get_pool_type(size)
        pool = self.pools[pool_type]
        
        if pool:
            return pool.popleft()
        
        # Allocate new buffer with alignment
        return self._allocate_aligned(size)

class QueryProcessor:
    """Optimized query processing pipeline"""
    
    def __init__(self):
        self.vectorizer = self._init_vectorizer()
        self.cache = TTLCache(maxsize=10000, ttl=3600)
        
    async def process_query(self, query: str) -> np.ndarray:
        """Process query with optimizations"""
        cache_key = xxhash.xxh64(query.encode()).hexdigest()
        
        if cached := self.cache.get(cache_key):
            return cached
            
        # Pipeline stages with early termination
        for processor in self.processors:
            if result := await processor.process(query):
                self.cache[cache_key] = result
                return result

class OptimizedThreadPool:
    """Enhanced thread pool with adaptive sizing"""
    
    def __init__(self):
        self.min_workers = max(2, cpu_count() // 2)
        self.max_workers = cpu_count() * 2
        self.current_workers = self.min_workers
        self.pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
    async def execute(self, func, *args):
        """Execute with adaptive thread count"""
        if self.should_scale_up():
            self.current_workers = min(
                self.current_workers + 2,
                self.max_workers
            )
        
        return await asyncio.get_event_loop().run_in_executor(
            self.pool, func, *args
        )

@cython.boundscheck(False)
def prefetch_vectors(self, queries: List[str]):
    """Prefetch vectors for upcoming queries"""
    predicted_vectors = self.predictor.predict_next_queries(queries)
    for vector in predicted_vectors:
        self.vector_cache.prefetch(vector)

def optimize_batch_size(self, current_latency: float) -> int:
    """Dynamically adjust batch size based on performance"""
    if current_latency > self.target_latency:
        return max(self.batch_size // 2, self.min_batch_size)
    return min(self.batch_size * 2, self.max_batch_size)