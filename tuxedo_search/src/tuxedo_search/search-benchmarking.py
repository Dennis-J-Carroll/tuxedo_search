"""
Comprehensive Search Engine Benchmarking and Optimization System
"""
import time
import asyncio
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import psutil
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import json
from pathlib import Path

@dataclass
class SearchMetrics:
    """Detailed search performance metrics"""
    query_time_ms: float
    indexing_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    result_count: int
    cache_hit_rate: float
    accuracy_score: float
    relevance_score: float

@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    metrics: List[SearchMetrics]
    mean_query_time: float
    p95_query_time: float
    p99_query_time: float
    mean_memory_usage: float
    cache_effectiveness: float
    optimization_suggestions: List[str]

class SearchBenchmark:
    """Benchmark and optimization system for the search engine"""
    
    def __init__(self, search_engine, config: Optional[Dict] = None):
        self.engine = search_engine
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Initialize performance monitoring"""
        self.process = psutil.Process()
        self.start_time = time.time()
        self.query_times = []
        self.cache_hits = defaultdict(int)
        self.cache_misses = defaultdict(int)
    
    async def run_benchmark(self, 
                          test_queries: List[str],
                          iterations: int = 1000,
                          parallel: bool = True) -> BenchmarkResult:
        """Run comprehensive benchmark suite"""
        all_metrics = []
        
        # Warm up the engine
        await self._warmup(test_queries[:10])
        
        # Run benchmarks
        if parallel:
            metrics = await self._run_parallel_benchmark(test_queries, iterations)
        else:
            metrics = await self._run_sequential_benchmark(test_queries, iterations)
        
        all_metrics.extend(metrics)
        
        # Analyze results
        benchmark_result = self._analyze_metrics(all_metrics)
        
        # Generate optimization suggestions
        benchmark_result.optimization_suggestions = (
            self._generate_optimization_suggestions(benchmark_result)
        )
        
        return benchmark_result
    
    async def profile_memory_usage(self, test_queries: List[str]) -> Dict:
        """Profile memory usage patterns"""
        import memory_profiler
        
        @memory_profiler.profile
        def _run_memory_test():
            results = []
            for query in test_queries:
                results.extend(self.engine.search(query))
            return results
        
        memory_stats = _run_memory_test()
        return self._analyze_memory_profile(memory_stats)
    
    async def analyze_cache_effectiveness(self) -> Dict:
        """Analyze cache hit rates and patterns"""
        total_queries = sum(self.cache_hits.values()) + sum(self.cache_misses.values())
        hit_rate = sum(self.cache_hits.values()) / total_queries if total_queries > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'cache_hits': dict(self.cache_hits),
            'cache_misses': dict(self.cache_misses),
            'suggestions': self._optimize_cache_config(hit_rate)
        }
    
    def visualize_performance(self, benchmark_result: BenchmarkResult) -> None:
        """Generate performance visualization"""
        plt.figure(figsize=(15, 10))
        
        # Query time distribution
        plt.subplot(2, 2, 1)
        query_times = [m.query_time_ms for m in benchmark_result.metrics]
        plt.hist(query_times, bins=50)
        plt.title('Query Time Distribution')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency')
        
        # Memory usage over time
        plt.subplot(2, 2, 2)
        memory_usage = [m.memory_usage_mb for m in benchmark_result.metrics]
        plt.plot(memory_usage)
        plt.title('Memory Usage Over Time')
        plt.xlabel('Query Number')
        plt.ylabel('Memory (MB)')
        
        # Cache hit rate
        plt.subplot(2, 2, 3)
        cache_rates = [m.cache_hit_rate for m in benchmark_result.metrics]
        plt.plot(cache_rates)
        plt.title('Cache Hit Rate')
        plt.xlabel('Query Number')
        plt.ylabel('Hit Rate')
        
        plt.tight_layout()
        plt.savefig('search_performance.png')
        plt.close()
    
    async def optimize_engine(self, benchmark_result: BenchmarkResult) -> None:
        """Automatically optimize engine based on benchmark results"""
        # 1. Adjust cache size
        if benchmark_result.cache_effectiveness < 0.7:
            new_cache_size = self.engine.cache_size * 1.5
            self.engine.resize_cache(int(new_cache_size))
        
        # 2. Tune thread pool
        if benchmark_result.mean_query_time > 100:  # ms
            optimal_threads = self._calculate_optimal_threads(benchmark_result)
            self.engine.resize_thread_pool(optimal_threads)
        
        # 3. Optimize index
        if benchmark_result.mean_memory_usage > 1000:  # MB
            await self._optimize_index()
        
        # 4. Adjust fuzzy matching threshold
        if benchmark_result.mean_query_time > 50:  # ms
            self._tune_fuzzy_threshold(benchmark_result)
    
    def generate_report(self, benchmark_result: BenchmarkResult) -> str:
        """Generate detailed performance report"""
        report = {
            'summary': {
                'mean_query_time': f"{benchmark_result.mean_query_time:.2f}ms",
                'p95_query_time': f"{benchmark_result.p95_query_time:.2f}ms",
                'p99_query_time': f"{benchmark_result.p99_query_time:.2f}ms",
                'memory_usage': f"{benchmark_result.mean_memory_usage:.1f}MB",
                'cache_effectiveness': f"{benchmark_result.cache_effectiveness:.1%}"
            },
            'optimization_suggestions': benchmark_result.optimization_suggestions,
            'detailed_metrics': self._generate_detailed_metrics(benchmark_result)
        }
        
        return json.dumps(report, indent=2)
    
    async def _warmup(self, queries: List[str]) -> None:
        """Warm up the engine before benchmarking"""
        for query in queries:
            await self.engine.search(query)
    
    async def _run_parallel_benchmark(self, 
                                    queries: List[str], 
                                    iterations: int) -> List[SearchMetrics]:
        """Run benchmark with parallel query execution"""
        metrics = []
        
        async def _run_query(query: str) -> SearchMetrics:
            start_time = time.perf_counter()
            results = await self.engine.search(query)
            query_time = (time.perf_counter() - start_time) * 1000
            
            return SearchMetrics(
                query_time_ms=query_time,
                indexing_time_ms=0,  # Set during indexing benchmark
                memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
                cpu_usage_percent=psutil.cpu_percent(),
                result_count=len(results),
                cache_hit_rate=self._calculate_cache_hit_rate(),
                accuracy_score=self._calculate_accuracy(results),
                relevance_score=self._calculate_relevance(results)
            )
        
        tasks = []
        for _ in range(iterations):
            for query in queries:
                tasks.append(_run_query(query))
        
        results = await asyncio.gather(*tasks)
        metrics.extend(results)
        
        return metrics
    
    def _analyze_metrics(self, metrics: List[SearchMetrics]) -> BenchmarkResult:
        """Analyze benchmark metrics"""
        query_times = [m.query_time_ms for m in metrics]
        
        return BenchmarkResult(
            metrics=metrics,
            mean_query_time=np.mean(query_times),
            p95_query_time=np.percentile(query_times, 95),
            p99_query_time=np.percentile(query_times, 99),
            mean_memory_usage=np.mean([m.memory_usage_mb for m in metrics]),
            cache_effectiveness=np.mean([m.cache_hit_rate for m in metrics]),
            optimization_suggestions=[]
        )
    
    def _generate_optimization_suggestions(self, 
                                        result: BenchmarkResult) -> List[str]:
        """Generate optimization suggestions based on benchmark results"""
        suggestions = []
        
        # Query time optimizations
        if result.mean_query_time > 100:  # ms
            suggestions.append(
                "High query times detected. Consider:\n"
                "- Increasing cache size\n"
                "- Optimizing fuzzy matching threshold\n"
                "- Adding more index shards"
            )
        
        # Memory optimizations
        if result.mean_memory_usage > 1000:  # MB
            suggestions.append(
                "High memory usage detected. Consider:\n"
                "- Implementing document compression\n"
                "- Reducing index size\n"
                "- Adjusting cache size"
            )
        
        # Cache optimizations
        if result.cache_effectiveness < 0.7:
            suggestions.append(
                "Low cache hit rate detected. Consider:\n"
                "- Increasing cache size\n"
                "- Adjusting cache eviction policy\n"
                "- Implementing predictive caching"
            )
        
        return suggestions
    
    async def _optimize_index(self) -> None:
        """Optimize search index"""
        # Compact index
        await self.engine.compact_index()
        
        # Optimize term dictionary
        await self.engine.optimize_terms()
        
        # Update statistics
        await self.engine.update_statistics()
    
    def _calculate_optimal_threads(self, 
                                 benchmark_result: BenchmarkResult) -> int:
        """Calculate optimal number of threads based on load"""
        cpu_cores = psutil.cpu_count()
        current_load = np.mean([m.cpu_usage_percent for m in benchmark_result.metrics])
        
        if current_load > 80:
            return max(4, cpu_cores - 2)  # Leave some cores for system
        else:
            return cpu_cores + 2  # Allow some oversubscription
    
    def _tune_fuzzy_threshold(self, benchmark_result: BenchmarkResult) -> None:
        """Tune fuzzy matching threshold based on performance"""
        current_threshold = self.engine.fuzzy_threshold
        mean_query_time = benchmark_result.mean_query_time
        
        if mean_query_time > 100:  # ms
            # Make matching stricter
            new_threshold = min(0.9, current_threshold + 0.1)
        else:
            # Can afford more fuzzy matches
            new_threshold = max(0.6, current_threshold - 0.1)
        
        self.engine.set_fuzzy_threshold(new_threshold)

# Usage example
async def main():
    from fastsearch import SearchEngine
    
    # Initialize engine and benchmark
    engine = SearchEngine()
    benchmark = SearchBenchmark(engine)
    
    # Test queries
    test_queries = [
        "machine learning",
        "python programming",
        "data science",
        # Add common misspellings
        "mashine lerning",
        "paython programing",
        # Add partial queries
        "mach learn",
        "py prog",
    ]
    
    # Run benchmark
    print("Running benchmark...")
    result = await benchmark.run_benchmark(
        test_queries,
        iterations=1000,
        parallel=True
    )
    
    # Generate visualization
    benchmark.visualize_performance(result)
    
    # Generate report
    report = benchmark.generate_report(result)
    print("\nBenchmark Report:")
    print(report)
    
    # Optimize based on results
    print("\nOptimizing engine...")
    await benchmark.optimize_engine(result)
    
    # Run benchmark again to verify improvements
    print("\nRe-running benchmark after optimization...")
    new_result = await benchmark.run_benchmark(
        test_queries,
        iterations=1000,
        parallel=True
    )
    
    print("\nImprovement Summary:")
    print(f"Query Time: {result.mean_query_time:.2f}ms -> {new_result.mean_query_time:.2f}ms")
    print(f"Memory Usage: {result.mean_memory_usage:.1f}MB -> {new_result.mean_memory_usage:.1f}MB")
    print(f"Cache Effectiveness: {result.cache_effectiveness:.1%} -> {new_result.cache_effectiveness:.1%}")

if __name__ == "__main__":
    asyncio.run(main())