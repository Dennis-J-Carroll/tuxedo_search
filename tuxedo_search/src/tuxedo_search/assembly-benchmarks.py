"""
Comprehensive Benchmark System for Assembly vs Python/C Implementations
"""
import time
import numpy as np
from typing import Dict, List, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import ctypes
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

@dataclass
class BenchmarkResult:
    operation: str
    python_time: float
    c_time: float
    assembly_time: float
    data_size: int
    speedup: float
    memory_usage: Dict[str, float]

class AssemblyBenchmark:
    """Benchmark system for comparing implementations"""
    
    def __init__(self):
        # Load optimized libraries
        self.asm_lib = ctypes.CDLL('./libasm_ops.so')
        self.c_lib = ctypes.CDLL('./libc_ops.so')
        self._setup_functions()
        
    def _setup_functions(self):
        """Setup function signatures for different implementations"""
        # Vector operations
        self._setup_vector_functions()
        # String operations
        self._setup_string_functions()
        # Bit operations
        self._setup_bit_functions()
        
    def benchmark_all(self, sizes: List[int]) -> Dict[str, List[BenchmarkResult]]:
        """Run all benchmarks with different data sizes"""
        results = {}
        
        # Test different operations
        operations = [
            ('vector_similarity', self._benchmark_vector_similarity),
            ('string_search', self._benchmark_string_search),
            ('bit_manipulation', self._benchmark_bit_ops),
            ('pattern_matching', self._benchmark_pattern_matching),
            ('hash_computation', self._benchmark_hash_computation)
        ]
        
        for op_name, benchmark_func in operations:
            op_results = []
            for size in sizes:
                result = benchmark_func(size)
                op_results.append(result)
            results[op_name] = op_results
            
        return results

    def _benchmark_vector_similarity(self, size: int) -> BenchmarkResult:
        """Benchmark vector similarity calculations"""
        # Generate test data
        vec1 = np.random.random(size).astype(np.float32)
        vec2 = np.random.random(size).astype(np.float32)
        
        # Python implementation
        start_time = time.perf_counter()
        python_result = self._python_vector_similarity(vec1, vec2)
        python_time = time.perf_counter() - start_time
        
        # C implementation
        start_time = time.perf_counter()
        c_result = self.c_lib.vector_similarity(
            vec1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            vec2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            size
        )
        c_time = time.perf_counter() - start_time
        
        # Assembly implementation
        start_time = time.perf_counter()
        asm_result = self.asm_lib.vector_similarity_asm(
            vec1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            vec2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            size
        )
        asm_time = time.perf_counter() - start_time
        
        return BenchmarkResult(
            operation='vector_similarity',
            python_time=python_time,
            c_time=c_time,
            assembly_time=asm_time,
            data_size=size,
            speedup=python_time/asm_time,
            memory_usage=self._measure_memory_usage()
        )

    def _benchmark_string_search(self, size: int) -> BenchmarkResult:
        """Benchmark string search operations"""
        # Generate test data
        text = ''.join(np.random.choice(list('ACTG'), size))
        pattern = ''.join(np.random.choice(list('ACTG'), 10))
        
        # Python implementation
        start_time = time.perf_counter()
        python_result = text.count(pattern)
        python_time = time.perf_counter() - start_time
        
        # C implementation with KMP
        start_time = time.perf_counter()
        c_result = self.c_lib.kmp_search(
            text.encode('utf-8'),
            pattern.encode('utf-8'),
            len(text),
            len(pattern)
        )
        c_time = time.perf_counter() - start_time
        
        # Assembly implementation with SIMD
        start_time = time.perf_counter()
        asm_result = self.asm_lib.string_search_asm(
            text.encode('utf-8'),
            pattern.encode('utf-8'),
            len(text),
            len(pattern)
        )
        asm_time = time.perf_counter() - start_time
        
        return BenchmarkResult(
            operation='string_search',
            python_time=python_time,
            c_time=c_time,
            assembly_time=asm_time,
            data_size=size,
            speedup=python_time/asm_time,
            memory_usage=self._measure_memory_usage()
        )

    def visualize_results(self, results: Dict[str, List[BenchmarkResult]]):
        """Create visualizations of benchmark results"""
        plt.figure(figsize=(15, 10))
        
        # 1. Speedup comparison
        plt.subplot(2, 2, 1)
        self._plot_speedup_comparison(results)
        
        # 2. Memory usage
        plt.subplot(2, 2, 2)
        self._plot_memory_usage(results)
        
        # 3. Scaling behavior
        plt.subplot(2, 2, 3)
        self._plot_scaling_behavior(results)
        
        # 4. Operation breakdown
        plt.subplot(2, 2, 4)
        self._plot_operation_breakdown(results)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png')
        plt.close()

    def _plot_speedup_comparison(self, results: Dict[str, List[BenchmarkResult]]):
        """Plot speedup comparison across implementations"""
        operations = list(results.keys())
        speedups = [np.mean([r.speedup for r in results[op]]) for op in operations]
        
        plt.bar(operations, speedups)
        plt.title('Average Speedup vs Python Implementation')
        plt.xticks(rotation=45)
        plt.ylabel('Speedup Factor')

    def generate_report(self, results: Dict[str, List[BenchmarkResult]]) -> str:
        """Generate detailed benchmark report"""
        report = ["Assembly Optimization Benchmark Report\n"]
        
        # Overall summary
        report.append("\nOverall Performance Summary:")
        for operation, benchmarks in results.items():
            avg_speedup = np.mean([b.speedup for b in benchmarks])
            report.append(f"\n{operation}:")
            report.append(f"  Average Speedup: {avg_speedup:.2f}x")
            report.append(f"  Best Speedup: {max(b.speedup for b in benchmarks):.2f}x")
            report.append(f"  Worst Speedup: {min(b.speedup for b in benchmarks):.2f}x")
        
        # Detailed analysis
        report.append("\nDetailed Analysis by Operation:")
        for operation, benchmarks in results.items():
            report.append(f"\n{operation} Analysis:")
            
            # Performance by data size
            report.append("  Performance by Data Size:")
            for benchmark in benchmarks:
                report.append(
                    f"    Size: {benchmark.data_size:,} elements"
                    f"    Python: {benchmark.python_time*1000:.2f}ms"
                    f"    Assembly: {benchmark.assembly_time*1000:.2f}ms"
                    f"    Speedup: {benchmark.speedup:.2f}x"
                )
        
        return "\n".join(report)

# Assembly-optimizable processes
class AssemblyOptimizedProcesses:
    """Demonstration of processes that benefit from Assembly optimization"""
    
    @staticmethod
    def vector_operations():
        """Vector and matrix operations"""
        # Benefit: SIMD instructions can process multiple elements simultaneously
        assembly_code = """
        ; Process 16 floats at once with AVX-512
        vmovups zmm0, [rdi]        ; Load 16 floats
        vmovups zmm1, [rsi]        ; Load 16 floats
        vfmadd231ps zmm2, zmm0, zmm1 ; Multiply-add
        """
        return assembly_code
    
    @staticmethod
    def pattern_matching():
        """String and pattern matching"""
        # Benefit: SIMD string operations and special string instructions
        assembly_code = """
        ; Compare 16 bytes at once
        vmovdqu ymm0, [rdi]        ; Load string
        vpcmpeqb ymm1, ymm0, ymm2  ; Compare bytes
        vpmovmskb eax, ymm1        ; Get match mask
        """
        return assembly_code
    
    @staticmethod
    def bit_manipulation():
        """Bit manipulation operations"""
        # Benefit: Direct access to CPU bit manipulation instructions
        assembly_code = """
        ; Fast bit counting and manipulation
        popcnt rax, rdi    ; Count set bits
        bsf rax, rdi       ; Find first set bit
        blsr rax, rdi      ; Reset lowest set bit
        """
        return assembly_code
    
    @staticmethod
    def hash_computation():
        """Hash computation operations"""
        # Benefit: CPU-specific hash instructions and SIMD
        assembly_code = """
        ; Fast hash computation
        crc32 rax, rdi     ; Hardware CRC32
        aesenc xmm0, xmm1  ; AES round
        """
        return assembly_code

# Example usage
def main():
    # Initialize benchmark system
    benchmark = AssemblyBenchmark()
    
    # Run benchmarks with different data sizes
    sizes = [1000, 10000, 100000, 1000000]
    results = benchmark.benchmark_all(sizes)
    
    # Generate visualizations
    benchmark.visualize_results(results)
    
    # Generate report
    report = benchmark.generate_report(results)
    print(report)
    
    # Save report
    with open('benchmark_report.txt', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    main()