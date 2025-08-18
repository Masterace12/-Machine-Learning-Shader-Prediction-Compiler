#!/usr/bin/env python3
"""
SIMD Optimizations for Vector Math Operations
Provides hardware-accelerated vector operations for ML feature processing
"""

try:
    import numpy as np
    HAS_NUMPY = True
    # Check NumPy version for compatibility
    NUMPY_VERSION = tuple(map(int, np.__version__.split('.')[:2]))
    HAS_NUMPY_2 = NUMPY_VERSION >= (2, 0)
except ImportError:
    HAS_NUMPY = False
    HAS_NUMPY_2 = False
    np = None
    NUMPY_VERSION = (0, 0)
    
from typing import Optional, Union, List
import logging

# Check for SIMD support
try:
    import numba
    from numba import jit, vectorize, float32, float64, int32, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    numba = None

# Check for Intel MKL
try:
    import mkl
    HAS_MKL = True
    mkl.set_num_threads(2)  # Limit threads for Steam Deck
except ImportError:
    HAS_MKL = False

logger = logging.getLogger(__name__)


class SIMDVectorOps:
    """SIMD-optimized vector operations for ML features"""
    
    def __init__(self):
        """Initialize SIMD operations"""
        self.use_simd = HAS_NUMBA
        self.use_mkl = HAS_MKL
        
        if self.use_simd:
            # Pre-compile JIT functions
            self._compile_functions()
            logger.info("SIMD optimizations enabled via Numba")
        elif self.use_mkl:
            logger.info("Using Intel MKL for optimized operations")
        else:
            logger.info("Using NumPy for vector operations")
    
    def _compile_functions(self):
        """Pre-compile Numba JIT functions"""
        if not HAS_NUMBA:
            return
        
        # Warm up JIT compilation with dummy data
        dummy = np.ones(12, dtype=np.float32)
        _ = self.dot_product_simd(dummy, dummy)
        _ = self.normalize_simd(dummy)
        _ = self.scale_features_simd(dummy, 2.0)
    
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=False) if HAS_NUMBA else lambda f: f
    def dot_product_simd(a: np.ndarray, b: np.ndarray) -> float:
        """SIMD-optimized dot product"""
        result = 0.0
        for i in range(len(a)):
            result += a[i] * b[i]
        return result
    
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=False) if HAS_NUMBA else lambda f: f
    def normalize_simd(vector: np.ndarray) -> np.ndarray:
        """SIMD-optimized vector normalization"""
        sum_sq = 0.0
        for i in range(len(vector)):
            sum_sq += vector[i] * vector[i]
        
        if sum_sq > 0:
            norm = np.sqrt(sum_sq)
            for i in range(len(vector)):
                vector[i] /= norm
        return vector
    
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=False) if HAS_NUMBA else lambda f: f
    def scale_features_simd(features: np.ndarray, scale: float) -> np.ndarray:
        """SIMD-optimized feature scaling"""
        for i in range(len(features)):
            features[i] *= scale
        return features
    
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True) if HAS_NUMBA else lambda f: f
    def batch_process_features(batch: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Process batch of features with SIMD parallelization"""
        n_samples, n_features = batch.shape
        result = np.empty(n_samples, dtype=np.float32)
        
        for i in prange(n_samples):
            sum_val = 0.0
            for j in range(n_features):
                sum_val += batch[i, j] * weights[j]
            result[i] = sum_val
        
        return result
    
    @staticmethod
    @vectorize([float32(float32, float32), float64(float64, float64)], 
               target='parallel' if HAS_NUMBA else 'cpu') if HAS_NUMBA else lambda f: f
    def element_wise_multiply(a, b):
        """SIMD element-wise multiplication"""
        return a * b
    
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity with SIMD optimization"""
        if self.use_simd:
            # Use SIMD-optimized operations
            dot = self.dot_product_simd(vec1, vec2)
            norm1 = np.sqrt(self.dot_product_simd(vec1, vec1))
            norm2 = np.sqrt(self.dot_product_simd(vec2, vec2))
        else:
            # Fallback to NumPy
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
        
        if norm1 * norm2 > 0:
            return dot / (norm1 * norm2)
        return 0.0
    
    def fast_matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fast matrix multiplication using available optimizations"""
        if self.use_mkl:
            # MKL will automatically use optimized routines
            return np.dot(a, b)
        elif self.use_simd and HAS_NUMBA:
            return self._numba_matmul(a, b)
        else:
            # NumPy's built-in is already reasonably optimized
            return np.dot(a, b)
    
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True) if HAS_NUMBA else lambda f: f
    def _numba_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Numba-optimized matrix multiplication"""
        m, k = a.shape
        k2, n = b.shape
        assert k == k2, "Matrix dimensions don't match"
        
        c = np.zeros((m, n), dtype=a.dtype)
        
        for i in prange(m):
            for j in range(n):
                sum_val = 0.0
                for l in range(k):
                    sum_val += a[i, l] * b[l, j]
                c[i, j] = sum_val
        
        return c
    
    def transform_features_batch(self, features: np.ndarray, 
                                transform_matrix: np.ndarray) -> np.ndarray:
        """Transform feature batch with SIMD optimizations"""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        return self.fast_matrix_multiply(features, transform_matrix)


# Global instance for easy access
_simd_ops = None


def get_simd_ops() -> SIMDVectorOps:
    """Get or create global SIMD operations instance"""
    global _simd_ops
    if _simd_ops is None:
        _simd_ops = SIMDVectorOps()
    return _simd_ops


if __name__ == "__main__":
    # Test SIMD optimizations
    import timeit
    
    print(f"SIMD Support: Numba={HAS_NUMBA}, MKL={HAS_MKL}")
    
    ops = get_simd_ops()
    
    # Create test data
    size = 1000
    
    # NumPy 2.x compatible random generation
    if HAS_NUMPY_2:
        rng = np.random.default_rng(42)
        vec1 = rng.standard_normal(size, dtype=np.float32)
        vec2 = rng.standard_normal(size, dtype=np.float32)
        matrix = rng.standard_normal((size, 100), dtype=np.float32)
    else:
        # NumPy 1.x legacy API
        np.random.seed(42)
        vec1 = np.random.randn(size).astype(np.float32)
        vec2 = np.random.randn(size).astype(np.float32)
        matrix = np.random.randn(size, 100).astype(np.float32)
    
    # Benchmark operations
    def benchmark_dot():
        return ops.dot_product_simd(vec1, vec2) if HAS_NUMBA else np.dot(vec1, vec2)
    
    def benchmark_normalize():
        return ops.normalize_simd(vec1.copy()) if HAS_NUMBA else vec1 / np.linalg.norm(vec1)
    
    def benchmark_matmul():
        return ops.fast_matrix_multiply(matrix.T, matrix)
    
    # Run benchmarks
    n_iterations = 1000
    
    dot_time = timeit.timeit(benchmark_dot, number=n_iterations) / n_iterations * 1000
    print(f"Dot product: {dot_time:.3f}ms")
    
    norm_time = timeit.timeit(benchmark_normalize, number=n_iterations) / n_iterations * 1000
    print(f"Normalization: {norm_time:.3f}ms")
    
    matmul_time = timeit.timeit(benchmark_matmul, number=10) / 10 * 1000
    print(f"Matrix multiply ({size}x100 @ 100x{size}): {matmul_time:.3f}ms")
    
    # Test similarity computation
    if HAS_NUMPY_2:
        rng = np.random.default_rng(123)
        features1 = rng.standard_normal(12, dtype=np.float32)
        features2 = rng.standard_normal(12, dtype=np.float32)
    else:
        np.random.seed(123)
        features1 = np.random.randn(12).astype(np.float32)
        features2 = np.random.randn(12).astype(np.float32)
    
    similarity = ops.compute_similarity(features1, features2)
    print(f"\nCosine similarity: {similarity:.4f}")