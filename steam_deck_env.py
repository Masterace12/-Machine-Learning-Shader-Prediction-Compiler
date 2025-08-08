#!/usr/bin/env python3
"""
Steam Deck Environment Configuration for ML Shader Prediction Compiler
Optimizes Python environment for Steam Deck's AMD APU and thermal constraints
"""

import os
import sys
import threading
import numpy as np
import psutil
from pathlib import Path

class SteamDeckOptimizer:
    """Configure Python environment for optimal Steam Deck performance"""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count(logical=True)
        self.physical_cpu_count = psutil.cpu_count(logical=False)
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
    def configure_numpy_threads(self):
        """Configure NumPy threading for Steam Deck APU"""
        # Steam Deck has 8 logical cores (4 physical + 4 SMT)
        # Limit NumPy threads to prevent thermal throttling
        max_threads = min(4, self.physical_cpu_count)
        
        # Set OpenBLAS thread limit
        os.environ['OPENBLAS_NUM_THREADS'] = str(max_threads)
        os.environ['MKL_NUM_THREADS'] = str(max_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(max_threads)
        os.environ['OMP_NUM_THREADS'] = str(max_threads)
        
        print(f"Configured NumPy/BLAS threads: {max_threads}")
        
    def configure_sklearn_threads(self):
        """Configure scikit-learn threading"""
        # Limit sklearn parallel jobs to prevent oversubscription
        max_jobs = min(2, self.physical_cpu_count // 2)
        os.environ['SKLEARN_N_JOBS'] = str(max_jobs)
        print(f"Configured sklearn threads: {max_jobs}")
        
    def configure_memory_limits(self):
        """Configure memory usage limits for ML operations"""
        # Reserve 2GB for system and Steam client
        available_memory_gb = max(1, self.memory_gb - 2)
        
        # Configure Pandas memory usage
        import pandas as pd
        pd.set_option('compute.use_numba', False)  # Disable numba compilation
        
        print(f"Available memory for ML operations: {available_memory_gb:.1f}GB")
        return available_memory_gb
        
    def check_thermal_status(self):
        """Monitor thermal status and adjust performance if needed"""
        try:
            # Try to read thermal zone temperatures
            thermal_paths = list(Path('/sys/class/thermal').glob('thermal_zone*/temp'))
            if thermal_paths:
                temps = []
                for path in thermal_paths:
                    try:
                        with open(path) as f:
                            temp = int(f.read().strip()) / 1000.0
                            temps.append(temp)
                    except (IOError, ValueError):
                        continue
                
                if temps:
                    max_temp = max(temps)
                    print(f"Current thermal zone temperatures: {temps}")
                    print(f"Maximum temperature: {max_temp:.1f}°C")
                    
                    # Warning if temperature is high
                    if max_temp > 75.0:
                        print("WARNING: High thermal load detected. Consider reducing ML workload.")
                        return True
                        
            return False
        except Exception as e:
            print(f"Unable to read thermal status: {e}")
            return False
            
    def optimize_for_gaming_mode(self):
        """Apply optimizations for when running alongside games"""
        # Reduce thread priority for ML operations
        try:
            # Set lower CPU priority
            p = psutil.Process()
            p.nice(10)  # Lower priority
            print("Applied gaming mode optimizations (lower CPU priority)")
        except Exception as e:
            print(f"Could not set process priority: {e}")
            
    def configure_environment(self, gaming_mode=False):
        """Apply all Steam Deck optimizations"""
        print("=== Steam Deck ML Environment Configuration ===")
        print(f"System: AMD Custom APU with {self.cpu_count} logical cores")
        print(f"Physical cores: {self.physical_cpu_count}")
        print(f"Total RAM: {self.memory_gb:.1f}GB")
        print()
        
        # Apply configurations
        self.configure_numpy_threads()
        self.configure_sklearn_threads()
        available_memory = self.configure_memory_limits()
        
        # Check thermal status
        thermal_warning = self.check_thermal_status()
        
        if gaming_mode:
            self.optimize_for_gaming_mode()
            
        print()
        print("=== Configuration Complete ===")
        if thermal_warning:
            print("⚠️  High temperature detected - monitor thermal performance")
        print("✅ Environment optimized for Steam Deck")
        
        return {
            'cpu_threads': min(4, self.physical_cpu_count),
            'sklearn_jobs': min(2, self.physical_cpu_count // 2),
            'available_memory_gb': available_memory,
            'thermal_warning': thermal_warning
        }

def initialize_steam_deck_environment(gaming_mode=False):
    """Initialize and configure the Steam Deck environment"""
    optimizer = SteamDeckOptimizer()
    return optimizer.configure_environment(gaming_mode=gaming_mode)

if __name__ == "__main__":
    # Test the configuration
    gaming_mode = '--gaming-mode' in sys.argv
    config = initialize_steam_deck_environment(gaming_mode=gaming_mode)
    
    # Test ML package imports
    print("\n=== Package Import Test ===")
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        import scipy
        print("✅ All ML packages imported successfully")
        
        # Quick performance test
        print("\n=== Quick Performance Test ===")
        import time
        
        # NumPy matrix operation test
        start_time = time.time()
        a = np.random.rand(1000, 1000)
        b = np.random.rand(1000, 1000)
        c = np.dot(a, b)
        numpy_time = time.time() - start_time
        print(f"NumPy 1000x1000 matrix multiply: {numpy_time:.3f}s")
        
        # Sklearn test
        start_time = time.time()
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        sklearn_time = time.time() - start_time
        print(f"Sklearn RandomForest training (1000 samples): {sklearn_time:.3f}s")
        
    except ImportError as e:
        print(f"❌ Package import failed: {e}")