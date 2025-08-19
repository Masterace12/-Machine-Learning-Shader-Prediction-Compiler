#!/usr/bin/env python3
"""
Steam Deck Threading Configuration System
Comprehensive threading setup for all ML libraries and system components
"""

import os
import sys
import logging
import threading
import multiprocessing
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import weakref
import atexit

@dataclass
class ThreadingConfig:
    """Threading configuration for Steam Deck optimization"""
    # Core thread limits
    max_threads: int = 6                    # Total system threads
    ml_threads: int = 2                     # ML inference threads
    compilation_threads: int = 2            # Shader compilation threads
    io_threads: int = 1                     # I/O operations
    
    # OpenMP and BLAS configuration
    omp_num_threads: int = 2               # OpenMP threads
    mkl_num_threads: int = 2               # Intel MKL threads
    openblas_num_threads: int = 2          # OpenBLAS threads
    numexpr_num_threads: int = 2           # NumExpr threads
    
    # Framework-specific settings
    lightgbm_threads: int = 2              # LightGBM threads
    sklearn_threads: int = 2               # Scikit-learn threads
    numba_threads: int = 2                 # Numba threads
    
    # Thread stack and memory settings
    thread_stack_size: int = 262144        # 256KB stack size
    thread_memory_limit: int = 50          # 50MB per thread
    
    # Adaptive settings
    enable_thermal_scaling: bool = True    # Scale based on temperature
    enable_battery_scaling: bool = True    # Scale based on battery
    enable_gaming_mode: bool = True        # Detect and adapt to gaming
    
    # Debug and monitoring
    enable_thread_monitoring: bool = True
    log_thread_lifecycle: bool = False

class SteamDeckThreadingConfigurator:
    """
    Comprehensive threading configurator for Steam Deck
    
    Handles:
    - Environment variable setup
    - ML library configuration
    - System thread limits
    - Thermal-aware scaling
    - Gaming mode detection
    """
    
    def __init__(self, config: Optional[ThreadingConfig] = None):
        self.config = config or ThreadingConfig()
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self._configured_libraries = set()
        self._original_env = {}
        self._thread_monitors = weakref.WeakSet()
        self._is_configured = False
        
        # Hardware detection
        self._cpu_count = multiprocessing.cpu_count()
        self._is_steam_deck = self._detect_steam_deck()
        
        # Adaptive state
        self._current_thermal_state = "normal"
        self._gaming_mode_active = False
        self._battery_level = 100.0
        
        # Register cleanup
        atexit.register(self.cleanup)
        
        self.logger.info(f"Threading configurator initialized for {self._cpu_count}-core system")
        if self._is_steam_deck:
            self.logger.info("Steam Deck detected - applying optimized configuration")
    
    def _detect_steam_deck(self) -> bool:
        """Detect if running on Steam Deck"""
        # Check DMI information
        try:
            dmi_path = Path("/sys/class/dmi/id/product_name")
            if dmi_path.exists():
                product_name = dmi_path.read_text().strip().lower()
                if "jupiter" in product_name or "steamdeck" in product_name:
                    return True
        except Exception:
            pass
        
        # Check for Steam Deck specific paths
        steam_deck_indicators = [
            Path("/home/deck"),
            Path("/sys/class/power_supply/BAT1"),  # Steam Deck battery
            Path("/dev/input/by-path/platform-i8042-serio-0-event-kbd"),  # Steam Deck keyboard
        ]
        
        return any(path.exists() for path in steam_deck_indicators)
    
    def configure_environment_variables(self):
        """Configure threading environment variables"""
        if self._is_configured:
            return
        
        env_vars = {
            # OpenMP configuration
            'OMP_NUM_THREADS': str(self.config.omp_num_threads),
            'OMP_THREAD_LIMIT': str(self.config.max_threads),
            'OMP_DYNAMIC': 'TRUE',  # Allow dynamic thread adjustment
            'OMP_NESTED': 'FALSE',  # Disable nested parallelism
            'OMP_WAIT_POLICY': 'PASSIVE',  # Reduce CPU spinning
            'OMP_PROC_BIND': 'TRUE',  # Bind threads to cores
            
            # Intel MKL configuration
            'MKL_NUM_THREADS': str(self.config.mkl_num_threads),
            'MKL_DYNAMIC': 'TRUE',
            'MKL_THREADING_LAYER': 'INTEL',
            
            # OpenBLAS configuration
            'OPENBLAS_NUM_THREADS': str(self.config.openblas_num_threads),
            'OPENBLAS_CORETYPE': 'ZENVER2',  # Optimized for Zen 2 (Steam Deck APU)
            
            # NumExpr configuration
            'NUMEXPR_NUM_THREADS': str(self.config.numexpr_num_threads),
            'NUMEXPR_MAX_THREADS': str(self.config.numexpr_num_threads),
            
            # TensorFlow (if used)
            'TF_NUM_INTRAOP_THREADS': str(self.config.ml_threads),
            'TF_NUM_INTEROP_THREADS': str(1),
            
            # PyTorch (if used)
            'OMP_SCHEDULE': 'STATIC',
            
            # General performance
            'MALLOC_ARENA_MAX': '2',  # Reduce memory fragmentation
            'PYTHONHASHSEED': '0',    # Reproducible hashing
        }
        
        # Apply environment variables
        for key, value in env_vars.items():
            if key not in os.environ:  # Don't override existing settings
                self._original_env[key] = os.environ.get(key)
                os.environ[key] = value
                self.logger.debug(f"Set {key}={value}")
        
        self.logger.info("Environment variables configured for Steam Deck threading")
    
    def configure_python_threading(self):
        """Configure Python's threading module"""
        try:
            # Set thread stack size (must be done before threads are created)
            if threading.stack_size() == 0:  # Only set if not already set
                threading.stack_size(self.config.thread_stack_size)
                self.logger.debug(f"Set thread stack size to {self.config.thread_stack_size} bytes")
        except Exception as e:
            self.logger.warning(f"Could not set thread stack size: {e}")
        
        # Configure thread switching
        try:
            sys.setswitchinterval(0.001)  # Faster thread switching for responsiveness
        except Exception as e:
            self.logger.warning(f"Could not set switch interval: {e}")
    
    def configure_lightgbm(self):
        """Configure LightGBM threading with Steam Deck optimizations"""
        try:
            import lightgbm as lgb
            
            # Configure for Steam Deck APU with conservative memory usage
            default_params = {
                'num_threads': self.config.lightgbm_threads,
                'force_row_wise': True,  # Better memory access pattern
                'device_type': 'cpu',
                'max_bin': 255,  # Reduce memory usage
                'verbose': -1,
                'histogram_pool_size': 128,  # Limit memory pool
                'enable_sparse': True,       # Optimize for sparse features
                'bagging_freq': 1,          # Enable bagging for stability
                'feature_fraction': 0.8,    # Use subset of features
            }
            
            # Store default params for easy access
            if not hasattr(lgb, '_steamdeck_default_params'):
                lgb._steamdeck_default_params = default_params
            
            # Modern LightGBM versions don't have set_number_threads
            # Use environment variables instead for better compatibility
            os.environ['LIGHTGBM_NUM_THREADS'] = str(self.config.lightgbm_threads)
            
            # Try parameter-based configuration (newer API)
            try:
                if hasattr(lgb, 'set_option'):
                    # Use new parameter setting API
                    lgb.set_option('num_threads', self.config.lightgbm_threads)
                    lgb.set_option('force_row_wise', True)
                    lgb.set_option('histogram_pool_size', 128)
                    self.logger.info("Configured LightGBM with parameter API")
                else:
                    self.logger.info("LightGBM threading controlled via environment variables")
            except Exception as param_error:
                self.logger.debug(f"Could not set LightGBM global threads: {e}")
            
            self._configured_libraries.add('lightgbm')
            self.logger.info(f"Configured LightGBM with {self.config.lightgbm_threads} threads")
            
        except ImportError:
            self.logger.debug("LightGBM not available")
        except Exception as e:
            self.logger.error(f"Failed to configure LightGBM: {e}")
    
    def configure_sklearn(self):
        """Configure scikit-learn threading"""
        try:
            from sklearn import set_config
            from sklearn.utils import parallel_backend
            
            # Set global config
            set_config(
                assume_finite=True,  # Skip input validation for performance
                working_memory=128,  # Limit memory usage
            )
            
            # Configure joblib backend
            try:
                import joblib
                joblib.parallel_backend('threading', n_jobs=self.config.sklearn_threads)
            except ImportError:
                pass
            
            self._configured_libraries.add('sklearn')
            self.logger.info(f"Configured scikit-learn with {self.config.sklearn_threads} threads")
            
        except ImportError:
            self.logger.debug("scikit-learn not available")
        except Exception as e:
            self.logger.error(f"Failed to configure scikit-learn: {e}")
    
    def configure_numba(self):
        """Configure Numba JIT threading"""
        try:
            import numba
            
            # Set number of threads
            numba.set_num_threads(self.config.numba_threads)
            
            # Configure for AMD Zen 2 (Steam Deck APU)
            numba.config.THREADING_LAYER = 'workqueue'
            numba.config.NUMBA_NUM_THREADS = self.config.numba_threads
            
            self._configured_libraries.add('numba')
            self.logger.info(f"Configured Numba with {self.config.numba_threads} threads")
            
        except ImportError:
            self.logger.debug("Numba not available")
        except Exception as e:
            self.logger.error(f"Failed to configure Numba: {e}")
    
    def configure_numpy(self):
        """Configure NumPy threading"""
        try:
            import numpy as np
            
            # Try to configure BLAS threads through NumPy
            if hasattr(np, '__config__') and hasattr(np.__config__, 'show'):
                config_info = np.__config__.show()
                self.logger.debug(f"NumPy build configuration: {config_info}")
            
            self._configured_libraries.add('numpy')
            self.logger.info("NumPy threading configured via environment variables")
            
        except ImportError:
            self.logger.debug("NumPy not available")
        except Exception as e:
            self.logger.error(f"Failed to configure NumPy: {e}")
    
    def configure_all_libraries(self):
        """Configure all available ML libraries"""
        self.configure_environment_variables()
        self.configure_python_threading()
        
        # Configure individual libraries
        self.configure_lightgbm()
        self.configure_sklearn()
        self.configure_numba()
        self.configure_numpy()
        
        self._is_configured = True
        
        configured_count = len(self._configured_libraries)
        self.logger.info(f"Configured {configured_count} ML libraries for Steam Deck threading")
    
    def get_optimal_params_lightgbm(self) -> Dict[str, Any]:
        """Get optimal LightGBM parameters for Steam Deck"""
        base_params = {
            'num_threads': self.config.lightgbm_threads,
            'force_row_wise': True,
            'device_type': 'cpu',
            'max_bin': 255,
            'verbose': -1,
        }
        
        # Adaptive parameters based on system state
        if self._current_thermal_state in ['hot', 'critical']:
            base_params.update({
                'num_threads': max(1, self.config.lightgbm_threads // 2),
                'max_depth': min(6, base_params.get('max_depth', 6)),
            })
        
        if self._gaming_mode_active:
            base_params.update({
                'num_threads': 1,  # Minimal threading during gaming
            })
        
        return base_params
    
    def get_optimal_params_sklearn(self) -> Dict[str, Any]:
        """Get optimal scikit-learn parameters for Steam Deck"""
        n_jobs = self.config.sklearn_threads
        
        # Adaptive threading
        if self._current_thermal_state in ['hot', 'critical']:
            n_jobs = max(1, n_jobs // 2)
        
        if self._gaming_mode_active:
            n_jobs = 1
        
        return {
            'n_jobs': n_jobs,
            'max_iter': 100,  # Limit iterations for responsiveness
        }
    
    def update_thermal_state(self, thermal_state: str):
        """Update thermal state and adjust threading accordingly"""
        if thermal_state != self._current_thermal_state:
            self.logger.info(f"Thermal state changed: {self._current_thermal_state} -> {thermal_state}")
            self._current_thermal_state = thermal_state
            
            if self.config.enable_thermal_scaling:
                self._apply_thermal_scaling()
    
    def update_gaming_mode(self, gaming_active: bool):
        """Update gaming mode state"""
        if gaming_active != self._gaming_mode_active:
            self.logger.info(f"Gaming mode: {self._gaming_mode_active} -> {gaming_active}")
            self._gaming_mode_active = gaming_active
            
            if self.config.enable_gaming_mode:
                self._apply_gaming_mode_scaling()
    
    def update_battery_level(self, battery_percent: float):
        """Update battery level and adjust threading"""
        self._battery_level = battery_percent
        
        if self.config.enable_battery_scaling and battery_percent < 20.0:
            self._apply_battery_scaling()
    
    def _apply_thermal_scaling(self):
        """Apply thermal-based thread scaling"""
        if self._current_thermal_state in ['hot', 'critical']:
            # Reduce threading for thermal management
            scale_factor = 0.5 if self._current_thermal_state == 'hot' else 0.25
            
            # Update LightGBM if available
            if 'lightgbm' in self._configured_libraries:
                try:
                    import lightgbm as lgb
                    new_threads = max(1, int(self.config.lightgbm_threads * scale_factor))
                    lgb.set_number_threads(new_threads)
                    self.logger.info(f"Scaled LightGBM threads to {new_threads} for thermal management")
                except Exception as e:
                    self.logger.error(f"Failed to scale LightGBM threads: {e}")
    
    def _apply_gaming_mode_scaling(self):
        """Apply gaming mode thread scaling"""
        if self._gaming_mode_active:
            # Minimal threading during gaming
            if 'lightgbm' in self._configured_libraries:
                try:
                    import lightgbm as lgb
                    lgb.set_number_threads(1)
                    self.logger.info("Reduced LightGBM threads for gaming mode")
                except Exception:
                    pass
    
    def _apply_battery_scaling(self):
        """Apply battery-based thread scaling"""
        if self._battery_level < 20.0:
            # Reduce threading to save battery
            scale_factor = 0.5
            
            if 'lightgbm' in self._configured_libraries:
                try:
                    import lightgbm as lgb
                    new_threads = max(1, int(self.config.lightgbm_threads * scale_factor))
                    lgb.set_number_threads(new_threads)
                    self.logger.info(f"Scaled LightGBM threads to {new_threads} for battery saving")
                except Exception:
                    pass
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get current configuration status"""
        return {
            'is_configured': self._is_configured,
            'is_steam_deck': self._is_steam_deck,
            'cpu_count': self._cpu_count,
            'configured_libraries': list(self._configured_libraries),
            'config': {
                'max_threads': self.config.max_threads,
                'ml_threads': self.config.ml_threads,
                'omp_threads': self.config.omp_num_threads,
            },
            'adaptive_state': {
                'thermal_state': self._current_thermal_state,
                'gaming_mode': self._gaming_mode_active,
                'battery_level': self._battery_level,
            },
            'environment_variables': {
                key: os.environ.get(key, 'not set')
                for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']
            }
        }
    
    def cleanup(self):
        """Cleanup and restore original environment"""
        if self._original_env:
            for key, original_value in self._original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
            
            self.logger.debug("Restored original environment variables")


# Global configurator instance
_global_configurator = None


def get_threading_configurator() -> SteamDeckThreadingConfigurator:
    """Get or create global threading configurator"""
    global _global_configurator
    if _global_configurator is None:
        _global_configurator = SteamDeckThreadingConfigurator()
    return _global_configurator


def configure_threading_for_steam_deck(config: Optional[ThreadingConfig] = None) -> SteamDeckThreadingConfigurator:
    """Configure threading for Steam Deck (main entry point)"""
    configurator = SteamDeckThreadingConfigurator(config)
    configurator.configure_all_libraries()
    return configurator


def get_lightgbm_params() -> Dict[str, Any]:
    """Get optimized LightGBM parameters for current system state"""
    configurator = get_threading_configurator()
    return configurator.get_optimal_params_lightgbm()


def get_sklearn_params() -> Dict[str, Any]:
    """Get optimized scikit-learn parameters for current system state"""
    configurator = get_threading_configurator()
    return configurator.get_optimal_params_sklearn()


if __name__ == "__main__":
    # Test threading configuration
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 Steam Deck Threading Configuration Test")
    print("=" * 50)
    
    # Configure threading
    configurator = configure_threading_for_steam_deck()
    
    # Show status
    status = configurator.get_configuration_status()
    print(f"Steam Deck detected: {status['is_steam_deck']}")
    print(f"CPU cores: {status['cpu_count']}")
    print(f"Configured libraries: {', '.join(status['configured_libraries'])}")
    print(f"Max threads: {status['config']['max_threads']}")
    
    # Test adaptive parameters
    print("\nOptimized Parameters:")
    lgb_params = get_lightgbm_params()
    print(f"LightGBM threads: {lgb_params.get('num_threads', 'N/A')}")
    
    sklearn_params = get_sklearn_params()
    print(f"scikit-learn n_jobs: {sklearn_params.get('n_jobs', 'N/A')}")
    
    # Test thermal scaling
    print("\nTesting thermal scaling...")
    configurator.update_thermal_state('hot')
    hot_params = get_lightgbm_params()
    print(f"LightGBM threads (hot): {hot_params.get('num_threads', 'N/A')}")
    
    configurator.update_thermal_state('normal')
    
    print("\n✅ Threading configuration test completed")